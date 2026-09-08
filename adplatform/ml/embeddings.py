# embeddings.py — learned ad and placement vectors.
#
# The other 17 features are request attributes or counts. The counts
# (ad_ctr_prior, placement_ctr_prior, pair_ctr_prior) are beta-smoothed click
# rates, and they do not generalise across entities: pair_ctr_prior for
# (ad_9134, plc_4) uses the impressions of that pair and nothing else. With
# thousands of ads and hundreds of placements most pairs are thin, so the prior
# shrinks them back to the global CTR. Nothing tells the model that ad_9134
# resembles the other outdoor-gear ads that do have data on plc_4.
#
# So: a dense vector per ad and per placement, fit by gradient descent on the
# click labels under a factorisation machine.
#
#     logit(click) = b + b_ad + b_placement + <e_ad, e_placement>
#
# The inner product makes that a low-rank model of the pair matrix. An ad with
# fifty impressions borrows from ads sitting near it in the space, whether or
# not they ever ran on the same placement.
#
# This is not the CTR model. It is fit first, on the training split, and six
# summary columns go into XGBoost. Trees split axis-aligned and cannot learn a
# dot product over sparse ids; the FM knows nothing about keywords or pacing.
# The promotion gates in train_ctr.py judge the combination.

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

# The block extract_features appends. Six columns whatever `dim` is. Raw latent
# components are useless to a tree, since no single coordinate of an
# arbitrarily-rotated embedding means anything by itself, so we hand over the
# FM's summary of the pair plus enough context to learn when to distrust it.
EMBEDDING_FEATURE_NAMES: tuple[str, ...] = (
    "emb_logit",             # b + b_ad + b_pl + <e_ad, e_pl> — the FM's own call
    "emb_affinity",          # <e_ad, e_pl> alone: the part counts cannot express
    "emb_ad_bias",           # learned ad quality, free of placement mix
    "emb_placement_bias",    # learned placement quality, free of ad mix
    "emb_ad_norm",           # ||e_ad||; ~0 for an ad the fit barely saw
    "emb_is_cold",           # 1.0 when either id fell back to the OOV row
)

EMBEDDING_BLOCK_SIZE = len(EMBEDDING_FEATURE_NAMES)

# What extract_features appends with no table loaded: a cold start, or a model
# trained with --no-embeddings. Constant columns the trees ignore, and the flag
# keeps them from reading as an FM score that happens to be 0.0.
NEUTRAL_EMBEDDING_BLOCK: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

# Row 0 of both tables is the out-of-vocabulary bucket, and it is trained rather
# than zero-padded: every id too rare for its own vector routes here during the
# fit, so it ends up describing the average rare ad. Better cold-start
# behaviour than the origin.
OOV_INDEX = 0

DEFAULT_DIM = 8
DEFAULT_MIN_COUNT = 20
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LR = 0.05
DEFAULT_L2 = 1e-5
DEFAULT_HOLDOUT_FRAC = 0.1

EMBEDDING_FILE = "embeddings.npz"
EMBEDDING_FORMAT_VERSION = 1


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # The naive form overflows at |z| > 700 and returns nan where it should
    # return 0 or 1. Branching on the sign keeps exp()'s argument negative.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@dataclass(frozen=True, eq=False)
class EmbeddingTable:
    """
    A fitted table. The only thing that turns ids into embedding features.

    Training and serving both go through `block()`. train_ctr.py recomputes the
    block for every historical row from the table it is about to ship; the
    gateway computes it per request from the table it loaded out of the same
    artifact directory. Same function, same table, same numbers.

    Frozen because it is shared across request threads. A reload replaces the
    reference wholesale; nothing mutates a live table.
    """

    dim: int
    ad_vocab: dict[str, int]
    placement_vocab: dict[str, int]
    ad_vectors: np.ndarray            # (n_ads + 1, dim), row 0 = OOV
    placement_vectors: np.ndarray     # (n_placements + 1, dim), row 0 = OOV
    ad_bias: np.ndarray               # (n_ads + 1,)
    placement_bias: np.ndarray        # (n_placements + 1,)
    global_bias: float
    version: str = "unversioned"
    trained_at: str = ""
    fit_report: dict = field(default_factory=dict)

    # -- inference ----------------------------------------------------------

    def block(self, ad_id: str, placement_id: str) -> list[float]:
        """
        The six embedding features for one (ad, placement) pair.

        Two dict lookups, two row reads and a `dim`-length dot product per
        eligible ad. At dim=8 that is tens of nanoseconds, cheap enough to sit
        inside extract_features without showing up in bid latency.
        """
        ia = self.ad_vocab.get(ad_id, OOV_INDEX)
        ip = self.placement_vocab.get(placement_id, OOV_INDEX)

        e_ad = self.ad_vectors[ia]
        e_pl = self.placement_vectors[ip]

        affinity = float(np.dot(e_ad, e_pl))
        b_ad = float(self.ad_bias[ia])
        b_pl = float(self.placement_bias[ip])

        return [
            self.global_bias + b_ad + b_pl + affinity,
            affinity,
            b_ad,
            b_pl,
            float(np.linalg.norm(e_ad)),
            1.0 if (ia == OOV_INDEX or ip == OOV_INDEX) else 0.0,
        ]

    def blocks_for_rows(
        self, ad_ids: Sequence[str], placement_ids: Sequence[str]
    ) -> np.ndarray:
        """
        The training-side entry point: one block per labelled row, where row i
        pairs ad_ids[i] with placement_ids[i].
        """
        n = len(ad_ids)
        if n != len(placement_ids):
            raise ValueError(
                f"ad_ids and placement_ids differ in length: {n} vs {len(placement_ids)}"
            )
        if n == 0:
            return np.zeros((0, EMBEDDING_BLOCK_SIZE), dtype=np.float32)

        ia = np.fromiter(
            (self.ad_vocab.get(a, OOV_INDEX) for a in ad_ids), dtype=np.int64, count=n
        )
        ip = np.fromiter(
            (self.placement_vocab.get(p, OOV_INDEX) for p in placement_ids),
            dtype=np.int64, count=n,
        )
        return self._blocks_by_index(ia, ip)

    def _blocks_by_index(self, ia: np.ndarray, ip: np.ndarray) -> np.ndarray:
        e_ad = self.ad_vectors[ia]
        e_pl = self.placement_vectors[ip]
        affinity = np.einsum("ij,ij->i", e_ad, e_pl)
        b_ad = self.ad_bias[ia]
        b_pl = self.placement_bias[ip]

        out = np.empty((len(ia), EMBEDDING_BLOCK_SIZE), dtype=np.float32)
        out[:, 0] = self.global_bias + b_ad + b_pl + affinity
        out[:, 1] = affinity
        out[:, 2] = b_ad
        out[:, 3] = b_pl
        out[:, 4] = np.linalg.norm(e_ad, axis=1)
        out[:, 5] = ((ia == OOV_INDEX) | (ip == OOV_INDEX)).astype(np.float32)
        return out

    def predict_proba(
        self, ad_ids: Sequence[str], placement_ids: Sequence[str]
    ) -> np.ndarray:
        """The FM's own CTR estimate. Reported in the training metadata to show
        what the table is worth alone. XGBoost owns the served prediction."""
        return _sigmoid(self.blocks_for_rows(ad_ids, placement_ids)[:, 0].astype(np.float64))

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """
        One .npz holding the arrays and the vocabularies. The vocabularies are
        stored as a JSON string rather than an object array so the file loads
        with allow_pickle=False. An artifact fetched from S3 is deserialised
        inside the serving process, where pickle would turn a bucket write into
        arbitrary code execution.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = {
            "format_version": EMBEDDING_FORMAT_VERSION,
            "dim": self.dim,
            "version": self.version,
            "trained_at": self.trained_at,
            "ad_vocab": self.ad_vocab,
            "placement_vocab": self.placement_vocab,
            "fit_report": self.fit_report,
        }
        np.savez(
            path,
            ad_vectors=self.ad_vectors.astype(np.float32),
            placement_vectors=self.placement_vectors.astype(np.float32),
            ad_bias=self.ad_bias.astype(np.float32),
            placement_bias=self.placement_bias.astype(np.float32),
            global_bias=np.asarray(self.global_bias, dtype=np.float32),
            sidecar=np.asarray(json.dumps(sidecar)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingTable":
        with np.load(Path(path), allow_pickle=False) as data:
            sidecar = json.loads(str(data["sidecar"]))
            fmt = sidecar.get("format_version")
            if fmt != EMBEDDING_FORMAT_VERSION:
                raise ValueError(
                    f"embedding table at {path} is format version {fmt}, "
                    f"this code reads {EMBEDDING_FORMAT_VERSION}"
                )
            table = cls(
                dim=int(sidecar["dim"]),
                ad_vocab={k: int(v) for k, v in sidecar["ad_vocab"].items()},
                placement_vocab={
                    k: int(v) for k, v in sidecar["placement_vocab"].items()
                },
                ad_vectors=np.asarray(data["ad_vectors"], dtype=np.float32),
                placement_vectors=np.asarray(
                    data["placement_vectors"], dtype=np.float32
                ),
                ad_bias=np.asarray(data["ad_bias"], dtype=np.float32),
                placement_bias=np.asarray(data["placement_bias"], dtype=np.float32),
                global_bias=float(data["global_bias"]),
                version=str(sidecar.get("version", "unversioned")),
                trained_at=str(sidecar.get("trained_at", "")),
                fit_report=dict(sidecar.get("fit_report", {})),
            )
        table.validate()
        return table

    def validate(self) -> None:
        """
        Shape agreement between the arrays and the vocabularies.

        A vocab pointing past the end of its matrix would otherwise raise
        IndexError on whichever request first looks that id up. Checking here
        makes it a refused artifact at load time.
        """
        n_ad, n_pl = len(self.ad_vectors), len(self.placement_vectors)
        if self.ad_vectors.ndim != 2 or self.placement_vectors.ndim != 2:
            raise ValueError("embedding matrices must be 2-D")
        if self.ad_vectors.shape[1] != self.dim or self.placement_vectors.shape[1] != self.dim:
            raise ValueError(
                f"embedding width {self.ad_vectors.shape[1]}/"
                f"{self.placement_vectors.shape[1]} does not match dim={self.dim}"
            )
        if len(self.ad_bias) != n_ad or len(self.placement_bias) != n_pl:
            raise ValueError("bias vectors do not match the embedding matrices")
        if n_ad < 1 or n_pl < 1:
            raise ValueError("embedding matrices must contain at least the OOV row")
        if self.ad_vocab and max(self.ad_vocab.values()) >= n_ad:
            raise ValueError("ad vocabulary indexes past the end of its matrix")
        if self.placement_vocab and max(self.placement_vocab.values()) >= n_pl:
            raise ValueError("placement vocabulary indexes past the end of its matrix")
        if OOV_INDEX in self.ad_vocab.values() or OOV_INDEX in self.placement_vocab.values():
            raise ValueError("row 0 is reserved for OOV; no id may map to it")

    def describe(self) -> str:
        return (
            f"{self.version}: {len(self.ad_vocab)} ads x "
            f"{len(self.placement_vocab)} placements @ dim={self.dim}"
        )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def build_vocab(ids: Sequence[str], min_count: int) -> dict[str, int]:
    """
    Ids seen at least `min_count` times get their own row; the rest share the
    OOV row.

    Without the threshold, an ad with four impressions and one click gets a
    vector fit to a 25% click rate. The FM has nothing pulling that back, unlike
    ad_ctr_prior, which is beta-smoothed by construction. Pooling the tail
    instead gives it one row estimated from all of it, and emb_is_cold marks
    which rows those were.
    """
    counts: dict[str, int] = {}
    for value in ids:
        counts[value] = counts.get(value, 0) + 1
    kept = sorted(k for k, c in counts.items() if c >= min_count and k != "")
    return {key: i for i, key in enumerate(kept, start=OOV_INDEX + 1)}


def _index(ids: Sequence[str], vocab: dict[str, int]) -> np.ndarray:
    return np.fromiter(
        (vocab.get(v, OOV_INDEX) for v in ids), dtype=np.int64, count=len(ids)
    )


class _Adam:
    """
    Sparse Adam over the rows a minibatch touched.

    A dense update is simpler to write but applies weight decay and a stale
    moment to every row in the table on every step, dragging the vectors of ads
    that were not in the batch toward zero. With a few thousand ads and a few
    hundred steps per epoch that is most of the table, most of the time.
    """

    BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

    def __init__(self, shape: tuple[int, ...]):
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)

    def step(self, param: np.ndarray, grad: np.ndarray, rows, lr: float, t: int) -> None:
        g = grad[rows]
        self.m[rows] = self.BETA1 * self.m[rows] + (1.0 - self.BETA1) * g
        self.v[rows] = self.BETA2 * self.v[rows] + (1.0 - self.BETA2) * g * g
        m_hat = self.m[rows] / (1.0 - self.BETA1 ** t)
        v_hat = self.v[rows] / (1.0 - self.BETA2 ** t)
        param[rows] -= lr * m_hat / (np.sqrt(v_hat) + self.EPS)


def train_embeddings(
    ad_ids: Sequence[str],
    placement_ids: Sequence[str],
    y: Sequence[int] | np.ndarray,
    *,
    dim: int = DEFAULT_DIM,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    l2: float = DEFAULT_L2,
    min_count: int = DEFAULT_MIN_COUNT,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    seed: int = 7,
    version: str = "unversioned",
) -> EmbeddingTable:
    """
    Fit the factorisation machine by minibatch Adam on binary cross-entropy.

    Rows must arrive in time order. The holdout that picks the epoch is the tail
    of the input, so the epoch chosen is the one that generalises forward rather
    than the one that memorises a shuffled sample of the same week. Same reason
    train_ctr.py splits chronologically.

    The caller passes the training split and nothing else. Fitting on rows that
    later score the calibrator leaks: an ad's embedding carries the clicks it is
    about to be evaluated on, every downstream metric improves, and the
    promotion gates wave it through.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if n == 0:
        raise ValueError("no rows to fit embeddings on")
    if not (len(ad_ids) == len(placement_ids) == n):
        raise ValueError(
            f"ragged input: {len(ad_ids)} ad ids, {len(placement_ids)} placement "
            f"ids, {n} labels"
        )
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")

    ad_vocab = build_vocab(ad_ids, min_count)
    placement_vocab = build_vocab(placement_ids, min_count)
    ia_all = _index(ad_ids, ad_vocab)
    ip_all = _index(placement_ids, placement_vocab)

    n_holdout = int(n * holdout_frac) if epochs > 1 else 0
    n_fit = n - n_holdout
    if n_fit < batch_size:
        # Fit on everything and pick the epoch on training loss instead.
        n_fit, n_holdout = n, 0

    ia, ip, y_fit = ia_all[:n_fit], ip_all[:n_fit], y[:n_fit]
    ia_ho, ip_ho, y_ho = ia_all[n_fit:], ip_all[n_fit:], y[n_fit:]

    rng = np.random.default_rng(seed)
    # A symmetric init leaves every inner product at zero with a zero gradient,
    # and the vectors never separate.
    ad_vectors = rng.normal(0.0, 0.05, size=(len(ad_vocab) + 1, dim))
    placement_vectors = rng.normal(0.0, 0.05, size=(len(placement_vocab) + 1, dim))
    ad_bias = np.zeros(len(ad_vocab) + 1, dtype=np.float64)
    placement_bias = np.zeros(len(placement_vocab) + 1, dtype=np.float64)
    # Starting anywhere else spends the first steps climbing from logit(0.5)
    # down toward a ~1% event instead of learning structure.
    global_bias = np.asarray(_logit(float(y_fit.mean())), dtype=np.float64)

    opt_av, opt_pv = _Adam(ad_vectors.shape), _Adam(placement_vectors.shape)
    opt_ab, opt_pb = _Adam(ad_bias.shape), _Adam(placement_bias.shape)
    opt_gb = _Adam(())

    # See _Adam for why these are zeroed per row rather than reallocated.
    g_av = np.zeros_like(ad_vectors)
    g_pv = np.zeros_like(placement_vectors)
    g_ab = np.zeros_like(ad_bias)
    g_pb = np.zeros_like(placement_bias)

    def forward(ia_b, ip_b):
        e_ad = ad_vectors[ia_b]
        e_pl = placement_vectors[ip_b]
        z = (
            float(global_bias)
            + ad_bias[ia_b]
            + placement_bias[ip_b]
            + np.einsum("ij,ij->i", e_ad, e_pl)
        )
        return e_ad, e_pl, z

    def holdout_loss() -> float:
        if n_holdout == 0:
            return float("nan")
        _, _, z = forward(ia_ho, ip_ho)
        return _logloss(_sigmoid(z), y_ho)

    history: list[dict] = []
    best = None
    best_loss = float("inf")
    step = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_fit)
        epoch_loss, epoch_rows = 0.0, 0

        for start in range(0, n_fit, batch_size):
            batch = order[start:start + batch_size]
            ia_b, ip_b, y_b = ia[batch], ip[batch], y_fit[batch]
            e_ad, e_pl, z = forward(ia_b, ip_b)

            p = _sigmoid(z)
            epoch_loss += _logloss(p, y_b) * len(batch)
            epoch_rows += len(batch)

            # d(mean BCE)/dz per row.
            g = (p - y_b) / len(batch)
            step += 1

            rows_a = np.unique(ia_b)
            rows_p = np.unique(ip_b)

            # Scatter-add, because an id appearing twice in a batch has to
            # accumulate both gradients. Fancy-index assignment keeps the last.
            np.add.at(g_av, ia_b, g[:, None] * e_pl)
            np.add.at(g_pv, ip_b, g[:, None] * e_ad)
            np.add.at(g_ab, ia_b, g)
            np.add.at(g_pb, ip_b, g)

            # L2 on touched rows only, same reason the Adam update is sparse.
            # Biases stay unregularised: shrinking a bias toward zero pulls the
            # ad toward a 50% click rate, not toward the base rate.
            if l2:
                g_av[rows_a] += l2 * ad_vectors[rows_a]
                g_pv[rows_p] += l2 * placement_vectors[rows_p]

            opt_av.step(ad_vectors, g_av, rows_a, lr, step)
            opt_pv.step(placement_vectors, g_pv, rows_p, lr, step)
            opt_ab.step(ad_bias, g_ab, rows_a, lr, step)
            opt_pb.step(placement_bias, g_pb, rows_p, lr, step)
            opt_gb.step(global_bias, np.asarray(g.sum()), Ellipsis, lr, step)

            g_av[rows_a] = 0.0
            g_pv[rows_p] = 0.0
            g_ab[rows_a] = 0.0
            g_pb[rows_p] = 0.0

        train_loss = epoch_loss / max(epoch_rows, 1)
        ho_loss = holdout_loss()
        history.append({
            "epoch": epoch,
            "train_logloss": round(train_loss, 6),
            "holdout_logloss": None if math.isnan(ho_loss) else round(ho_loss, 6),
        })
        log.info("embeddings epoch %d/%d train_logloss=%.6f holdout_logloss=%s",
                 epoch, epochs, train_loss,
                 "n/a" if math.isnan(ho_loss) else f"{ho_loss:.6f}")

        # An FM over sparse ids overfits quickly, so the last epoch is rarely
        # the one to keep.
        score = train_loss if math.isnan(ho_loss) else ho_loss
        if score < best_loss:
            best_loss = score
            best = (
                ad_vectors.copy(), placement_vectors.copy(),
                ad_bias.copy(), placement_bias.copy(), float(global_bias),
                epoch,
            )

    assert best is not None
    a_vec, p_vec, a_bias, p_bias, g_bias, best_epoch = best

    # Narrow to float32 here, before the table is handed back, rather than on
    # the way to disk. Fitting runs in float64 and the artifact stores float32,
    # so narrowing at save() would leave the table that built the training
    # matrix differing from the one the gateway loads in the ninth digit.
    # Invisible in every metric, and still train/serve skew. Round once and both
    # sides are bit-identical.
    table = EmbeddingTable(
        dim=dim,
        ad_vocab=ad_vocab,
        placement_vocab=placement_vocab,
        ad_vectors=a_vec.astype(np.float32),
        placement_vectors=p_vec.astype(np.float32),
        ad_bias=a_bias.astype(np.float32),
        placement_bias=p_bias.astype(np.float32),
        global_bias=float(np.float32(g_bias)),
        version=version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        fit_report={
            "rows": n,
            "rows_fit": int(n_fit),
            "rows_holdout": int(n_holdout),
            "positives": int(y.sum()),
            "dim": dim,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_logloss": round(best_loss, 6),
            "min_count": min_count,
            "lr": lr,
            "l2": l2,
            "batch_size": batch_size,
            "seed": seed,
            "n_ads": len(ad_vocab),
            "n_placements": len(placement_vocab),
            "oov_rate_ads": round(float((ia_all == OOV_INDEX).mean()), 6),
            "oov_rate_placements": round(float((ip_all == OOV_INDEX).mean()), 6),
            "history": history,
        },
    )
    table.validate()
    log.info("fitted embedding table %s (best epoch %d, logloss %.6f)",
             table.describe(), best_epoch, best_loss)
    return table
