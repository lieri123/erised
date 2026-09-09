# train_ctr.py — offline CTR training. Run nightly via cron or a k8s CronJob.
 
from __future__ import annotations
 
import argparse
import json
import logging
import pickle
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
 
import numpy as np
 
from .embeddings import (
    DEFAULT_DIM,
    DEFAULT_EPOCHS,
    EMBEDDING_FEATURE_NAMES,
    EMBEDDING_FILE,
    NEUTRAL_EMBEDDING_BLOCK,
    EmbeddingTable,
    train_embeddings,
)
from .features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    N_BASE_FEATURES,
    N_FEATURES,
)
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_ctr")
 
ATTRIBUTION_WINDOW_HOURS = 1
LABEL_CUTOFF_HOURS = 2       # must be >= attribution window
TARGET_NEGATIVES_PER_POSITIVE = 20.0

# Logged feature versions this trainer accepts, and the vector width of each.
#
# A version bump normally makes older impressions untrainable, and it should:
# column i means something different on either side of it. v3 is an exception
# because of how it was made. It appends to an unchanged v2 base, so a v2 vector
# is a v3 vector's first 17 columns.
#
# The learned columns are no help either way. A v3 row logged them from whatever
# table was live at serve time, which belongs to the previous model, not the one
# this run is about to fit. Every row's block is recomputed below from the table
# being shipped, so those columns are ignored on v3 rows and absent on v2 ones.
#
# Drop the 2 entry once the 90-day TTL has aged v2 impressions out.
TRAINABLE_FEATURE_VERSIONS: dict[int, int] = {
    2: N_BASE_FEATURES,
    3: N_FEATURES,
}

# Data loading

TRAINING_QUERY = """
SELECT
    toUnixTimestamp64Milli(i.ts)      AS ts_ms,
    i.features                        AS features,
    i.feature_version                 AS feature_version,
    i.ad_id                           AS ad_id,
    i.placement_id                    AS placement_id,
    i.serve_propensity                AS serve_propensity,
    i.is_exploration                  AS is_exploration,
    if(c.click_ts >= i.ts
       AND c.click_ts <= i.ts + INTERVAL {attr_hours:UInt8} HOUR, 1, 0) AS clicked
FROM ad_impressions AS i
LEFT JOIN (
    SELECT impression_id, min(ts) AS click_ts
    FROM ad_clicks
    GROUP BY impression_id
) AS c ON i.impression_id = c.impression_id
WHERE i.ts >= {start_ts:DateTime64(3)}
  AND i.ts <  now() - INTERVAL {cutoff_hours:UInt8} HOUR
  AND i.feature_version IN {feature_versions:Array(UInt16)}
  AND length(i.features) >= {n_base_features:UInt16}
  AND i.ad_id != ''
  AND i.placement_id != ''
ORDER BY i.ts ASC
"""

def load_from_clickhouse(days: int, dsn: str) -> dict[str, np.ndarray]:
    """Pull labelled impressions. Requires `pip install clickhouse-connect`."""
    import clickhouse_connect

    client = clickhouse_connect.get_client(dsn=dsn)
    start_ts = datetime.now(timezone.utc) - timedelta(days=days)

    result = client.query(
        TRAINING_QUERY,
        parameters={
            "attr_hours": ATTRIBUTION_WINDOW_HOURS,
            "cutoff_hours": LABEL_CUTOFF_HOURS,
            "start_ts": start_ts,
            "feature_versions": sorted(TRAINABLE_FEATURE_VERSIONS),
            "n_base_features": N_BASE_FEATURES,
        },
    )

    rows = result.result_rows
    if not rows:
        raise SystemExit("no labelled impressions in range — nothing to train on")

    return rows_to_arrays(rows)


def rows_to_arrays(rows: list) -> dict[str, np.ndarray]:
    """
    Shape the query result, keeping only the base block of each vector.

    The width check is per row and per version rather than global. SQL can
    filter on `length(features) >= 17` but not on "17 if v2, 23 if v3", and a
    truncated v3 row would otherwise slide into the base block unnoticed.
    Mismatches are dropped and counted, never repaired.
    """
    kept_x: list[list[float]] = []
    kept: list[tuple] = []
    dropped: dict[int, int] = {}

    for row in rows:
        _, features, version, *_ = row
        expected = TRAINABLE_FEATURE_VERSIONS.get(int(version))
        if expected is None or len(features) != expected:
            dropped[int(version)] = dropped.get(int(version), 0) + 1
            continue
        kept_x.append(list(features[:N_BASE_FEATURES]))
        kept.append(row)

    if dropped:
        log.warning("dropped %d rows with unusable vectors, by feature_version: %s",
                    sum(dropped.values()), dict(sorted(dropped.items())))
    if not kept:
        raise SystemExit(
            "every row had a vector width its feature_version does not permit — "
            f"expected {TRAINABLE_FEATURE_VERSIONS}"
        )

    versions = np.array([int(r[2]) for r in kept], dtype=np.uint16)
    by_version = {int(v): int((versions == v).sum()) for v in np.unique(versions)}
    log.info("pulled %d labelled impressions by feature_version: %s",
             len(kept), by_version)

    return {
        "ts_ms": np.array([r[0] for r in kept], dtype=np.int64),
        "X": np.array(kept_x, dtype=np.float32),
        "feature_version": versions,
        "ad_id": np.array([str(r[3]) for r in kept], dtype=object),
        "placement_id": np.array([str(r[4]) for r in kept], dtype=object),
        "propensity": np.array([r[5] for r in kept], dtype=np.float32),
        "is_exploration": np.array([r[6] for r in kept], dtype=np.uint8),
        "y": np.array([r[7] for r in kept], dtype=np.int8),
    }
 
 
# Splitting and sampling
 
def time_split(
    data: dict[str, np.ndarray],
    valid_frac: float = 0.15,
    calib_frac: float = 0.15,
) -> tuple[dict, dict, dict]:
    """
    Split chronologically, never randomly.
 
    Three splits, not two:
      train  — fit the trees (negatives downsampled)
      valid  — early stopping (negatives downsampled, same distribution)
      calib  — fit the calibrator and gate promotion (NOT downsampled, so it
               carries the true class balance)
    """
    n = len(data["y"])
    n_calib = int(n * calib_frac)
    n_valid = int(n * valid_frac)
    n_train = n - n_valid - n_calib
    if n_train <= 0:
        raise SystemExit(f"not enough rows to split: {n}")
 
    def take(lo: int, hi: int) -> dict:
        return {k: v[lo:hi] for k, v in data.items()}
 
    return take(0, n_train), take(n_train, n_train + n_valid), take(n_train + n_valid, n)
 
 
def downsample_negatives(
    split: dict[str, np.ndarray],
    rng: np.random.Generator,
    target_ratio: float = TARGET_NEGATIVES_PER_POSITIVE,
) -> tuple[dict[str, np.ndarray], float]:
    y = split["y"]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise SystemExit("no positive examples — check the attribution join")
 
    wanted_neg = min(n_neg, int(n_pos * target_ratio))
    keep_rate = wanted_neg / n_neg if n_neg else 1.0
    if keep_rate >= 1.0:
        log.info("no downsampling needed (%d pos / %d neg)", n_pos, n_neg)
        return split, 1.0
 
    neg_idx = np.flatnonzero(y == 0)
    pos_idx = np.flatnonzero(y == 1)
    kept_neg = rng.choice(neg_idx, size=wanted_neg, replace=False)
    idx = np.sort(np.concatenate([pos_idx, kept_neg]))
 
    log.info("downsampled negatives %d -> %d (keep_rate=%.5f)", n_neg, wanted_neg, keep_rate)
    return {k: v[idx] for k, v in split.items()}, keep_rate
 
 
def undo_negative_downsampling(p: np.ndarray, keep_rate: float) -> np.ndarray:
    """Inverse of the downsampling shift. Mirrors ctr_model._undo_negative_downsampling."""
    if keep_rate >= 1.0:
        return p
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (keep_rate * p) / (keep_rate * p + 1.0 - p)
 
# Embeddings

def fit_embedding_table(
    train_split: dict[str, np.ndarray],
    *,
    dim: int,
    epochs: int,
    min_count: int,
    seed: int,
    version: str,
) -> EmbeddingTable:
    """
    Fit the ad/placement table on the training split.

    On the full split, not the downsampled one. Downsampling drops 95% of the
    negatives to keep the boosting problem tractable; the FM is one logistic
    layer over a few thousand parameters and every impression sharpens it. The
    bias term absorbs the class balance.

    Never on the calibration split. That split decides whether the model ships,
    and an embedding that has already seen those clicks improves the model on
    exactly the rows chosen to be unseen. The gates cannot tell that apart from
    a real improvement.
    """
    return train_embeddings(
        train_split["ad_id"],
        train_split["placement_id"],
        train_split["y"],
        dim=dim,
        epochs=epochs,
        min_count=min_count,
        seed=seed,
        version=version,
    )


def attach_embedding_block(
    split: dict[str, np.ndarray], table: EmbeddingTable | None
) -> dict[str, np.ndarray]:
    """
    Widen a split's base vectors to the full N_FEATURES.

    Every row's block comes from the same EmbeddingTable the gateway calls per
    request, and that table travels inside the artifact next to the booster fit
    on top of it. One function, one table.

    `table=None` fills the neutral block instead, so a --no-embeddings model
    keeps the width serving expects and carries six constant columns.
    """
    n = len(split["y"])
    if table is None:
        block = np.tile(
            np.asarray(NEUTRAL_EMBEDDING_BLOCK, dtype=np.float32), (n, 1)
        )
    else:
        block = table.blocks_for_rows(split["ad_id"], split["placement_id"])

    widened = dict(split)
    widened["X"] = np.hstack([split["X"], block]).astype(np.float32)
    assert widened["X"].shape[1] == N_FEATURES, widened["X"].shape
    return widened


 # Metrics
 
@dataclass
class Metrics:
    log_loss: float
    auc: float
    calibration_ratio: float   # sum(predicted) / sum(actual); 1.0 is perfect
    mean_predicted: float
    mean_actual: float
    n_rows: int
    n_positives: int
 
 
def evaluate(p: np.ndarray, y: np.ndarray) -> Metrics:
    from sklearn.metrics import log_loss as sk_log_loss, roc_auc_score
 
    p = np.clip(p, 1e-7, 1 - 1e-7)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")   # single-class split
 
    return Metrics(
        log_loss=float(sk_log_loss(y, p, labels=[0, 1])),
        auc=auc,
        calibration_ratio=float(p.sum() / max(y.sum(), 1e-9)),
        mean_predicted=float(p.mean()),
        mean_actual=float(y.mean()),
        n_rows=int(len(y)),
        n_positives=int(y.sum()),
    )
 
 
def decile_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """
    Predicted vs actual CTR by predicted-probability decile. This is the plot
    that reveals miscalibration; AUC is rank-invariant and cannot see it.
    """
    order = np.argsort(p)
    p_sorted, y_sorted = p[order], y[order]
    out = []
    for chunk_p, chunk_y in zip(
        np.array_split(p_sorted, n_bins), np.array_split(y_sorted, n_bins)
    ):
        if len(chunk_y) == 0:
            continue
        out.append({
            "predicted": round(float(chunk_p.mean()), 6),
            "actual": round(float(chunk_y.mean()), 6),
            "n": int(len(chunk_y)),
        })
    return out
 
 
def baseline_predictions(split: dict[str, np.ndarray]) -> np.ndarray:
    """
    The Phase-1 baseline, reconstructed from the logged features so it is scored
    on exactly the same rows as the model. pair_ctr_prior is already a
    beta-smoothed historical CTR, so it stands alone as a prediction.
    """
    idx = FEATURE_NAMES.index("pair_ctr_prior")
    return np.clip(split["X"][:, idx].astype(np.float64), 1e-6, 0.5)
 
 # Training
 
def train_model(
    train: dict, valid: dict, use_ips: bool = False, num_rounds: int = 600
):
    import xgboost as xgb
 
    def weights(split: dict) -> np.ndarray | None:
        if not use_ips:
            return None
        prop = np.clip(split["propensity"].astype(np.float64), 0.01, 1.0)
        w = 1.0 / prop
        return w / w.mean()
 
    dtrain = xgb.DMatrix(train["X"], label=train["y"], weight=weights(train),
                         feature_names=list(FEATURE_NAMES))
    dvalid = xgb.DMatrix(valid["X"], label=valid["y"], weight=weights(valid),
                         feature_names=list(FEATURE_NAMES))
 
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 20,      
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "tree_method": "hist",
        "nthread": 0,
    }
 
    evals_result: dict = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=40,
        evals_result=evals_result,
        verbose_eval=50,
    )
    log.info("best iteration %d (valid logloss %.6f)",
             booster.best_iteration, booster.best_score)
    return booster
 
 
def fit_calibrator(p_corrected: np.ndarray, y: np.ndarray):   
    from sklearn.isotonic import IsotonicRegression

    n_pos = int(y.sum())
    if n_pos == 0:
        raise SystemExit(
            "calibrator-fit split contains no positive labels, so isotonic "
            "regression would collapse to the constant 0.\nThe impression log "
            "is almost certainly mixing data regimes."
        )
    if n_pos < 30:
        log.warning("only %d positives available to fit the calibrator — the "
                    "fit will be coarse and the calibration gate unreliable. "
                    "Generate more traffic.", n_pos)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_corrected, y)
    return iso


def check_calibrator_output(p_final: np.ndarray) -> None:
    """A constant output means the fit degenerated; see fit_calibrator."""
    if len(np.unique(p_final)) == 1:
        raise SystemExit(
            f"the calibrator produced a constant {p_final[0]:.6f} for every "
            f"row, so AUC is 0.5 by construction and the gates below are "
            f"meaningless.\nThe scored values fall outside the range the "
            f"calibrator was fit on and out_of_bounds='clip' pinned them all "
            f"to one endpoint. The two halves of the calibration split are not "
            f"comparable — check the impression log for mixed data regimes."
        )
 
 # Orchestration
 
def run(data: dict[str, np.ndarray], out_dir: Path, dry_run: bool = False,
        use_ips: bool = False, seed: int = 7, use_embeddings: bool = True,
        embedding_dim: int = DEFAULT_DIM,
        embedding_epochs: int = DEFAULT_EPOCHS,
        embedding_min_count: int = 20) -> dict:
    import xgboost as xgb

    rng = np.random.default_rng(seed)
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

    train_raw, valid_raw, calib = time_split(data)
    log.info("split: train=%d valid=%d calib=%d (global CTR %.4f%%)",
             len(train_raw["y"]), len(valid_raw["y"]), len(calib["y"]),
             100 * data["y"].mean())

    embedding_table = None
    if use_embeddings:
        embedding_table = fit_embedding_table(
            train_raw, dim=embedding_dim, epochs=embedding_epochs,
            min_count=embedding_min_count, seed=seed, version=version,
        )
    else:
        log.info("--no-embeddings: the learned block will be constant")

    train_raw = attach_embedding_block(train_raw, embedding_table)
    valid_raw = attach_embedding_block(valid_raw, embedding_table)
    calib = attach_embedding_block(calib, embedding_table)

    train, keep_rate = downsample_negatives(train_raw, rng)
    valid, _ = downsample_negatives(valid_raw, rng)

    booster = train_model(train, valid, use_ips=use_ips)
 
    # Score the untouched calibration split.
    dcalib = xgb.DMatrix(calib["X"], feature_names=list(FEATURE_NAMES))
    raw = booster.predict(dcalib, iteration_range=(0, booster.best_iteration + 1))
    corrected = undo_negative_downsampling(np.asarray(raw, dtype=np.float64), keep_rate)
 
    # Calibrate on the first half, measure on the second 
    mid = len(calib["y"]) // 2
    cal_fit = {k: v[:mid] for k, v in calib.items()}
    cal_test = {k: v[mid:] for k, v in calib.items()}
 
    calibrator = fit_calibrator(corrected[:mid], cal_fit["y"])
    p_calibrated = calibrator.predict(corrected[mid:])
    check_calibrator_output(p_calibrated)
 
    m_raw = evaluate(np.clip(np.asarray(raw)[mid:], 1e-7, 1 - 1e-7), cal_test["y"])
    m_corrected = evaluate(corrected[mid:], cal_test["y"])
    m_calibrated = evaluate(p_calibrated, cal_test["y"])
    m_baseline = evaluate(baseline_predictions(cal_test), cal_test["y"])

    # What the table is worth alone, on the same held-out rows as everything
    # else. Never a candidate for serving, since it knows nothing about
    # keywords, device or pacing, but it answers whether the vectors learned
    # anything or the trees are just routing around noise.
    m_embeddings = (
        evaluate(
            embedding_table.predict_proba(
                cal_test["ad_id"], cal_test["placement_id"]
            ),
            cal_test["y"],
        )
        if embedding_table is not None
        else None
    )

    use_isotonic = m_calibrated.log_loss < m_corrected.log_loss
    if use_isotonic:
        log.info("isotonic calibration IMPROVES log loss (%.6f -> %.6f) — keeping it",
                 m_corrected.log_loss, m_calibrated.log_loss)
        p_final, m_final = p_calibrated, m_calibrated
    else:
        log.info("isotonic calibration does NOT improve log loss (%.6f -> %.6f) — "
                 "skipping it; the downsampling correction is already sufficient",
                 m_corrected.log_loss, m_calibrated.log_loss)
        calibrator, p_final, m_final = None, corrected[mid:], m_corrected
 
    log.info("uncorrected   logloss=%.6f auc=%.4f calib_ratio=%.3f",
             m_raw.log_loss, m_raw.auc, m_raw.calibration_ratio)
    log.info("corrected     logloss=%.6f auc=%.4f calib_ratio=%.3f",
             m_corrected.log_loss, m_corrected.auc, m_corrected.calibration_ratio)
    log.info("calibrated    logloss=%.6f auc=%.4f calib_ratio=%.3f",
             m_final.log_loss, m_final.auc, m_final.calibration_ratio)
    log.info("baseline      logloss=%.6f auc=%.4f calib_ratio=%.3f",
             m_baseline.log_loss, m_baseline.auc, m_baseline.calibration_ratio)
    if m_embeddings is not None:
        log.info("embeddings    logloss=%.6f auc=%.4f calib_ratio=%.3f  (FM alone)",
                 m_embeddings.log_loss, m_embeddings.auc,
                 m_embeddings.calibration_ratio)

    gate_failures = []
    if not (m_final.log_loss < m_baseline.log_loss):
        gate_failures.append(
            f"log loss {m_final.log_loss:.6f} does not beat baseline {m_baseline.log_loss:.6f}"
        )
    if not (0.85 <= m_final.calibration_ratio <= 1.15):
        gate_failures.append(
            f"calibration ratio {m_final.calibration_ratio:.3f} outside [0.85, 1.15]"
        )
    if not (m_final.auc > 0.55):
        gate_failures.append(f"AUC {m_final.auc:.4f} below 0.55 — barely better than random")
 
    metadata = {
        "model_version": version,
        "feature_version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "feature_names": list(FEATURE_NAMES),
        # Read by ctr_model at load time. True means the booster's last six
        # columns came from a table that has to be present, and an artifact
        # missing embeddings.npz is refused rather than served with constants.
        "uses_embeddings": embedding_table is not None,
        "embedding": (
            {
                "dim": embedding_table.dim,
                "block": list(EMBEDDING_FEATURE_NAMES),
                "fit": embedding_table.fit_report,
            }
            if embedding_table is not None
            else None
        ),
        "negative_keep_rate": keep_rate,
        "best_iteration": int(booster.best_iteration),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "used_ips_weighting": use_ips,
        "used_isotonic_calibration": bool(use_isotonic),
        "rows": {"train": len(train_raw["y"]), "valid": len(valid_raw["y"]),
                 "calib": len(calib["y"])},
        "metrics": {
            "model": asdict(m_final),
            "model_uncalibrated": asdict(m_corrected),
            "baseline": asdict(m_baseline),
            "embeddings_only": asdict(m_embeddings) if m_embeddings else None,
        },
        "decile_table": decile_table(p_final, cal_test["y"]),
        "feature_importance": {
            k: float(v) for k, v in sorted(
                booster.get_score(importance_type="gain").items(),
                key=lambda kv: -kv[1],
            )
        },
        "promoted": not gate_failures and not dry_run,
        "gate_failures": gate_failures,
    }
 
    if gate_failures:
        log.error("PROMOTION BLOCKED:")
        for f in gate_failures:
            log.error("  - %s", f)
    elif dry_run:
        log.info("gates passed; --dry-run set, not promoting")
    else:
        log.info("gates passed; promoting %s", version)
 
  
    version_dir = out_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(version_dir / "model.json"))

    # The table ships with the booster, in the same immutable version
    # directory. The vectors only mean anything to the trees fit on the exact
    # numbers they produced, so a separately updatable table is a slower route
    # to the same skew.
    if embedding_table is not None:
        embedding_table.save(version_dir / EMBEDDING_FILE)

    if calibrator is not None:
        with (version_dir / "calibrator.pkl").open("wb") as fh:
            pickle.dump(calibrator, fh)
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
 
    if metadata["promoted"]:
        current = out_dir / "current"
        staging = out_dir / ".current.staging"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(version_dir, staging)
        if current.exists():
            shutil.rmtree(current)
        staging.rename(current)
        log.info("promoted to %s", current)
 
    return metadata
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--dsn", default="clickhouse://default@localhost:8123/default")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--use-ips", action="store_true",
                    help="inverse-propensity weighting; needs real exploration data")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="train without the learned ad/placement table; the "
                         "block is filled with constants so the artifact still "
                         "has the width serving expects. Use it to measure what "
                         "the embeddings are actually buying.")
    ap.add_argument("--embedding-dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--embedding-epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--embedding-min-count", type=int, default=20,
                    help="ids with fewer impressions than this share the OOV row")
    args = ap.parse_args()

    data = load_from_clickhouse(args.days, args.dsn)
    run(data, args.out, dry_run=args.dry_run, use_ips=args.use_ips,
        use_embeddings=not args.no_embeddings,
        embedding_dim=args.embedding_dim,
        embedding_epochs=args.embedding_epochs,
        embedding_min_count=args.embedding_min_count)
 
 
if __name__ == "__main__":
    main()