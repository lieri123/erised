# ctr_model.py — model loading and inference for the serving path.

from __future__ import annotations

import json
import logging
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..settings import settings
from .artifacts import (
    CALIBRATOR_FILE,
    METADATA_FILE,
    MODEL_FILE,
    ArtifactStore,
    LocalArtifactStore,
    build_store,
)
from .features import (
    FEATURE_VERSION,
    N_FEATURES,
    CtrStats,
    RequestContext,
    extract_features,
)

log = logging.getLogger(__name__)

MIN_CTR = 0.0001
MAX_CTR = 0.25


def _undo_negative_downsampling(p: np.ndarray, keep_rate: float) -> np.ndarray:
    if keep_rate >= 1.0:
        return p
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (keep_rate * p) / (keep_rate * p + 1.0 - p)


def _tree_range(booster, best_iteration) -> tuple[int, int] | None:
    """
    The slice of trees that training actually gated, as an `iteration_range`.

    `xgb.train(..., early_stopping_rounds=40)` keeps boosting for 40 rounds
    past the optimum and returns a booster holding ALL of them. train_ctr.py
    scores the calibration split with `iteration_range=(0, best_iteration + 1)`,
    so every number promotion depends on -- log loss, AUC, calibration ratio,
    and the isotonic calibrator's fitted domain -- describes only that prefix.
    `booster.predict(m)` with no range uses every tree, including the 40 rounds
    of overfitting after the optimum. Serving that way means the model in
    production is not the model that passed the gates, and the calibrator is
    applied to a distribution it was never fit on.

    Returns None when the artifact does not record best_iteration, which keeps
    pre-existing artifacts loading exactly as before.
    """
    if best_iteration is None:
        return None
    try:
        end = int(best_iteration) + 1
    except (TypeError, ValueError):
        log.warning("artifact best_iteration=%r is not an integer; "
                    "serving the whole booster", best_iteration)
        return None
    if end < 1:
        return None
    # An end past the last tree makes predict() raise, which would turn every
    # bid into a baseline fallback. Clamp instead.
    try:
        n_rounds = int(booster.num_boosted_rounds())
    except Exception:
        return (0, end)
    if end > n_rounds:
        log.warning("artifact best_iteration=%s exceeds its %d boosted rounds; "
                    "serving all of them", best_iteration, n_rounds)
        return (0, n_rounds)
    return (0, end)


@dataclass
class _Artifact:
    booster: object
    calibrator: object | None
    keep_rate: float
    model_version: str
    # (0, best_iteration + 1), or None when the artifact does not say. See
    # _tree_range below for why serving the whole booster is wrong.
    iteration_range: tuple[int, int] | None = None


class CtrModel:
    """
    Holds the live artifact and scores against it.

    Where the artifact comes from is the store's problem — a local directory in
    development, an S3 prefix behind a `current.json` pointer when there is more
    than one replica. Everything below the store boundary loads from a local
    directory either way.
    """

    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        store: ArtifactStore | None = None,
    ):
        # An explicit directory always means local — that is what the tests and
        # the training script pass, and neither should reach for S3.
        if store is not None:
            self.store = store
        elif artifact_dir is not None:
            self.store = LocalArtifactStore(artifact_dir)
        else:
            self.store = build_store()

        self._artifact: _Artifact | None = None
        self._loaded_token: str | None = None
        self._load_lock = threading.Lock()

    # -- properties ---------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._artifact is not None

    @property
    def model_version(self) -> str:
        return self._artifact.model_version if self._artifact else "baseline"

    # -- loading ------------------------------------------------------------

    def load(self) -> bool:
        """
        Load or reload the artifact. Returns True if a new model became active.
        Safe to call repeatedly; no-ops when the published artifact is unchanged.
        Call from a background task, never from a request handler — the S3 store
        does blocking network I/O in here.
        """
        with self._load_lock:
            try:
                resolved = self.store.resolve(self._loaded_token)
            except Exception:
                log.exception("artifact store %s failed; keeping previous model",
                              self.store.describe())
                return False

            if resolved is None:
                return False

            try:
                directory = resolved.directory
                meta = json.loads((directory / METADATA_FILE).read_text())

                trained_version = meta.get("feature_version")
                if trained_version != FEATURE_VERSION:
                    log.error(
                        "refusing model %s: trained on feature_version=%s, "
                        "serving code is at %s. Retrain before deploying.",
                        meta.get("model_version"), trained_version, FEATURE_VERSION,
                    )
                    # Record the token anyway. Without this, every refresh tick
                    # re-resolves and re-rejects the same bad artifact, which in
                    # the S3 case is a paid GET every 60 seconds forever.
                    self._loaded_token = resolved.token
                    return False

                if meta.get("n_features") != N_FEATURES:
                    log.error("refusing model: expected %d features, artifact has %s",
                              N_FEATURES, meta.get("n_features"))
                    self._loaded_token = resolved.token
                    return False

                import xgboost as xgb

                booster = xgb.Booster()
                booster.load_model(str(directory / MODEL_FILE))

                calibrator = None
                cal_path = directory / CALIBRATOR_FILE
                if cal_path.exists():
                    with cal_path.open("rb") as fh:
                        calibrator = pickle.load(fh)

                self._artifact = _Artifact(
                    booster=booster,
                    calibrator=calibrator,
                    keep_rate=float(meta.get("negative_keep_rate", 1.0)),
                    model_version=str(meta.get("model_version", "unknown")),
                    iteration_range=_tree_range(booster, meta.get("best_iteration")),
                )
                self._loaded_token = resolved.token
                log.info("loaded CTR model %s from %s (calibrated=%s, trees=%s)",
                         self._artifact.model_version, self.store.describe(),
                         calibrator is not None,
                         self._artifact.iteration_range[1]
                         if self._artifact.iteration_range else "all")
                return True

            except Exception:
                # Keep serving whatever was already loaded. Deliberately does
                # NOT record the token — a transient read failure must be
                # retried, unlike a structurally incompatible artifact above.
                log.exception("CTR model load failed; keeping previous model")
                return False

    def status(self) -> dict:
        """For /health. Cheap, no I/O."""
        return {
            "trained": self.is_trained,
            "model_version": self.model_version,
            "source": self.store.describe(),
        }

    # -- inference ----------------------------------------------------------

    def predict_batch(
        self,
        ads: list,
        ctx: RequestContext,
        stats: CtrStats,
    ) -> tuple[list[float], list[list[float]]]:
        """
        Score every eligible ad in one shot.

        Returns (ctrs, feature_vectors). The feature vectors come back so the
        caller can log the winner's exact input — do not recompute them.
        """
        vectors = [extract_features(ad, ctx, stats) for ad in ads]
        if not vectors:
            return [], []

        # Read the reference once. load() rebinds self._artifact wholesale from
        # a background thread, so taking a local reference means a swap that
        # lands mid-request cannot pair one model's booster with another's
        # calibrator.
        artifact = self._artifact

        if artifact is None:
            return [
                self._baseline_ctr(ad, ctx, stats) for ad in ads
            ], vectors

        try:
            import xgboost as xgb

            matrix = xgb.DMatrix(np.asarray(vectors, dtype=np.float32))
            if artifact.iteration_range is not None:
                raw = artifact.booster.predict(
                    matrix, iteration_range=artifact.iteration_range)
            else:
                raw = artifact.booster.predict(matrix)
            corrected = _undo_negative_downsampling(np.asarray(raw), artifact.keep_rate)

            if artifact.calibrator is not None:
                corrected = artifact.calibrator.predict(corrected)

            clipped = np.clip(corrected, MIN_CTR, MAX_CTR)
            return [float(x) for x in clipped], vectors

        except Exception:
            log.exception("CTR inference failed; falling back to baseline")
            return [self._baseline_ctr(ad, ctx, stats) for ad in ads], vectors

    def _baseline_ctr(self, ad, ctx: RequestContext, stats: CtrStats) -> float:
        base = stats.pair_ctr(ad.ad_id, ctx.placement_id)
        ad_kws = {k.lower() for k in (ad.target_keywords or ())}
        overlap = len(ad_kws & set(ctx.page_keywords))
        boost = 1.0 + min(0.6, 0.15 * overlap)
        return float(np.clip(base * boost, MIN_CTR, MAX_CTR))


# Process-wide singleton, imported by rtb.py. Building the store reads settings
# but opens no connections, so import stays cheap.
ctr_model = CtrModel()
