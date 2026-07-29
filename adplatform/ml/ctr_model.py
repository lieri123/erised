# ctr_model.py — model loading and inference for the serving path.
#
# Design goals, in priority order:
#   1. Never raise on the hot path. A broken model must degrade to the baseline,
#      not 500 the bid request.
#   2. Never block. Loading happens in a background task; predict() only reads
#      already-resident objects.
#   3. Never silently mismatch. A model trained on a different feature version
#      is refused at load time.

from __future__ import annotations

import json
import logging
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..settings import settings
from .features import (
    FEATURE_VERSION,
    N_FEATURES,
    CtrStats,
    RequestContext,
    extract_features,
)

log = logging.getLogger(__name__)

# Hard clamps. A model that predicts 0.0 makes bid_value 0 and the ad can never
# win again — a self-reinforcing dead zone. A model that predicts 0.9 on a
# 1% inventory will drain an advertiser's daily budget in minutes.
MIN_CTR = 0.0001
MAX_CTR = 0.25


def _undo_negative_downsampling(p: np.ndarray, keep_rate: float) -> np.ndarray:
    """
    A model trained on negatives downsampled at `keep_rate` predicts inflated
    probabilities. Recover the true scale.

    Downsampling negatives by w multiplies the odds by 1/w, so:
        odds_true = w * odds_sampled
        p_true    = w*p' / (w*p' + 1 - p')

    Skipping this step is the single most expensive bug available to you here:
    the model still ranks correctly, so offline AUC looks great, and you
    overbid on every impression in production.
    """
    if keep_rate >= 1.0:
        return p
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (keep_rate * p) / (keep_rate * p + 1.0 - p)


@dataclass
class _Artifact:
    booster: object
    calibrator: object | None
    keep_rate: float
    model_version: str


class CtrModel:
    """
    Thread-safe holder for the current model. Swap-on-reload via a single
    reference assignment, which is atomic under the GIL — no lock needed on the
    read path.
    """

    def __init__(self, artifact_dir: str | Path | None = None):
        self.artifact_dir = Path(artifact_dir or settings.model_dir)
        self._artifact: _Artifact | None = None
        self._loaded_stamp: float | None = None
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
        Safe to call repeatedly; no-ops when the on-disk artifact is unchanged.
        Call from a background task, never from a request handler.
        """
        meta_path = self.artifact_dir / "metadata.json"
        if not meta_path.exists():
            return False

        with self._load_lock:
            try:
                stamp = meta_path.stat().st_mtime
                if self._loaded_stamp == stamp:
                    return False

                meta = json.loads(meta_path.read_text())

                trained_version = meta.get("feature_version")
                if trained_version != FEATURE_VERSION:
                    log.error(
                        "refusing model %s: trained on feature_version=%s, "
                        "serving code is at %s. Retrain before deploying.",
                        meta.get("model_version"), trained_version, FEATURE_VERSION,
                    )
                    return False

                if meta.get("n_features") != N_FEATURES:
                    log.error("refusing model: expected %d features, artifact has %s",
                              N_FEATURES, meta.get("n_features"))
                    return False

                import xgboost as xgb

                booster = xgb.Booster()
                booster.load_model(str(self.artifact_dir / "model.json"))

                calibrator = None
                cal_path = self.artifact_dir / "calibrator.pkl"
                if cal_path.exists():
                    with cal_path.open("rb") as fh:
                        calibrator = pickle.load(fh)

                self._artifact = _Artifact(
                    booster=booster,
                    calibrator=calibrator,
                    keep_rate=float(meta.get("negative_keep_rate", 1.0)),
                    model_version=str(meta.get("model_version", "unknown")),
                )
                self._loaded_stamp = stamp
                log.info("loaded CTR model %s (calibrated=%s)",
                         self._artifact.model_version, calibrator is not None)
                return True

            except Exception:
                # Keep serving whatever was already loaded.
                log.exception("CTR model load failed; keeping previous model")
                return False

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

        One batched call, not one call per ad: XGBoost's per-call overhead
        dominates for small inputs, so scoring 40 ads individually can cost
        20x a single batched predict.
        """
        vectors = [extract_features(ad, ctx, stats) for ad in ads]
        if not vectors:
            return [], []

        artifact = self._artifact  # single read; may be swapped concurrently

        if artifact is None:
            return [
                self._baseline_ctr(ad, ctx, stats) for ad in ads
            ], vectors

        try:
            import xgboost as xgb

            matrix = xgb.DMatrix(np.asarray(vectors, dtype=np.float32))
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
        """
        Phase-1 baseline: beta-smoothed historical CTR for the (ad, placement)
        pair, nudged by keyword overlap so cold-start ads are not all identical.

        Do not dismiss this as a placeholder. It is also your promotion gate —
        train_ctr.py refuses to ship a model that cannot beat it on held-out
        log loss, which is the check that catches most training bugs.
        """
        base = stats.pair_ctr(ad.ad_id, ctx.placement_id)
        ad_kws = {k.lower() for k in (ad.target_keywords or ())}
        overlap = len(ad_kws & set(ctx.page_keywords))
        # Multiplicative so the adjustment scales with the ad's own history
        # rather than swamping it.
        boost = 1.0 + min(0.6, 0.15 * overlap)
        return float(np.clip(base * boost, MIN_CTR, MAX_CTR))


# Process-wide singleton, imported by rtb.py.
ctr_model = CtrModel()