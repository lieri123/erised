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
 
from .features import FEATURE_NAMES, FEATURE_VERSION, N_FEATURES
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_ctr")
 
ATTRIBUTION_WINDOW_HOURS = 1
LABEL_CUTOFF_HOURS = 2       # must be >= attribution window
TARGET_NEGATIVES_PER_POSITIVE = 20.0
 
# Data loading
 
TRAINING_QUERY = """
SELECT
    toUnixTimestamp64Milli(i.ts)      AS ts_ms,
    i.features                        AS features,
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
  AND i.feature_version = {feature_version:UInt16}
  AND length(i.features) = {n_features:UInt16}
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
            "feature_version": FEATURE_VERSION,
            "n_features": N_FEATURES,
        },
    )
 
    rows = result.result_rows
    if not rows:
        raise SystemExit("no labelled impressions in range — nothing to train on")
 
    log.info("pulled %d labelled impressions", len(rows))
    return {
        "ts_ms": np.array([r[0] for r in rows], dtype=np.int64),
        "X": np.array([r[1] for r in rows], dtype=np.float32),
        "propensity": np.array([r[2] for r in rows], dtype=np.float32),
        "is_exploration": np.array([r[3] for r in rows], dtype=np.uint8),
        "y": np.array([r[4] for r in rows], dtype=np.int8),
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
        use_ips: bool = False, seed: int = 7) -> dict:
    import xgboost as xgb
 
    rng = np.random.default_rng(seed)
 
    train_raw, valid_raw, calib = time_split(data)
    log.info("split: train=%d valid=%d calib=%d (global CTR %.4f%%)",
             len(train_raw["y"]), len(valid_raw["y"]), len(calib["y"]),
             100 * data["y"].mean())
 
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
 
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    metadata = {
        "model_version": version,
        "feature_version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "feature_names": list(FEATURE_NAMES),
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
    args = ap.parse_args()
 
    data = load_from_clickhouse(args.days, args.dsn)
    run(data, args.out, dry_run=args.dry_run, use_ips=args.use_ips)
 
 
if __name__ == "__main__":
    main()