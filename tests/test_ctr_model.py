# test_ctr_model.py — loading, refusal, and graceful degradation.
#
# The theme: the gateway must keep serving ads no matter what state the model is
# in. A missing artifact, a corrupt one, a booster that raises mid-predict — all
# of these degrade to the baseline CTR heuristic. None of them 500 a bid.
#
# The mirror image also matters: an artifact that would produce WRONG scores
# must be refused outright rather than degraded into. A model trained on a
# different feature layout does not fail loudly; it returns confident numbers
# computed from columns that mean something else.

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from adplatform.ml.artifacts import LocalArtifactStore
from adplatform.ml.ctr_model import MAX_CTR, MIN_CTR, CtrModel, _undo_negative_downsampling
from adplatform.ml.features import (
    FEATURE_VERSION,
    N_FEATURES,
    CtrStats,
    RequestContext,
    extract_features,
)

import numpy as np


@dataclass
class FakeAd:
    ad_id: str = "ad_1"
    target_cpm: float = 5.0
    target_keywords: tuple = field(default_factory=tuple)
    daily_budget_usd: float = 100.0
    spent_today_usd: float = 0.0
    created_at: datetime | None = None


REQUEST_TS = datetime(2026, 8, 25, 14, 30, tzinfo=timezone.utc)


def ctx() -> RequestContext:
    return RequestContext.build(
        publisher_id="pub_1", placement_id="place_1", device_type="mobile",
        page_keywords=["running"],
        request_ts=REQUEST_TS,
    )


def write_metadata(directory: Path, **overrides) -> None:
    meta = {
        "model_version": "v_test",
        "feature_version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "negative_keep_rate": 1.0,
        "promoted": True,
    }
    meta.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps(meta))


class ShrinkingCalibrator:
    """Module scope on purpose: pickle resolves classes by qualified name, so a
    class defined inside a test function cannot be dumped at all.

    Shrinks rather than inflates. An inflating stub is invisible here because
    the tiny booster already predicts near MAX_CTR and both sides clip to the
    same value — the test passes for the wrong reason, or fails for one."""

    def predict(self, p):
        return np.asarray(p) * 0.1


class BrokenCalibrator:
    def predict(self, p):
        raise RuntimeError("calibrator exploded")


def train_tiny_booster(path: Path) -> None:
    """A real 2-tree booster on random data — enough to exercise the load path."""
    import xgboost as xgb

    rng = np.random.default_rng(0)
    X = rng.random((200, N_FEATURES)).astype(np.float32)
    y = (rng.random(200) < 0.3).astype(int)
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 2},
                        xgb.DMatrix(X, label=y), num_boost_round=2)
    booster.save_model(str(path))


# ---------------------------------------------------------------------------

class TestUntrainedModel:

    def test_reports_baseline_version_before_loading(self):
        model = CtrModel(artifact_dir="/nonexistent")
        assert model.is_trained is False
        assert model.model_version == "baseline"

    def test_load_on_a_missing_directory_returns_false(self):
        assert CtrModel(artifact_dir="/nonexistent").load() is False

    def test_scores_with_the_baseline_heuristic(self):
        # No model, but the gateway still needs numbers to run an auction with.
        model = CtrModel(artifact_dir="/nonexistent")
        ctrs, vectors = model.predict_batch([FakeAd()], ctx(), CtrStats())
        assert len(ctrs) == 1
        assert MIN_CTR <= ctrs[0] <= MAX_CTR

    def test_returns_feature_vectors_even_untrained(self):
        # rtb_integration logs winner.features regardless of model state. If
        # these came back empty, exploration impressions would be untrainable.
        _, vectors = CtrModel(artifact_dir="/nope").predict_batch(
            [FakeAd()], ctx(), CtrStats())
        assert len(vectors[0]) == N_FEATURES

    def test_empty_ad_list_returns_two_empty_lists(self):
        ctrs, vectors = CtrModel(artifact_dir="/nope").predict_batch(
            [], ctx(), CtrStats())
        assert ctrs == [] and vectors == []

    def test_baseline_rewards_keyword_overlap(self):
        model = CtrModel(artifact_dir="/nope")
        stats = CtrStats(global_ctr=0.01)
        matched = FakeAd(ad_id="a", target_keywords=("running",))
        unmatched = FakeAd(ad_id="b", target_keywords=("cooking",))
        ctrs, _ = model.predict_batch([matched, unmatched], ctx(), stats)
        assert ctrs[0] > ctrs[1]


class TestLoadGates:
    """
    Refusing a bad artifact is the whole reason load() is not just a file read.
    """

    def test_loads_a_valid_artifact(self, tmp_path):
        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        model = CtrModel(artifact_dir=tmp_path)

        assert model.load() is True
        assert model.is_trained is True
        assert model.model_version == "v_test"

    def test_refuses_a_stale_feature_version(self, tmp_path):
        # THE IMPORTANT ONE. A model trained on an older feature layout will
        # happily predict — on columns that have since shifted meaning. There
        # is no runtime symptom. Serving it is strictly worse than serving the
        # baseline heuristic, so the gate refuses and the gateway degrades.
        write_metadata(tmp_path, feature_version=FEATURE_VERSION - 1)
        train_tiny_booster(tmp_path / "model.json")
        model = CtrModel(artifact_dir=tmp_path)

        assert model.load() is False
        assert model.is_trained is False

    def test_refuses_a_wrong_feature_count(self, tmp_path):
        write_metadata(tmp_path, n_features=N_FEATURES + 3)
        train_tiny_booster(tmp_path / "model.json")
        assert CtrModel(artifact_dir=tmp_path).load() is False

    def test_a_refused_artifact_does_not_evict_a_good_one(self, tmp_path):
        # Serving must survive a bad nightly retrain. The previously loaded
        # model stays live rather than the gateway falling back to baseline.
        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        model = CtrModel(artifact_dir=tmp_path)
        assert model.load() is True

        write_metadata(tmp_path, model_version="v_bad",
                       feature_version=FEATURE_VERSION - 1)
        assert model.load() is False
        assert model.model_version == "v_test"

    def test_corrupt_model_file_does_not_raise(self, tmp_path):
        write_metadata(tmp_path)
        (tmp_path / "model.json").write_text("this is not a booster")
        model = CtrModel(artifact_dir=tmp_path)

        assert model.load() is False
        assert model.is_trained is False

    def test_missing_metadata_is_a_no_op(self, tmp_path):
        train_tiny_booster(tmp_path / "model.json")
        assert CtrModel(artifact_dir=tmp_path).load() is False


class TestReload:

    def test_second_load_of_an_unchanged_artifact_is_a_no_op(self, tmp_path):
        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        model = CtrModel(artifact_dir=tmp_path)

        assert model.load() is True
        assert model.load() is False      # nothing changed

    def test_a_new_artifact_is_picked_up(self, tmp_path):
        import os

        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        model = CtrModel(artifact_dir=tmp_path)
        model.load()

        write_metadata(tmp_path, model_version="v_next")
        os.utime(tmp_path / "metadata.json", (2_000_000_000, 2_000_000_000))

        assert model.load() is True
        assert model.model_version == "v_next"

    def test_accepts_an_injected_store(self, tmp_path):
        # The seam the S3 backend uses. An explicit directory must still mean
        # local, never S3, or tests and train_ctr.py would reach for a bucket.
        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        model = CtrModel(store=LocalArtifactStore(tmp_path))
        assert model.load() is True

    def test_status_reports_the_source(self, tmp_path):
        model = CtrModel(artifact_dir=tmp_path)
        status = model.status()
        assert status["trained"] is False
        assert str(tmp_path) in status["source"]


class TestCalibrator:

    def test_calibrator_is_applied_when_present(self, tmp_path):
        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        with (tmp_path / "calibrator.pkl").open("wb") as fh:
            pickle.dump(ShrinkingCalibrator(), fh)

        model = CtrModel(artifact_dir=tmp_path)
        model.load()
        with_cal, _ = model.predict_batch([FakeAd()], ctx(), CtrStats())

        (tmp_path / "calibrator.pkl").unlink()
        bare = CtrModel(artifact_dir=tmp_path)
        bare.load()
        without_cal, _ = bare.predict_batch([FakeAd()], ctx(), CtrStats())

        assert with_cal[0] < without_cal[0]

    def test_missing_calibrator_still_loads(self, tmp_path):
        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        assert CtrModel(artifact_dir=tmp_path).load() is True

    def test_a_raising_calibrator_falls_back_to_baseline(self, tmp_path):
        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        with (tmp_path / "calibrator.pkl").open("wb") as fh:
            pickle.dump(BrokenCalibrator(), fh)

        model = CtrModel(artifact_dir=tmp_path)
        model.load()
        ctrs, vectors = model.predict_batch([FakeAd()], ctx(), CtrStats())

        assert len(ctrs) == 1
        assert MIN_CTR <= ctrs[0] <= MAX_CTR
        assert len(vectors[0]) == N_FEATURES


class TestInferenceBounds:

    def test_predictions_are_clipped_to_the_serving_range(self, tmp_path):
        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        model = CtrModel(artifact_dir=tmp_path)
        model.load()

        ads = [FakeAd(ad_id=f"ad_{i}") for i in range(20)]
        ctrs, _ = model.predict_batch(ads, ctx(), CtrStats())
        assert all(MIN_CTR <= c <= MAX_CTR for c in ctrs)

    def test_one_ctr_and_one_vector_per_ad_in_order(self, tmp_path):
        write_metadata(tmp_path)
        train_tiny_booster(tmp_path / "model.json")
        model = CtrModel(artifact_dir=tmp_path)
        model.load()

        ads = [FakeAd(ad_id=f"ad_{i}") for i in range(5)]
        ctrs, vectors = model.predict_batch(ads, ctx(), CtrStats())
        assert len(ctrs) == len(vectors) == 5
        assert all(len(v) == N_FEATURES for v in vectors)


class TestDownsamplingCorrection:
    """
    Training downsamples negatives, which biases predicted probabilities upward
    by a known factor. Undoing it is what makes the output a real CTR rather
    than a ranking score — and everything downstream, bid_value included, treats
    it as a real CTR.
    """

    def test_keep_rate_of_one_is_the_identity(self):
        p = np.array([0.1, 0.5, 0.9])
        assert np.allclose(_undo_negative_downsampling(p, 1.0), p)

    def test_correction_lowers_probabilities(self):
        p = np.array([0.5])
        assert _undo_negative_downsampling(p, 0.1)[0] < 0.5

    def test_stays_a_probability(self):
        p = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
        out = _undo_negative_downsampling(p, 0.05)
        assert np.all((out > 0.0) & (out < 1.0))

    def test_preserves_ranking(self):
        # If the correction reordered ads, the auction would change outcome
        # purely as a function of the training sampler.
        p = np.array([0.2, 0.4, 0.6, 0.8])
        out = _undo_negative_downsampling(p, 0.1)
        assert np.all(np.diff(out) > 0)

    def test_handles_the_endpoints_without_dividing_by_zero(self):
        out = _undo_negative_downsampling(np.array([0.0, 1.0]), 0.1)
        assert np.all(np.isfinite(out))


class TestServesTheGatedTrees:
    """
    The model that serves must be the model that passed the promotion gates.

    train_ctr.py fits with `early_stopping_rounds=40`, which leaves the booster
    holding 40 rounds of trees PAST the optimum, and then scores the calibration
    split with `iteration_range=(0, best_iteration + 1)`. Every gate — log loss
    against the baseline, the [0.85, 1.15] calibration ratio, AUC > 0.55 — and
    the isotonic calibrator's fitted domain therefore describe only that prefix.

    Serving with a bare `booster.predict(m)` uses every tree instead, so what
    runs in production is a model nobody measured, differing exactly in the
    rounds early stopping identified as overfitting. It fails silently: scores
    stay plausible, gates stay green, and the miscalibration shows up only as
    bids that are systematically slightly wrong.
    """

    KEYWORDS = ("running", "cooking", "python", "travel", "finance", "music")

    @classmethod
    def _corpus(cls, rng, n):
        """
        Feature vectors built by extract_features, not sampled uniformly.

        Training on uniform noise and predicting on real vectors puts every
        prediction outside the training range, where the booster extrapolates
        to a constant that clips to MAX_CTR — and a constant is equal to itself
        no matter which trees produced it.
        """
        vectors, overlaps = [], []
        for i in range(n):
            ad_kw = tuple(rng.choice(cls.KEYWORDS, size=rng.integers(1, 4), replace=False))
            page_kw = list(rng.choice(cls.KEYWORDS, size=rng.integers(1, 4), replace=False))
            ad = FakeAd(
                ad_id=f"ad_{i % 50}",
                target_cpm=float(rng.uniform(1.0, 12.0)),
                target_keywords=ad_kw,
                spent_today_usd=float(rng.uniform(0.0, 90.0)),
                created_at=REQUEST_TS - timedelta(days=float(rng.uniform(0, 60))),
            )
            request_ctx = RequestContext.build(
                publisher_id="pub_1",
                placement_id=f"place_{i % 7}",
                device_type=rng.choice(["mobile", "desktop", "tablet"]),
                page_keywords=page_kw,
                request_ts=REQUEST_TS - timedelta(hours=float(rng.uniform(0, 240))),
            )
            vectors.append(extract_features(ad, request_ctx, CtrStats()))
            overlaps.append(len(set(ad_kw) & set(request_ctx.page_keywords)))
        return np.asarray(vectors, dtype=np.float32), np.asarray(overlaps)

    @classmethod
    def _booster_past_the_optimum(cls, path: Path):
        """Train well past the optimum, so best_iteration < total rounds."""
        import xgboost as xgb

        rng = np.random.default_rng(3)
        X, overlap = cls._corpus(rng, 4000)
        # A real but weak signal, at a plausible CTR base rate. A high base rate
        # would push predictions into the MAX_CTR clip, where the assertions
        # below hold whichever trees were used.
        y = (rng.random(len(X)) < 0.01 + 0.06 * np.minimum(overlap, 2)).astype(int)

        booster = xgb.train(
            {"objective": "binary:logistic", "eval_metric": "logloss",
             "max_depth": 3, "eta": 0.3},
            xgb.DMatrix(X[:3000], label=y[:3000]),
            num_boost_round=60,
            evals=[(xgb.DMatrix(X[3000:], label=y[3000:]), "valid")],
            early_stopping_rounds=5,
            verbose_eval=False,
        )
        booster.save_model(str(path))
        return booster

    def _varied_ads(self, n=8):
        rng = np.random.default_rng(77)
        return [
            FakeAd(
                ad_id=f"ad_{i}",
                target_cpm=float(rng.uniform(1.0, 12.0)),
                target_keywords=tuple(
                    rng.choice(self.KEYWORDS, size=rng.integers(1, 4), replace=False)),
                spent_today_usd=float(rng.uniform(0.0, 90.0)),
                created_at=REQUEST_TS - timedelta(days=float(rng.uniform(0, 60))),
            )
            for i in range(n)
        ]

    def test_predicts_with_only_the_trees_training_measured(self, tmp_path):
        import xgboost as xgb

        booster = self._booster_past_the_optimum(tmp_path / "model.json")
        assert booster.best_iteration + 1 < booster.num_boosted_rounds(), (
            "fixture did not overshoot the optimum, so this test proves nothing"
        )
        write_metadata(tmp_path, best_iteration=booster.best_iteration)

        model = CtrModel(artifact_dir=tmp_path)
        assert model.load() is True

        served, vectors = model.predict_batch(self._varied_ads(), ctx(), CtrStats())

        matrix = xgb.DMatrix(np.asarray(vectors, dtype=np.float32))
        gated = booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
        every_tree = booster.predict(matrix)

        # Both must sit strictly inside the clip band. Saturated at MAX_CTR the
        # final equality holds whichever trees were used, and the test silently
        # stops testing anything.
        for name, preds in (("gated", gated), ("every_tree", every_tree)):
            assert MIN_CTR < preds.min() and preds.max() < MAX_CTR, (
                f"{name} predictions reached the clip band; the fixture no "
                f"longer distinguishes the two tree counts"
            )
        # And the two must actually differ, or the bug would pass unnoticed.
        assert not np.allclose(gated, every_tree), (
            "the extra trees changed nothing here; the fixture is too weak"
        )

        assert np.allclose(served, gated), (
            "serving path does not match the predictions training gated"
        )

    def test_artifact_without_best_iteration_still_serves_every_tree(self, tmp_path):
        """Artifacts predating this fix carry no best_iteration. They must load."""
        train_tiny_booster(tmp_path / "model.json")
        write_metadata(tmp_path)   # no best_iteration key

        model = CtrModel(artifact_dir=tmp_path)
        assert model.load() is True
        assert model._artifact.iteration_range is None

        ctrs, _ = model.predict_batch([FakeAd()], ctx(), CtrStats())
        assert MIN_CTR <= ctrs[0] <= MAX_CTR

    def test_best_iteration_past_the_last_tree_is_clamped(self, tmp_path):
        """
        xgboost raises when iteration_range runs off the end, and predict_batch
        catches that and degrades to the baseline — so a nonsense best_iteration
        would silently disable the model on every bid. Clamp instead.
        """
        train_tiny_booster(tmp_path / "model.json")   # 2 rounds
        write_metadata(tmp_path, best_iteration=500)

        model = CtrModel(artifact_dir=tmp_path)
        assert model.load() is True
        assert model._artifact.iteration_range == (0, 2)

        ctrs, _ = model.predict_batch([FakeAd()], ctx(), CtrStats())
        assert MIN_CTR <= ctrs[0] <= MAX_CTR
