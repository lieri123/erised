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
from datetime import datetime, timezone
from pathlib import Path

import pytest

from adplatform.ml.artifacts import LocalArtifactStore
from adplatform.ml.ctr_model import MAX_CTR, MIN_CTR, CtrModel, _undo_negative_downsampling
from adplatform.ml.features import FEATURE_VERSION, N_FEATURES, CtrStats, RequestContext

import numpy as np


@dataclass
class FakeAd:
    ad_id: str = "ad_1"
    target_cpm: float = 5.0
    target_keywords: tuple = field(default_factory=tuple)
    daily_budget_usd: float = 100.0
    spent_today_usd: float = 0.0
    created_at: datetime | None = None


def ctx() -> RequestContext:
    return RequestContext.build(
        publisher_id="pub_1", placement_id="place_1", device_type="mobile",
        page_keywords=["running"],
        request_ts=datetime(2026, 8, 25, 14, 30, tzinfo=timezone.utc),
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
