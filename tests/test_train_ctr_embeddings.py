# test_train_ctr_embeddings.py — the embedding pipeline end to end.
#
# test_embeddings.py covers the table itself. This file covers what only exists
# once the pieces are joined: a training run has to produce an artifact the
# serving code will accept, load and score with, and the six learned columns the
# booster was fit on have to be the six the gateway computes per request.
#
# It runs a real XGBoost pass over synthetic impressions rather than mocking
# one. A mock would be faster and would have caught none of the four things that
# went wrong while this was written.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pytest

from adplatform.ml import train_ctr
from adplatform.ml.artifacts import EMBEDDING_FILE, METADATA_FILE
from adplatform.ml.ctr_model import CtrModel
from adplatform.ml.embeddings import EMBEDDING_BLOCK_SIZE, NEUTRAL_EMBEDDING_BLOCK
from adplatform.ml.features import (
    BASE_FEATURE_NAMES,
    FEATURE_VERSION,
    N_BASE_FEATURES,
    N_FEATURES,
    CtrStats,
    RequestContext,
)

N_ADS, N_PLACEMENTS = 40, 6
PAIR_PRIOR_IDX = BASE_FEATURE_NAMES.index("pair_ctr_prior")


@dataclass
class FakeAd:
    ad_id: str = "ad_1"
    target_cpm: float = 5.0
    target_keywords: tuple = field(default_factory=tuple)
    daily_budget_usd: float = 100.0
    spent_today_usd: float = 0.0
    created_at: datetime | None = None


def synthetic_log(n: int = 60_000, seed: int = 12, n_ads: int = N_ADS,
                  n_placements: int = N_PLACEMENTS) -> dict[str, np.ndarray]:
    """
    An impression log shaped like what load_from_clickhouse returns.

    Clicks come from a low-rank ad x placement interaction plus a device effect.
    pair_ctr_prior below is built from real counts, so the baseline the
    promotion gates compare against knows something real; it just cannot pool
    across pairs, which is the gap the embeddings are meant to close.
    """
    rng = np.random.default_rng(seed)

    u = rng.normal(0, 1, (n_ads, 2))
    v = rng.normal(0, 1, (n_placements, 2))

    ai = rng.integers(0, n_ads, n)
    pi = rng.integers(0, n_placements, n)
    device = rng.integers(0, 3, n)          # 0 mobile, 1 desktop, 2 tablet
    overlap = rng.integers(0, 4, n)

    z = (
        -3.6
        + 0.85 * np.einsum("ij,ij->i", u[ai], v[pi])
        + np.array([0.30, -0.25, 0.0])[device]
        + 0.15 * overlap
    )
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.int8)

    # pair_ctr_prior as the refresh loop would supply it: beta-smoothed counts
    # from an earlier snapshot of the same traffic, not from the future.
    half = n // 2
    imps = np.zeros((n_ads, n_placements))
    clicks = np.zeros((n_ads, n_placements))
    np.add.at(imps, (ai[:half], pi[:half]), 1)
    np.add.at(clicks, (ai[:half], pi[:half]), y[:half])
    prior = (clicks + 0.01 * 200.0) / (imps + 200.0)

    X = np.zeros((n, N_BASE_FEATURES), dtype=np.float32)
    X[:, BASE_FEATURE_NAMES.index("hour_of_day")] = rng.integers(0, 24, n)
    X[:, BASE_FEATURE_NAMES.index("day_of_week")] = rng.integers(0, 7, n)
    X[:, BASE_FEATURE_NAMES.index("is_mobile")] = (device == 0)
    X[:, BASE_FEATURE_NAMES.index("is_desktop")] = (device == 1)
    X[:, BASE_FEATURE_NAMES.index("is_tablet")] = (device == 2)
    X[:, BASE_FEATURE_NAMES.index("keyword_overlap")] = overlap
    X[:, BASE_FEATURE_NAMES.index("n_ad_keywords")] = 3.0
    X[:, BASE_FEATURE_NAMES.index("target_cpm")] = 5.0
    X[:, BASE_FEATURE_NAMES.index("ad_ctr_prior")] = prior[ai].mean()
    X[:, BASE_FEATURE_NAMES.index("placement_ctr_prior")] = 0.02
    X[:, PAIR_PRIOR_IDX] = prior[ai, pi]
    X[:, BASE_FEATURE_NAMES.index("pair_impressions_log")] = np.log1p(imps[ai, pi])

    return {
        "ts_ms": np.arange(n, dtype=np.int64),
        "X": X,
        "feature_version": np.full(n, FEATURE_VERSION, dtype=np.uint16),
        "ad_id": np.array([f"ad_{i}" for i in ai], dtype=object),
        "placement_id": np.array([f"plc_{i}" for i in pi], dtype=object),
        "propensity": np.full(n, 0.9, dtype=np.float32),
        "is_exploration": np.zeros(n, dtype=np.uint8),
        "y": y,
    }


@pytest.fixture(scope="module")
def log_data() -> dict[str, np.ndarray]:
    return synthetic_log()


@pytest.fixture(scope="module")
def trained(log_data, tmp_path_factory) -> tuple[dict, object]:
    out = tmp_path_factory.mktemp("models")
    meta = train_ctr.run(log_data, out, seed=7, embedding_epochs=4)
    return meta, out


@pytest.fixture(scope="module")
def trained_without(log_data, tmp_path_factory) -> tuple[dict, object]:
    out = tmp_path_factory.mktemp("models_control")
    meta = train_ctr.run(log_data, out, seed=7, use_embeddings=False)
    return meta, out


# ---------------------------------------------------------------------------

class TestRowLoading:
    """rows_to_arrays is all that stands between a malformed logged vector and
    a shifted training matrix."""

    def row(self, version: int, width: int, ad="ad_1", placement="plc_1"):
        return (0, [0.5] * width, version, ad, placement, 1.0, 0, 0)

    def test_accepts_the_current_version_at_full_width(self):
        out = train_ctr.rows_to_arrays([self.row(3, N_FEATURES)] * 3)
        assert out["X"].shape == (3, N_BASE_FEATURES)

    def test_accepts_the_previous_version_at_base_width(self):
        # v2 rows stay trainable because the base block kept its indexes and the
        # learned block is recomputed regardless.
        out = train_ctr.rows_to_arrays([self.row(2, N_BASE_FEATURES)] * 3)
        assert out["X"].shape == (3, N_BASE_FEATURES)

    def test_keeps_only_the_base_block_of_a_v3_row(self):
        # The logged learned columns came from the previous model's table. Using
        # them trains the booster on one table and serves it another.
        row = (0, list(range(N_FEATURES)), 3, "ad_1", "plc_1", 1.0, 0, 0)
        out = train_ctr.rows_to_arrays([row])
        np.testing.assert_array_equal(
            out["X"][0], np.arange(N_BASE_FEATURES, dtype=np.float32)
        )

    def test_drops_a_row_whose_width_is_wrong_for_its_version(self):
        rows = [self.row(3, N_FEATURES), self.row(3, N_BASE_FEATURES)]
        out = train_ctr.rows_to_arrays(rows)
        assert len(out["y"]) == 1

    def test_drops_an_unknown_feature_version(self):
        rows = [self.row(3, N_FEATURES), self.row(99, N_FEATURES)]
        assert len(train_ctr.rows_to_arrays(rows)["y"]) == 1

    def test_refuses_a_result_set_with_nothing_usable(self):
        with pytest.raises(SystemExit, match="width"):
            train_ctr.rows_to_arrays([self.row(2, N_FEATURES)])

    def test_ids_survive_as_strings(self):
        out = train_ctr.rows_to_arrays([self.row(3, N_FEATURES, "ad_9", "plc_4")])
        assert out["ad_id"][0] == "ad_9"
        assert out["placement_id"][0] == "plc_4"


class TestBlockAttachment:

    def split(self, n: int = 5) -> dict[str, np.ndarray]:
        return {
            "X": np.zeros((n, N_BASE_FEATURES), dtype=np.float32),
            "y": np.zeros(n, dtype=np.int8),
            "ad_id": np.array([f"ad_{i}" for i in range(n)], dtype=object),
            "placement_id": np.array(["plc_1"] * n, dtype=object),
        }

    def test_widens_to_the_serving_width(self, trained):
        table = _table_from(trained)
        widened = train_ctr.attach_embedding_block(self.split(), table)
        assert widened["X"].shape == (5, N_FEATURES)

    def test_without_a_table_the_width_is_still_right(self):
        # A --no-embeddings artifact still has to load in the same gateway.
        widened = train_ctr.attach_embedding_block(self.split(), None)
        assert widened["X"].shape == (5, N_FEATURES)
        np.testing.assert_allclose(
            widened["X"][:, N_BASE_FEATURES:],
            np.tile(NEUTRAL_EMBEDDING_BLOCK, (5, 1)),
        )

    def test_the_appended_block_is_the_tables_own(self, trained):
        table = _table_from(trained)
        widened = train_ctr.attach_embedding_block(self.split(), table)
        for i in range(5):
            np.testing.assert_allclose(
                widened["X"][i, N_BASE_FEATURES:],
                np.asarray(table.block(f"ad_{i}", "plc_1"), dtype=np.float32),
                rtol=0, atol=1e-6,
            )

    def test_the_base_block_is_left_alone(self, trained):
        split = self.split()
        split["X"][:] = 1.25
        widened = train_ctr.attach_embedding_block(split, _table_from(trained))
        np.testing.assert_array_equal(
            widened["X"][:, :N_BASE_FEATURES], np.full((5, N_BASE_FEATURES), 1.25)
        )


class TestTrainedArtifact:

    def test_ships_the_table_next_to_the_booster(self, trained):
        meta, out = trained
        version_dir = out / meta["model_version"]
        assert (version_dir / EMBEDDING_FILE).exists()
        assert (version_dir / "model.json").exists()

    def test_metadata_declares_the_dependency(self, trained):
        meta, _ = trained
        # The flag that makes a missing table a refusal rather than six columns
        # of constants.
        assert meta["uses_embeddings"] is True
        assert meta["n_features"] == N_FEATURES
        assert meta["feature_version"] == FEATURE_VERSION

    def test_metadata_records_the_fit(self, trained):
        meta, _ = trained
        fit = meta["embedding"]["fit"]
        assert fit["n_ads"] == N_ADS
        assert fit["n_placements"] == N_PLACEMENTS
        assert 1 <= fit["best_epoch"] <= fit["epochs"]
        assert len(meta["embedding"]["block"]) == EMBEDDING_BLOCK_SIZE

    def test_metadata_is_json_serialisable(self, trained):
        # A numpy scalar in the fit report reads back fine here and blows up in
        # json.dumps inside the training container.
        meta, out = trained
        json.loads((out / meta["model_version"] / METADATA_FILE).read_text())

    def test_the_embeddings_alone_beat_the_count_baseline(self, trained):
        # On the held-out calibration rows: pooling across pairs beats
        # counting within them.
        meta, _ = trained
        emb = meta["metrics"]["embeddings_only"]
        baseline = meta["metrics"]["baseline"]
        assert emb is not None
        assert emb["log_loss"] < baseline["log_loss"]
        assert emb["auc"] > baseline["auc"]

    def test_trees_split_on_the_learned_columns(self, trained):
        # A block the model never splits on cost latency for nothing. That is a
        # plausible outcome, not a formality.
        meta, _ = trained
        importance = meta["feature_importance"]
        assert any(name.startswith("emb_") for name in importance)

    def test_the_model_passes_its_promotion_gates(self, trained):
        meta, _ = trained
        assert meta["gate_failures"] == []
        assert meta["promoted"] is True

    def test_control_run_ships_no_table(self, trained_without):
        meta, out = trained_without
        assert meta["uses_embeddings"] is False
        assert meta["embedding"] is None
        assert meta["metrics"]["embeddings_only"] is None
        assert not (out / meta["model_version"] / EMBEDDING_FILE).exists()

    def test_control_run_still_has_the_serving_width(self, trained_without):
        assert trained_without[0]["n_features"] == N_FEATURES


class TestServingTheArtifact:
    """What training wrote, read back by the gateway's own loader."""

    def test_loads_and_scores(self, trained):
        _, out = trained
        model = CtrModel(artifact_dir=out / "current")
        assert model.load() is True
        assert model.is_trained

        ctrs, vectors = model.predict_batch(
            [FakeAd(ad_id="ad_3"), FakeAd(ad_id="ad_7")],
            RequestContext.build("pub", "plc_2", "mobile", ["running"]),
            CtrStats(),
        )
        assert len(ctrs) == 2
        assert all(len(v) == N_FEATURES for v in vectors)

    def test_the_scored_vector_carries_the_artifacts_own_block(self, trained):
        _, out = trained
        model = CtrModel(artifact_dir=out / "current")
        model.load()

        table = model._artifact.embeddings
        assert table is not None

        _, vectors = model.predict_batch(
            [FakeAd(ad_id="ad_3")],
            RequestContext.build("pub", "plc_2", "mobile", []),
            CtrStats(),
        )
        assert vectors[0][N_BASE_FEATURES:] == table.block("ad_3", "plc_2")

    def test_status_reports_the_table(self, trained):
        _, out = trained
        model = CtrModel(artifact_dir=out / "current")
        model.load()
        assert str(N_ADS) in model.status()["embeddings"]

    def test_an_artifact_missing_its_table_is_refused(self, trained, tmp_path):
        # Without the refusal the gateway scores six constant columns into trees
        # fit on real values, and reports a healthy model version while it does.
        import shutil

        _, out = trained
        broken = tmp_path / "broken"
        shutil.copytree(out / "current", broken)
        (broken / EMBEDDING_FILE).unlink()

        model = CtrModel(artifact_dir=broken)
        assert model.load() is False
        assert model.is_trained is False

    def test_a_control_artifact_loads_without_a_table(self, trained_without):
        meta, out = trained_without
        if not meta["promoted"]:
            pytest.skip("control model did not promote; nothing at current/")
        model = CtrModel(artifact_dir=out / "current")
        assert model.load() is True
        assert model._artifact.embeddings is None


class TestWhereTheLiftComesFrom:
    """
    Where the block earns its cost.

    Embeddings are not free: a table to fit, an artifact to ship, six columns
    per eligible ad. What pays for that is generalisation across pairs, so the
    test worth writing is not "embeddings help" but "they help where counting
    stops working". With 40 ads and 6 placements every pair has hundreds of
    impressions, pair_ctr_prior is already a good estimate, and the fixtures
    above show only a slim gain. That is the correct result. Widen the inventory
    until pairs are thin and the gap opens.
    """

    # 5000 pairs over 60k impressions: about a dozen each, and a click rate of
    # a few percent on top of that.
    SPARSE = dict(seed=31, n_ads=250, n_placements=20)

    @staticmethod
    @pytest.fixture(scope="class")
    def sparse_pair(tmp_path_factory):
        sparse = TestWhereTheLiftComesFrom.SPARSE
        with_emb = train_ctr.run(
            synthetic_log(**sparse), tmp_path_factory.mktemp("sparse_emb"),
            seed=7, embedding_epochs=6,
        )
        without = train_ctr.run(
            synthetic_log(**sparse), tmp_path_factory.mktemp("sparse_ctl"),
            seed=7, use_embeddings=False,
        )
        return with_emb, without

    def test_embeddings_beat_the_same_model_without_them(self, sparse_pair):
        with_emb, without = sparse_pair
        assert with_emb["metrics"]["model"]["log_loss"] < \
            without["metrics"]["model"]["log_loss"]

    def test_embeddings_improve_auc(self, sparse_pair):
        with_emb, without = sparse_pair
        assert with_emb["metrics"]["model"]["auc"] > without["metrics"]["model"]["auc"]

    def test_the_learned_columns_are_near_the_top(self, sparse_pair):
        with_emb, _ = sparse_pair
        ranked = list(with_emb["feature_importance"])
        assert any(name.startswith("emb_") for name in ranked[:3])


def _table_from(trained) -> object:
    from adplatform.ml.embeddings import EmbeddingTable

    meta, out = trained
    return EmbeddingTable.load(out / meta["model_version"] / EMBEDDING_FILE)
