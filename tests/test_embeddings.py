# test_embeddings.py — the learned block and the contract around it.
#
# Two failure modes here, and only one of them has a safety net.
#
# If the embeddings learn nothing, the promotion gates in train_ctr.py catch it:
# a table of noise leaves the model no better than the baseline, and it does not
# ship.
#
# If the block computed at training time differs from the block computed at
# serving time, nothing catches it. Both sides produce six plausible floats, the
# gates pass on training's version, and the gateway feeds the trees a different
# one on every request from then on. So most of what follows is one property:
# block(), blocks_for_rows() and a table round-tripped through .npz agree to the
# last bit.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pytest

from adplatform.ml.embeddings import (
    EMBEDDING_BLOCK_SIZE,
    EMBEDDING_FEATURE_NAMES,
    EMBEDDING_FILE,
    EMBEDDING_FORMAT_VERSION,
    NEUTRAL_EMBEDDING_BLOCK,
    OOV_INDEX,
    EmbeddingTable,
    build_vocab,
    train_embeddings,
)
from adplatform.ml.features import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    N_BASE_FEATURES,
    N_FEATURES,
    CtrStats,
    RequestContext,
    extract_features,
    split_blocks,
)


@dataclass
class FakeAd:
    ad_id: str = "ad_1"
    target_cpm: float = 5.0
    target_keywords: tuple = field(default_factory=tuple)
    daily_budget_usd: float = 100.0
    spent_today_usd: float = 0.0
    created_at: datetime | None = None


TS = datetime(2026, 8, 25, 14, 30, tzinfo=timezone.utc)


def ctx(placement_id: str = "plc_1") -> RequestContext:
    return RequestContext.build(
        publisher_id="pub_1", placement_id=placement_id, device_type="mobile",
        page_keywords=["running"], request_ts=TS,
    )


def synthetic_pairs(
    n: int = 40_000, n_ads: int = 50, n_placements: int = 8, dim: int = 2,
    strength: float = 0.9, seed: int = 0,
):
    """
    Impressions whose click probability depends on a low-rank ad x placement
    interaction and nothing else.

    Low-rank on purpose: it is the only structure an FM can represent, and the
    only one worth representing. If every pair had an independent effect there
    would be nothing to generalise from, and counting clicks per pair (which the
    CTR priors already do) would be optimal.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, (n_ads, dim))
    v = rng.normal(0, 1, (n_placements, dim))

    ai = rng.integers(0, n_ads, n)
    pi = rng.integers(0, n_placements, n)
    z = -3.5 + strength * np.einsum("ij,ij->i", u[ai], v[pi])
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int8)

    return (
        [f"ad_{i}" for i in ai],
        [f"plc_{i}" for i in pi],
        y,
        p,
    )


# ---------------------------------------------------------------------------

class TestBlockContract:
    """If any test in this class fails, do not deploy — retrain first."""

    def test_block_width_matches_its_names(self):
        assert len(EMBEDDING_FEATURE_NAMES) == EMBEDDING_BLOCK_SIZE

    def test_neutral_block_width_matches(self):
        assert len(NEUTRAL_EMBEDDING_BLOCK) == EMBEDDING_BLOCK_SIZE

    def test_feature_names_are_base_then_embeddings(self):
        # If this ever fails, train_ctr's TRAINABLE_FEATURE_VERSIONS is a lie
        # and every v2 impression trains on shifted columns.
        assert FEATURE_NAMES[:N_BASE_FEATURES] == BASE_FEATURE_NAMES
        assert FEATURE_NAMES[N_BASE_FEATURES:] == EMBEDDING_FEATURE_NAMES

    def test_total_width_is_the_sum(self):
        assert N_FEATURES == N_BASE_FEATURES + EMBEDDING_BLOCK_SIZE

    def test_names_are_unique_across_the_whole_vector(self):
        # A collision makes features_to_dict lose a column silently.
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_neutral_block_flags_itself_cold(self):
        # Six zeros on their own would read as an FM score that happens to be
        # 0.0. The cold flag marks it as a placeholder.
        neutral = dict(zip(EMBEDDING_FEATURE_NAMES, NEUTRAL_EMBEDDING_BLOCK))
        assert neutral["emb_is_cold"] == 1.0
        assert neutral["emb_logit"] == 0.0


class TestVocabulary:

    def test_rare_ids_fall_into_the_oov_row(self):
        ids = ["a"] * 30 + ["b"] * 5
        vocab = build_vocab(ids, min_count=20)
        assert "a" in vocab
        assert "b" not in vocab

    def test_no_id_is_ever_assigned_row_zero(self):
        # Row 0 is the shared OOV bucket, and an id landing there merges its
        # history with every rare id in the table.
        vocab = build_vocab(["a"] * 50 + ["b"] * 50, min_count=1)
        assert OOV_INDEX not in vocab.values()
        assert sorted(vocab.values()) == [1, 2]

    def test_empty_ids_are_excluded(self):
        assert "" not in build_vocab([""] * 100, min_count=1)

    def test_vocabulary_is_order_independent(self):
        # Otherwise a retrain on identical data produces a table whose rows
        # mean different things.
        rng = np.random.default_rng(1)
        ids = [f"ad_{i%7}" for i in range(700)]
        shuffled = list(rng.permutation(ids))
        assert build_vocab(ids, 10) == build_vocab(shuffled, 10)


class TestFitting:

    def test_recovers_a_low_rank_interaction(self):
        from sklearn.metrics import log_loss, roc_auc_score

        ads, placements, y, true_p = synthetic_pairs()
        table = train_embeddings(ads, placements, y, dim=4, epochs=6, seed=3)
        pred = table.predict_proba(ads, placements)

        constant = np.full(len(y), y.mean())
        # The bar is the global rate, which is what an ad-level count prior
        # collapses to when the signal lives in the pair.
        assert log_loss(y, pred) < log_loss(y, constant)
        # And most of the way to the oracle that generated the data.
        assert roc_auc_score(y, pred) > 0.5 + 0.8 * (roc_auc_score(y, true_p) - 0.5)

    def test_learns_nothing_from_labels_that_carry_nothing(self):
        # Scored on rows the fit never saw. In-sample it will separate them;
        # a few thousand free parameters against pure noise always can. That is
        # why train_ctr.py fits on the training split alone and evaluates on the
        # calibration split.
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(4)
        n, cut = 25_000, 20_000
        ads = [f"ad_{int(i)}" for i in rng.integers(0, 40, n)]
        placements = [f"plc_{int(i)}" for i in rng.integers(0, 6, n)]
        y = (rng.random(n) < 0.05).astype(np.int8)

        table = train_embeddings(ads[:cut], placements[:cut], y[:cut],
                                 dim=4, epochs=4, seed=5)
        held_out = table.predict_proba(ads[cut:], placements[cut:])
        assert 0.44 < roc_auc_score(y[cut:], held_out) < 0.56

    def test_is_deterministic_under_a_fixed_seed(self):
        ads, placements, y, _ = synthetic_pairs(n=6_000, seed=2)
        kwargs = dict(dim=3, epochs=2, seed=11)
        a = train_embeddings(ads, placements, y, **kwargs)
        b = train_embeddings(ads, placements, y, **kwargs)
        np.testing.assert_array_equal(a.ad_vectors, b.ad_vectors)
        np.testing.assert_array_equal(a.placement_bias, b.placement_bias)

    def test_holdout_is_the_tail_not_a_sample(self):
        # The epoch chosen has to be the one that generalises forward.
        ads, placements, y, _ = synthetic_pairs(n=20_000)
        report = train_embeddings(ads, placements, y, epochs=3).fit_report
        assert report["rows_holdout"] == 2_000
        assert report["rows_fit"] == 18_000

    def test_reports_the_epoch_it_kept(self):
        ads, placements, y, _ = synthetic_pairs(n=20_000)
        report = train_embeddings(ads, placements, y, dim=4, epochs=5).fit_report
        assert 1 <= report["best_epoch"] <= 5
        assert len(report["history"]) == 5

    def test_rejects_ragged_input(self):
        with pytest.raises(ValueError, match="ragged"):
            train_embeddings(["a", "b"], ["p"], [0, 1])

    def test_rejects_an_empty_log(self):
        with pytest.raises(ValueError, match="no rows"):
            train_embeddings([], [], [])


class TestTrainServeAgreement:
    """The one property nothing downstream can check for itself."""

    @staticmethod
    @pytest.fixture(scope="class")
    def table():
        ads, placements, y, _ = synthetic_pairs(n=12_000, seed=6)
        return train_embeddings(ads, placements, y, dim=4, epochs=3, seed=7)

    def test_scalar_and_batch_paths_agree(self, table):
        # blocks_for_rows builds the training matrix, block() serves each
        # request. A discrepancy here is train/serve skew, directly.
        ads = ["ad_1", "ad_2", "ad_1", "unseen_ad"]
        placements = ["plc_1", "plc_3", "plc_3", "plc_1"]
        batch = table.blocks_for_rows(ads, placements)
        for i, (a, p) in enumerate(zip(ads, placements)):
            np.testing.assert_allclose(
                batch[i], np.asarray(table.block(a, p), dtype=np.float32),
                rtol=0, atol=1e-6,
            )

    def test_a_round_trip_through_disk_changes_nothing(self, table, tmp_path):
        # The artifact crosses a process boundary, trained in a CronJob and
        # served in a gateway, so equality has to survive the file.
        path = table.save(tmp_path / EMBEDDING_FILE)
        reloaded = EmbeddingTable.load(path)
        for a, p in [("ad_1", "plc_1"), ("ad_9", "plc_4"), ("gone", "plc_0")]:
            assert reloaded.block(a, p) == table.block(a, p)
        assert reloaded.dim == table.dim
        assert reloaded.version == table.version
        assert reloaded.fit_report == table.fit_report

    def test_the_file_loads_without_pickle(self, table, tmp_path):
        # An artifact fetched from S3 is deserialised inside the serving
        # process, where allow_pickle would turn a bucket write into remote
        # code execution.
        path = table.save(tmp_path / EMBEDDING_FILE)
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) >= {"ad_vectors", "ad_bias", "sidecar"}

    def test_a_future_format_version_is_refused(self, table, tmp_path):
        path = table.save(tmp_path / EMBEDDING_FILE)
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        sidecar = json.loads(str(arrays["sidecar"]))
        sidecar["format_version"] = EMBEDDING_FORMAT_VERSION + 1
        arrays["sidecar"] = np.asarray(json.dumps(sidecar))
        np.savez(path, **arrays)

        with pytest.raises(ValueError, match="format version"):
            EmbeddingTable.load(path)


class TestColdStart:

    @staticmethod
    @pytest.fixture(scope="class")
    def table():
        rng = np.random.default_rng(9)
        ads = [f"ad_{i%4}" for i in range(8_000)] + [f"rare_{i}" for i in range(300)]
        placements = [f"plc_{i%3}" for i in range(8_300)]
        y = (rng.random(8_300) < 0.05).astype(np.int8)
        return train_embeddings(ads, placements, y, dim=3, epochs=2,
                                min_count=50, seed=2)

    def test_an_unseen_ad_is_flagged_cold(self, table):
        block = dict(zip(EMBEDDING_FEATURE_NAMES, table.block("brand_new", "plc_1")))
        assert block["emb_is_cold"] == 1.0

    def test_an_unseen_placement_is_flagged_cold(self, table):
        block = dict(zip(EMBEDDING_FEATURE_NAMES, table.block("ad_0", "plc_999")))
        assert block["emb_is_cold"] == 1.0

    def test_a_known_pair_is_not_flagged(self, table):
        block = dict(zip(EMBEDDING_FEATURE_NAMES, table.block("ad_0", "plc_1")))
        assert block["emb_is_cold"] == 0.0

    def test_the_tail_shares_one_row(self, table):
        # Every rare id resolves to the same vector, so a cold ad predicts like
        # the average cold ad rather than like the origin.
        assert table.block("rare_7", "plc_1") == table.block("rare_8", "plc_1")

    def test_a_cold_lookup_never_raises_or_returns_nan(self, table):
        block = table.block("中文", "")
        assert len(block) == EMBEDDING_BLOCK_SIZE
        assert all(np.isfinite(block))


class TestValidation:

    def base(self, **overrides) -> dict:
        kwargs = dict(
            dim=2,
            ad_vocab={"a": 1},
            placement_vocab={"p": 1},
            ad_vectors=np.zeros((2, 2), dtype=np.float32),
            placement_vectors=np.zeros((2, 2), dtype=np.float32),
            ad_bias=np.zeros(2, dtype=np.float32),
            placement_bias=np.zeros(2, dtype=np.float32),
            global_bias=-4.0,
        )
        kwargs.update(overrides)
        return kwargs

    def test_a_well_formed_table_validates(self):
        EmbeddingTable(**self.base()).validate()

    def test_vocabulary_past_the_end_of_the_matrix_is_caught(self):
        # Otherwise this surfaces as an IndexError on one unlucky request
        # months later.
        with pytest.raises(ValueError, match="indexes past"):
            EmbeddingTable(**self.base(ad_vocab={"a": 5})).validate()

    def test_width_disagreeing_with_dim_is_caught(self):
        with pytest.raises(ValueError, match="does not match dim"):
            EmbeddingTable(**self.base(dim=8)).validate()

    def test_bias_length_disagreement_is_caught(self):
        with pytest.raises(ValueError, match="bias vectors"):
            EmbeddingTable(**self.base(ad_bias=np.zeros(5, dtype=np.float32))).validate()

    def test_an_id_mapped_to_the_oov_row_is_caught(self):
        with pytest.raises(ValueError, match="row 0 is reserved"):
            EmbeddingTable(**self.base(ad_vocab={"a": 0})).validate()


class TestFeatureIntegration:
    """extract_features is still the only place features are computed."""

    @staticmethod
    @pytest.fixture(scope="class")
    def table():
        ads, placements, y, _ = synthetic_pairs(n=8_000, seed=8)
        return train_embeddings(ads, placements, y, dim=3, epochs=2, seed=1)

    def test_vector_is_the_full_width_with_a_table(self, table):
        vec = extract_features(FakeAd(), ctx(), CtrStats(), table)
        assert len(vec) == N_FEATURES

    def test_vector_is_the_full_width_without_one(self):
        # A gateway with no model loaded still logs vectors, and those rows
        # have to be trainable later.
        assert len(extract_features(FakeAd(), ctx(), CtrStats())) == N_FEATURES

    def test_the_learned_half_is_exactly_the_table_block(self, table):
        ad = FakeAd(ad_id="ad_3")
        _, learned = split_blocks(extract_features(ad, ctx("plc_2"), CtrStats(), table))
        assert learned == table.block("ad_3", "plc_2")

    def test_no_table_means_the_neutral_block(self):
        _, learned = split_blocks(extract_features(FakeAd(), ctx(), CtrStats()))
        assert learned == list(NEUTRAL_EMBEDDING_BLOCK)

    def test_the_base_half_is_untouched_by_the_table(self, table):
        # If the bump perturbed even one hand-written column, every v2
        # impression in ClickHouse becomes untrainable.
        with_table, _ = split_blocks(
            extract_features(FakeAd(), ctx(), CtrStats(), table)
        )
        without, _ = split_blocks(extract_features(FakeAd(), ctx(), CtrStats()))
        assert with_table == without
        assert len(with_table) == N_BASE_FEATURES

    def test_every_value_is_finite(self, table):
        vec = extract_features(FakeAd(ad_id="never_seen"), ctx("nowhere"),
                               CtrStats(), table)
        assert all(np.isfinite(v) for v in vec)
