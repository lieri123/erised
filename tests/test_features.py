# test_features.py — the train/serve skew defence.
#
# extract_features is the only place features are computed, and that is the
# whole design: the serving path logs the vector it actually scored, and the
# training job consumes that vector verbatim. Nothing recomputes.
#
# Which means one invariant carries everything: the vector's length and ORDER
# must match FEATURE_NAMES. Insert a feature in the middle of the return list
# without updating FEATURE_NAMES and every downstream consumer silently reads
# the wrong column. No exception, no error — just a model trained on shifted
# data, and metrics that look plausible because they degrade smoothly.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from adplatform.ml.features import (
    FEATURE_NAMES,
    N_FEATURES,
    CtrStats,
    RequestContext,
    extract_features,
    features_to_dict,
)


@dataclass
class FakeAd:
    ad_id: str = "ad_1"
    target_cpm: float = 5.0
    target_keywords: tuple = field(default_factory=tuple)
    daily_budget_usd: float = 100.0
    spent_today_usd: float = 0.0
    created_at: datetime | None = None


TS = datetime(2026, 8, 25, 14, 30, tzinfo=timezone.utc)   # a Tuesday


def ctx(**kwargs) -> RequestContext:
    defaults = dict(
        publisher_id="pub_1",
        placement_id="place_1",
        device_type="mobile",
        page_keywords=["running", "shoes"],
        request_ts=TS,
    )
    defaults.update(kwargs)
    return RequestContext.build(**defaults)


def feat(ad, context=None, stats=None) -> dict[str, float]:
    """Features by name, so tests never index by position."""
    return features_to_dict(extract_features(ad, context or ctx(), stats or CtrStats()))


# ---------------------------------------------------------------------------

class TestVectorContract:
    """If any test in this class fails, do not deploy — retrain first."""

    def test_length_matches_feature_names(self):
        assert len(extract_features(FakeAd(), ctx())) == len(FEATURE_NAMES)

    def test_length_matches_n_features(self):
        assert len(extract_features(FakeAd(), ctx())) == N_FEATURES

    def test_n_features_matches_feature_names(self):
        assert N_FEATURES == len(FEATURE_NAMES)

    def test_feature_names_are_unique(self):
        # A duplicate name makes features_to_dict lose a column silently.
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_every_value_is_a_finite_float(self):
        # NaN reaches XGBoost as a missing value and inf poisons a split. Both
        # are survivable at predict time and disastrous at train time.
        for value in extract_features(FakeAd(), ctx()):
            assert isinstance(value, float)
            assert math.isfinite(value)

    def test_length_is_stable_across_wildly_different_inputs(self):
        cases = [
            (FakeAd(), ctx()),
            (FakeAd(target_keywords=()), ctx(page_keywords=[])),
            (FakeAd(target_keywords=tuple(f"k{i}" for i in range(100))),
             ctx(page_keywords=[f"k{i}" for i in range(100)])),
            (FakeAd(daily_budget_usd=0.0), ctx(device_type="")),
            (FakeAd(created_at=None), ctx(device_type="smart-fridge")),
        ]
        for ad, context in cases:
            assert len(extract_features(ad, context)) == N_FEATURES


class TestDeterminism:

    def test_same_inputs_give_the_same_vector(self):
        ad, context = FakeAd(target_keywords=("running",)), ctx()
        assert extract_features(ad, context) == extract_features(ad, context)

    def test_page_keyword_order_does_not_matter(self):
        # RequestContext.build sorts and dedupes, so the caller's ordering
        # cannot leak into the model. Without this, the same page could produce
        # two different vectors depending on how the tag serialised its list.
        a = extract_features(FakeAd(), ctx(page_keywords=["shoes", "running"]))
        b = extract_features(FakeAd(), ctx(page_keywords=["running", "shoes"]))
        assert a == b

    def test_duplicate_page_keywords_are_collapsed(self):
        one = feat(FakeAd(), ctx(page_keywords=["running"]))
        many = feat(FakeAd(), ctx(page_keywords=["running", "running", "Running"]))
        assert one["n_page_keywords"] == many["n_page_keywords"] == 1.0


class TestTimeFeatures:

    def test_hour_and_day_come_from_the_request_timestamp(self):
        f = feat(FakeAd(), ctx())
        assert f["hour_of_day"] == 14.0
        assert f["day_of_week"] == 1.0        # Tuesday, Monday=0

    def test_weekend_flag(self):
        saturday = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
        sunday = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
        assert feat(FakeAd(), ctx(request_ts=saturday))["is_weekend"] == 1.0
        assert feat(FakeAd(), ctx(request_ts=sunday))["is_weekend"] == 1.0
        assert feat(FakeAd(), ctx())["is_weekend"] == 0.0

    def test_ad_age_in_days(self):
        ad = FakeAd(created_at=TS - timedelta(days=7, hours=12))
        assert feat(ad, ctx())["ad_age_days"] == pytest.approx(7.5)

    def test_naive_created_at_is_treated_as_utc(self):
        # Postgres can hand back a naive datetime depending on the column type.
        # Subtracting it from an aware ts raises TypeError, which would 500 a
        # bid — so the code coerces rather than trusting the driver.
        ad = FakeAd(created_at=datetime(2026, 8, 18, 14, 30))
        assert feat(ad, ctx())["ad_age_days"] == pytest.approx(7.0)

    def test_missing_created_at_gives_zero_age(self):
        assert feat(FakeAd(created_at=None), ctx())["ad_age_days"] == 0.0

    def test_future_created_at_clamps_to_zero(self):
        # Clock skew between the app and the database. A negative age is a
        # value the model never saw in training.
        ad = FakeAd(created_at=TS + timedelta(days=3))
        assert feat(ad, ctx())["ad_age_days"] == 0.0


class TestDeviceFeatures:

    @pytest.mark.parametrize("device,expected", [
        ("mobile", ("is_mobile",)),
        ("desktop", ("is_desktop",)),
        ("tablet", ("is_tablet",)),
    ])
    def test_one_hot_is_exclusive(self, device, expected):
        f = feat(FakeAd(), ctx(device_type=device))
        for name in ("is_mobile", "is_desktop", "is_tablet"):
            assert f[name] == (1.0 if name in expected else 0.0)

    def test_device_matching_is_case_insensitive(self):
        assert feat(FakeAd(), ctx(device_type="MOBILE"))["is_mobile"] == 1.0

    def test_unknown_device_sets_no_flag(self):
        f = feat(FakeAd(), ctx(device_type="smart-fridge"))
        assert f["is_mobile"] == f["is_desktop"] == f["is_tablet"] == 0.0


class TestKeywordFeatures:

    def test_overlap_counts_shared_keywords(self):
        ad = FakeAd(target_keywords=("running", "shoes", "marathon"))
        assert feat(ad, ctx(page_keywords=["running", "shoes"]))["keyword_overlap"] == 2.0

    def test_overlap_ratio_is_over_ad_keywords(self):
        ad = FakeAd(target_keywords=("running", "shoes", "marathon", "trail"))
        f = feat(ad, ctx(page_keywords=["running", "shoes"]))
        assert f["keyword_overlap_ratio"] == pytest.approx(0.5)

    def test_no_ad_keywords_gives_zero_ratio_not_a_crash(self):
        # Division by len(ad_kws). An untargeted campaign is legal.
        f = feat(FakeAd(target_keywords=()), ctx())
        assert f["keyword_overlap_ratio"] == 0.0
        assert f["n_ad_keywords"] == 0.0

    def test_ad_keywords_are_lowercased_at_scoring_time(self):
        # Campaign writes normalise keywords, but inventory predating that
        # validator holds mixed case. extract_features compensates. This
        # asymmetry — ad keywords lowercased here, page keywords lowercased in
        # RequestContext.build — is deliberate and must be preserved by any
        # reimplementation, including the Rust port.
        ad = FakeAd(target_keywords=("Running", "SHOES"))
        assert feat(ad, ctx(page_keywords=["running", "shoes"]))["keyword_overlap"] == 2.0

    def test_empty_page_keywords_give_zero_overlap(self):
        ad = FakeAd(target_keywords=("running",))
        f = feat(ad, ctx(page_keywords=[]))
        assert f["keyword_overlap"] == 0.0
        assert f["n_page_keywords"] == 0.0

    def test_blank_page_keywords_are_dropped(self):
        f = feat(FakeAd(), ctx(page_keywords=["running", "  ", ""]))
        assert f["n_page_keywords"] == 1.0


class TestCtrPriors:

    def test_unseen_ad_falls_back_to_the_global_rate(self):
        stats = CtrStats(global_ctr=0.02)
        assert feat(FakeAd(), ctx(), stats)["ad_ctr_prior"] == pytest.approx(0.02)

    def test_smoothing_pulls_small_samples_toward_the_global_rate(self):
        # 1 click in 10 impressions is 10% raw. With prior_strength 200 the
        # smoothed value must sit far closer to the 1% global rate — otherwise
        # a fluke click on a brand new ad dominates its score.
        stats = CtrStats(global_ctr=0.01, prior_strength=200.0,
                         ad_counts={"ad_1": (10, 1)})
        prior = feat(FakeAd(ad_id="ad_1"), ctx(), stats)["ad_ctr_prior"]
        assert 0.01 < prior < 0.03

    def test_large_samples_dominate_the_prior(self):
        stats = CtrStats(global_ctr=0.01, prior_strength=200.0,
                         ad_counts={"ad_1": (1_000_000, 50_000)})
        prior = feat(FakeAd(ad_id="ad_1"), ctx(), stats)["ad_ctr_prior"]
        assert prior == pytest.approx(0.05, abs=0.001)

    def test_pair_impressions_are_logged_not_raw(self):
        # log1p, so the model can learn "trust the pair prior more as evidence
        # accumulates" without the feature spanning six orders of magnitude.
        stats = CtrStats(pair_counts={("ad_1", "place_1"): (999, 10)})
        f = feat(FakeAd(ad_id="ad_1"), ctx(), stats)
        assert f["pair_impressions_log"] == pytest.approx(math.log1p(999))

    def test_zero_pair_impressions_log_to_zero(self):
        assert feat(FakeAd(), ctx(), CtrStats())["pair_impressions_log"] == 0.0

    def test_priors_stay_within_zero_and_one(self):
        stats = CtrStats(global_ctr=0.5, prior_strength=1.0,
                         ad_counts={"ad_1": (3, 3)},
                         pair_counts={("ad_1", "place_1"): (1, 1)})
        f = feat(FakeAd(ad_id="ad_1"), ctx(), stats)
        for name in ("ad_ctr_prior", "placement_ctr_prior", "pair_ctr_prior"):
            assert 0.0 <= f[name] <= 1.0


class TestBudgetPacing:

    def test_pacing_is_spend_over_budget(self):
        ad = FakeAd(daily_budget_usd=100.0, spent_today_usd=25.0)
        assert feat(ad, ctx())["budget_pacing"] == pytest.approx(0.25)

    def test_zero_budget_gives_zero_not_a_division_error(self):
        assert feat(FakeAd(daily_budget_usd=0.0), ctx())["budget_pacing"] == 0.0

    def test_overspend_exceeds_one(self):
        # Budget enforcement is eventually consistent — Redis counters can
        # overshoot before reconciliation. The feature is documented as 0.0-1.0+
        # and the model must see the real value, not a clamped one.
        ad = FakeAd(daily_budget_usd=100.0, spent_today_usd=110.0)
        assert feat(ad, ctx())["budget_pacing"] == pytest.approx(1.1)

    def test_missing_attributes_default_to_zero(self):
        class Bare:
            ad_id = "ad_1"
            target_cpm = 5.0
            target_keywords = ()

        assert feat(Bare(), ctx())["budget_pacing"] == 0.0


class TestFeaturesToDict:

    def test_round_trips_positionally(self):
        vec = extract_features(FakeAd(), ctx())
        as_dict = features_to_dict(vec)
        assert list(as_dict.keys()) == list(FEATURE_NAMES)
        assert list(as_dict.values()) == vec
