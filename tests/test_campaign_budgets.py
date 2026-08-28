# tests/test_campaign_budgets.py
#
# The bug: daily_budget_usd is a campaign property, but budget.py keyed Redis on
# ad_id. A campaign with N creatives got N independent counters, each compared
# against the campaign's full budget -- N x overspend, with every counter
# reporting itself healthy.
#
# It was dormant while get_eligible_ads served MOCK_ADS, because those had no
# campaigns at all: one ad, one budget, accidentally correct. Wiring up real
# inventory is what activated it.
#
# test_three_creatives_share_one_budget is the regression test. It fails on the
# old ad-keyed code with kept == 3.

import pytest

from adplatform.ml import budget
from adplatform.ml.budget import BudgetTracker, budget_key, filter_by_budget
from adplatform.rtb import Ad

DAY = "2026-08-01"


class FakeRedis:
    """In-memory stand-in supporting the four operations BudgetTracker uses."""

    def __init__(self, initial=None, fail=False):
        self.store = dict(initial or {})
        self.fail = fail
        self.mget_calls = 0

    async def mget(self, keys):
        if self.fail:
            raise ConnectionError("redis down")
        self.mget_calls += 1
        return [self.store.get(k) for k in keys]

    async def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = str(value)

    def register_script(self, _src):
        async def run(keys, args):
            k, amount = keys[0], float(args[0])
            self.store[k] = str(float(self.store.get(k, 0.0)) + amount)
            return self.store[k]
        return run


def ad(ad_id, campaign_id, budget_usd=100.0):
    return Ad(
        ad_id=ad_id,
        advertiser_id="adv_1",
        creative_html="<div/>",
        destination_url="https://example.com",
        target_cpm=5.0,
        floor_price=1.0,
        target_device="all",
        daily_budget_usd=budget_usd,
        campaign_id=campaign_id,
    )


# --- the regression -------------------------------------------------------

async def test_three_creatives_share_one_budget():
    """
    One campaign, $100/day, three creatives, $100 already spent. All three must
    be excluded. Ad-keyed code kept all three: each creative had its own empty
    counter and read $0 spent against a $100 budget.
    """
    tracker = BudgetTracker(FakeRedis({budget_key("camp_1"): "100.0"}))
    ads = [ad(f"ad_{i}", "camp_1", 100.0) for i in "abc"]

    assert await filter_by_budget(ads, tracker) == []


async def test_spend_on_one_creative_counts_against_its_siblings():
    """The whole point: creatives in a campaign draw down a shared pot."""
    redis = FakeRedis()
    tracker = BudgetTracker(redis)

    await tracker.record_spend("camp_1", 99.5)

    ads = [ad("ad_a", "camp_1", 100.0), ad("ad_b", "camp_1", 100.0)]
    assert len(await filter_by_budget(ads, tracker)) == 2  # $99.50 < $100

    await tracker.record_spend("camp_1", 1.0)             # now $100.50
    assert await filter_by_budget(ads, tracker) == []


async def test_campaigns_are_independent():
    """Exhausting one campaign must not affect another."""
    tracker = BudgetTracker(FakeRedis({budget_key("camp_broke"): "500.0"}))
    ads = [ad("ad_1", "camp_broke", 500.0), ad("ad_2", "camp_rich", 500.0)]

    kept = await filter_by_budget(ads, tracker)
    assert [a.ad_id for a in kept] == ["ad_2"]


# --- keys -----------------------------------------------------------------

def test_key_namespace_is_distinct_from_the_old_one():
    """
    "budget:camp:" not "budget:". A campaign_id that happens to equal a legacy
    ad_id must not inherit its spend.
    """
    key = budget_key("camp_1", DAY)
    assert key == f"budget:camp:camp_1:{DAY}"
    assert not key.startswith(f"budget:camp_1")


def test_keys_are_per_utc_day():
    assert budget_key("camp_1", "2026-08-01") != budget_key("camp_1", "2026-08-02")


# --- read path ------------------------------------------------------------

async def test_one_mget_regardless_of_candidate_count():
    redis = FakeRedis()
    tracker = BudgetTracker(redis)
    ads = [ad(f"ad_{i}", f"camp_{i % 4}") for i in range(40)]

    await filter_by_budget(ads, tracker)

    assert redis.mget_calls == 1, "budget read must not be per-ad"


async def test_redis_failure_fails_open():
    """A cache outage must not become a revenue outage."""
    tracker = BudgetTracker(FakeRedis(fail=True))
    ads = [ad("ad_a", "camp_1"), ad("ad_b", "camp_2")]

    assert len(await filter_by_budget(ads, tracker)) == 2


async def test_ad_without_campaign_is_excluded():
    """
    Cannot be charged, so must not serve. Should be impossible -- servable_ads
    joins through campaigns -- so it means a malformed inventory row.
    """
    tracker = BudgetTracker(FakeRedis())
    orphan = ad("ad_orphan", "camp_1")
    orphan.campaign_id = None

    kept = await filter_by_budget([orphan, ad("ad_ok", "camp_1")], tracker)
    assert [a.ad_id for a in kept] == ["ad_ok"]


async def test_record_spend_without_campaign_is_refused():
    redis = FakeRedis()
    tracker = BudgetTracker(redis)

    assert await tracker.record_spend("", 1.0) is None
    assert redis.store == {}, "spend must not land under an empty key"


async def test_empty_candidate_set():
    assert await filter_by_budget([], BudgetTracker(FakeRedis())) == []


# --- reconcile ------------------------------------------------------------

class FakeCH:
    def __init__(self, rows):
        self.rows = rows
        self.query_text = None

    def query(self, query, _params=None):
        self.query_text = query

        class R:
            result_rows = self.rows
        return R()


async def test_reconcile_writes_campaign_keys():
    """
    Reconcile must write the SAME keys enforcement reads. Grouping by ad_id
    while enforcement reads campaign keys is worse than not reconciling: the
    campaign keys would never be rebuilt after a Redis restart, so every
    campaign would reset to zero and serve a second full budget.
    """
    redis = FakeRedis()
    tracker = BudgetTracker(redis)
    ch = FakeCH([("camp_1", 12.5), ("camp_2", 3.0)])

    corrected = await tracker.reconcile(ch, day=DAY)

    assert corrected == {"camp_1": 12.5, "camp_2": 3.0}
    assert redis.store[budget_key("camp_1", DAY)] == "12.5"
    assert "GROUP BY campaign_id" in ch.query_text


async def test_reconcile_corrects_a_counter_upward():
    """
    What reconciliation is actually for. record_spend is fire-and-forget, so a
    crash between the auction and the increment loses it; the impression log
    still has the row, and reconcile puts the money back.
    """
    redis = FakeRedis({budget_key("camp_1", DAY): "5.0"})
    tracker = BudgetTracker(redis)

    await tracker.reconcile(FakeCH([("camp_1", 12.5)]), day=DAY)

    assert redis.store[budget_key("camp_1", DAY)] == "12.5"


async def test_reconcile_refuses_to_lower_a_counter():
    """
    This assertion is the reverse of the one it replaces, on purpose.

    The old contract was "Redis is a cache; the impression log is the source of
    truth" -- but ad_impressions arrives over Kafka, and publish_event drops
    events silently when the broker is unreachable. The log is lossy in exactly
    one direction: it can be missing impressions, never invent them.

    So a lower ClickHouse total is not evidence that Redis overcounted. It is
    the shape a broker outage makes. Overwriting downward hands the campaign
    every dropped impression's budget back, and nothing anywhere reports an
    error, because from Redis's point of view nothing failed.
    """
    redis = FakeRedis({budget_key("camp_1", DAY): "999.0"})
    tracker = BudgetTracker(redis)

    corrected = await tracker.reconcile(FakeCH([("camp_1", 12.5)]), day=DAY)

    assert redis.store[budget_key("camp_1", DAY)] == "999.0"
    assert corrected["camp_1"] == 999.0, "the caller must see the figure in force"


async def test_a_refusal_is_logged_as_a_warning(caplog):
    """
    The one symptom of dropped events that reaches an operator. INFO would put
    it under the per-request log line the gateway emits for every bid.
    """
    import logging

    tracker = BudgetTracker(FakeRedis({budget_key("camp_1", DAY): "999.0"}))

    with caplog.at_level(logging.WARNING, logger="adplatform.ml.budget"):
        await tracker.reconcile(FakeCH([("camp_1", 12.5)]), day=DAY)

    messages = [r.getMessage() for r in caplog.records]
    assert any("refused to lower" in m for m in messages), messages
    assert any("999" in m and "12.5" in m for m in messages), (
        "the warning must carry both figures, or it cannot be acted on"
    )


async def test_float_noise_is_not_treated_as_drift():
    """
    INCRBYFLOAT and ClickHouse sum() will not agree to the last bit on the same
    impressions. A sub-cent difference must not trip the refusal path.
    """
    redis = FakeRedis({budget_key("camp_1", DAY): "12.5"})
    tracker = BudgetTracker(redis)

    await tracker.reconcile(FakeCH([("camp_1", 12.4999)]), day=DAY)

    assert float(redis.store[budget_key("camp_1", DAY)]) == pytest.approx(12.4999)


async def test_reconcile_excludes_pre_migration_rows():
    """
    Impressions logged before migration 003 carry campaign_id = ''. They have no
    campaign to charge, and folding them into one would overstate it.
    """
    ch = FakeCH([("camp_1", 5.0)])
    await BudgetTracker(FakeRedis()).reconcile(ch, day=DAY)

    assert "campaign_id != ''" in ch.query_text


async def test_reconcile_survives_a_failed_query():
    tracker = BudgetTracker(FakeRedis())

    class Broken:
        def query(self, *a, **k):
            raise ConnectionError("clickhouse down")

    assert await tracker.reconcile(Broken(), day=DAY) == {}


# --- the CPM/dollars distinction -----------------------------------------

def test_impression_cost_is_one_thousandth_of_cpm():
    """
    win_price is a CPM. Decrementing a budget by win_price overcharges 1000x and
    burns a $500 daily budget in about 125 impressions.
    """
    assert budget.impression_cost_usd(13.42) == pytest.approx(0.01342)


# --- the spend stamp ------------------------------------------------------
#
# `budget_pacing` is one of the 17 features in FEATURE_NAMES, and features.py
# builds it from `ad.spent_today_usd / ad.daily_budget_usd`. inventory._row_to_ad
# sets spent_today_usd to 0.0 and defers to filter_by_budget to fill it in --
# this is the only place on the serving path that ever reads a campaign's spend.
#
# When filter_by_budget dropped that value on the floor, budget_pacing was
# constant 0.0 in production: logged into every impression row, trained on as a
# dead column, and served as one. Nothing failed. test_features.py passed
# throughout, because it hands extract_features an ad with the field already
# populated and never exercises the wiring that populates it.

async def test_surviving_ads_carry_todays_spend():
    tracker = BudgetTracker(FakeRedis({budget_key("camp_1"): "25.0"}))

    kept = await filter_by_budget([ad("ad_a", "camp_1", 100.0)], tracker)

    assert [a.spent_today_usd for a in kept] == [25.0]


async def test_spend_stamp_reaches_the_budget_pacing_feature():
    """End to end: Redis -> filter_by_budget -> extract_features."""
    from datetime import datetime, timezone

    from adplatform.ml.features import RequestContext, extract_features, features_to_dict

    tracker = BudgetTracker(FakeRedis({budget_key("camp_1"): "25.0"}))
    kept = await filter_by_budget([ad("ad_a", "camp_1", 100.0)], tracker)

    ctx = RequestContext.build(
        publisher_id="pub_1", placement_id="place_1", device_type="mobile",
        page_keywords=["python"],
        request_ts=datetime(2026, 8, 25, 14, 30, tzinfo=timezone.utc),
    )
    pacing = features_to_dict(extract_features(kept[0], ctx))["budget_pacing"]

    assert pacing == pytest.approx(0.25), (
        "budget_pacing is dead — the model trains and serves on a constant"
    )


async def test_each_campaign_gets_its_own_spend():
    tracker = BudgetTracker(FakeRedis({
        budget_key("camp_1"): "10.0",
        budget_key("camp_2"): "70.0",
    }))
    ads = [ad("ad_a", "camp_1", 100.0), ad("ad_b", "camp_2", 100.0)]

    kept = await filter_by_budget(ads, tracker)

    assert {a.ad_id: a.spent_today_usd for a in kept} == {"ad_a": 10.0, "ad_b": 70.0}


async def test_the_shared_inventory_snapshot_is_not_mutated():
    """
    filter_by_budget receives the Ad objects that live in the process-wide
    inventory snapshot, shared by every concurrent request. Stamping spend onto
    them in place would have all of them writing to the same objects and would
    leave stale spend behind between refreshes.
    """
    tracker = BudgetTracker(FakeRedis({budget_key("camp_1"): "25.0"}))
    original = ad("ad_a", "camp_1", 100.0)

    kept = await filter_by_budget([original], tracker)

    assert kept[0] is not original
    assert original.spent_today_usd == 0.0, "snapshot ad was mutated in place"


async def test_a_redis_outage_stamps_zero_rather_than_stale_spend():
    """Fails open on the filter, and on the feature too: unknown spend is 0.0."""
    tracker = BudgetTracker(FakeRedis(fail=True))

    kept = await filter_by_budget([ad("ad_a", "camp_1", 100.0)], tracker)

    assert [a.spent_today_usd for a in kept] == [0.0]
