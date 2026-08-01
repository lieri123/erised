# tests/test_eligible_ads.py
#
# These exist because two bugs sat in the repo simultaneously and hid each
# other:
#
#   1. get_eligible_ads read rtb.MOCK_ADS, so nothing an advertiser created was
#      ever servable.
#   2. inventory._row_to_ad passed campaign_id and created_at to an Ad dataclass
#      that had neither, so load_inventory returned 0 ads on every refresh —
#      silently, because the per-row `except` logs and continues.
#
# Fixing only (1) would have taken the platform from serving five fake ads to
# serving nothing. test_load_inventory_loads_every_valid_row is the one that
# catches (2); it fails loudly on a count of zero rather than trusting the log.

from datetime import datetime, timedelta, timezone

import pytest

from adplatform import inventory
from adplatform.rtb import MOCK_ADS, Ad, get_eligible_ads


# --- fixtures --------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_inventory():
    """Every test starts from a cold, empty snapshot and leaves one behind."""
    inventory._reset_for_tests()
    yield
    inventory._reset_for_tests()


def seed(*ads):
    """Install a snapshot directly, bypassing Postgres."""
    inventory._ads = list(ads)
    inventory._loaded_at = 1.0  # any non-None value marks the cache as warm
    inventory._load_failures = 0


def make_ad(ad_id="ad_test", device="all", cpm=5.0, **kw):
    return Ad(
        ad_id=ad_id,
        advertiser_id="adv_test",
        creative_html="<div>test</div>",
        destination_url="https://example.com",
        target_cpm=cpm,
        floor_price=kw.pop("floor_price", 1.0),
        target_device=device,
        target_keywords=kw.pop("target_keywords", ["python"]),
        campaign_id=kw.pop("campaign_id", "camp_test"),
        created_at=kw.pop("created_at", datetime.now(timezone.utc)),
        **kw,
    )


def row(**over):
    """A dict shaped like one row of the servable_ads view."""
    base = {
        "ad_id": "ad_db",
        "advertiser_id": "adv_db",
        "creative_html": "<div>from db</div>",
        "destination_url": "https://db.example.com",
        "target_cpm": 7.5,
        "floor_price": 2.0,
        "target_device": "all",
        # asyncpg hands JSONB back as a str unless a codec is registered, and
        # db.py registers none. _row_to_ad has to cope with both.
        "target_keywords": '["python", "ml"]',
        "daily_budget_usd": 500.0,
        "campaign_id": "camp_db",
        "created_at": datetime.now(timezone.utc) - timedelta(days=3),
    }
    base.update(over)
    return base


class FakePool:
    def __init__(self, rows):
        self._rows = rows

    async def fetch(self, _query):
        return self._rows


# --- the wiring ------------------------------------------------------------

async def test_serves_inventory_not_mock_ads():
    """The regression test for bug (1)."""
    seed(make_ad(ad_id="ad_from_db"))

    got = await get_eligible_ads(device_type="desktop", page_keywords=[])

    assert [a.ad_id for a in got] == ["ad_from_db"]

    mock_ids = {a.ad_id for a in MOCK_ADS}
    assert not mock_ids & {a.ad_id for a in got}, "MOCK_ADS leaked into serving"


async def test_new_ad_becomes_eligible_without_restart():
    """An advertiser's ad must serve as soon as the snapshot is replaced."""
    seed(make_ad(ad_id="ad_old"))
    assert len(await get_eligible_ads("desktop", [])) == 1

    seed(make_ad(ad_id="ad_old"), make_ad(ad_id="ad_new"))
    assert {a.ad_id for a in await get_eligible_ads("desktop", [])} == {
        "ad_old",
        "ad_new",
    }


async def test_empty_inventory_yields_no_fill():
    """A warm but empty snapshot returns nothing — it does not fall back."""
    seed()
    assert await get_eligible_ads("desktop", []) == []


# --- the hard filters ------------------------------------------------------

async def test_device_targeting():
    seed(
        make_ad(ad_id="ad_all", device="all"),
        make_ad(ad_id="ad_mobile", device="mobile"),
        make_ad(ad_id="ad_desktop", device="desktop"),
    )

    got = {a.ad_id for a in await get_eligible_ads("mobile", [])}
    assert got == {"ad_all", "ad_mobile"}


async def test_publisher_floor_excludes_cheap_ads():
    seed(make_ad(ad_id="ad_cheap", cpm=1.0), make_ad(ad_id="ad_rich", cpm=9.0))

    got = {a.ad_id for a in await get_eligible_ads("desktop", [], floor_price=5.0)}
    assert got == {"ad_rich"}


async def test_floor_is_inclusive():
    """target_cpm == floor is a win for the advertiser, not a rejection."""
    seed(make_ad(ad_id="ad_exact", cpm=5.0))
    assert len(await get_eligible_ads("desktop", [], floor_price=5.0)) == 1


async def test_keywords_are_not_a_hard_filter():
    """Keyword overlap is a stage-2 scoring signal. Zero overlap still bids."""
    seed(make_ad(target_keywords=["gardening"]))
    assert len(await get_eligible_ads("desktop", ["python", "docker"])) == 1


async def test_snapshot_is_not_mutated():
    """get_eligible_ads builds a new list; current_inventory returns the real one."""
    seed(make_ad(ad_id="a"), make_ad(ad_id="b", device="mobile"))
    before = list(inventory.current_inventory())

    await get_eligible_ads("desktop", [])

    assert inventory.current_inventory() == before


# --- bug (2): the row → Ad contract ----------------------------------------

def test_row_to_ad_accepts_every_servable_ads_column():
    """
    _row_to_ad passes campaign_id and created_at. If the Ad dataclass ever
    stops accepting them this raises TypeError here, in a test, instead of
    being swallowed by the per-row except in load_inventory.
    """
    ad = inventory._row_to_ad(row())

    assert ad.ad_id == "ad_db"
    assert ad.campaign_id == "camp_db", "campaign_id is the budget key"
    assert isinstance(ad.created_at, datetime), "created_at feeds ad_age_days"
    assert ad.target_keywords == ["python", "ml"], "JSONB string was not decoded"
    assert ad.spent_today_usd == 0.0


async def test_load_inventory_loads_every_valid_row():
    """
    The test that would have caught bug (2). load_inventory never raises, so a
    total failure looks identical to an empty ads table unless you assert on
    the count.
    """
    n = await inventory.load_inventory(FakePool([row(ad_id="a"), row(ad_id="b")]))

    assert n == 2, "rows were dropped by the per-row exception handler"
    assert {a.ad_id for a in inventory.current_inventory()} == {"a", "b"}


async def test_one_bad_row_does_not_cost_the_whole_snapshot():
    bad = row(ad_id="broken")
    del bad["target_cpm"]

    n = await inventory.load_inventory(FakePool([row(ad_id="good"), bad]))

    assert n == 1
    assert [a.ad_id for a in inventory.current_inventory()] == ["good"]


async def test_failed_refresh_keeps_the_previous_snapshot():
    """Stale inventory is a rounding error. An empty one is an outage."""

    class BrokenPool:
        async def fetch(self, _query):
            raise ConnectionError("postgres went away")

    await inventory.load_inventory(FakePool([row(ad_id="keep_me")]))
    n = await inventory.load_inventory(BrokenPool())

    assert n == 0
    assert [a.ad_id for a in inventory.current_inventory()] == ["keep_me"]


async def test_ad_from_db_is_servable_end_to_end():
    """load_inventory → get_eligible_ads, the path that was broken in two places."""
    await inventory.load_inventory(
        FakePool([row(ad_id="ad_real", target_device="mobile", target_cpm=8.0)])
    )

    got = await get_eligible_ads("mobile", ["python"], floor_price=3.0)

    assert [a.ad_id for a in got] == ["ad_real"]
    assert got[0].campaign_id == "camp_db"
