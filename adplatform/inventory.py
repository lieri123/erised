# inventory.py — the live ad inventory, cached in process.
#
# Replaces rtb.MOCK_ADS as the source of servable ads.
#
# ---------------------------------------------------------------------------
# WHY A CACHE AND NOT A QUERY PER BID
# ---------------------------------------------------------------------------
# The obvious implementation is the one written in get_eligible_ads' docstring:
#
#     SELECT * FROM ads WHERE status='active' AND target_device IN (...)
#
# One Postgres round trip is 1-2ms of a ~20ms budget, and it puts the database
# on the critical path of every auction — a slow query or a failover and you
# stop bidding entirely. Worse, it is the same query with the same handful of
# results tens of thousands of times a second.
#
# Total inventory is small. Even 100k live ads with creatives is a few hundred
# MB, and realistically you have hundreds. So: load everything once, refresh in
# the background, filter in memory. Same pattern as cors.py and auth.py, which
# is deliberate — three caches behaving three different ways is how you end up
# unsure which one is stale.
#
# ---------------------------------------------------------------------------
# FAILURE POLICY: keep serving
# ---------------------------------------------------------------------------
# A failed refresh keeps the previous snapshot. Stale inventory means an ad that
# was paused two minutes ago might serve once more; an empty snapshot means the
# platform returns no_fill for everything and every publisher's page goes blank.
# The first is a rounding error in someone's budget, the second is an outage.
#
# On a COLD cache (never loaded) we fall back to rtb.MOCK_ADS in development so
# the thing boots and demos without Postgres, and serve nothing in production.
# Silently serving five hardcoded example creatives to real publisher traffic is
# not a failure mode anyone wants.

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from .settings import settings

log = logging.getLogger("inventory")

# Live snapshot. Replaced wholesale, never mutated in place, so an auction that
# reads it mid-refresh sees a consistent list rather than a half-swapped one.
_ads: list = []
_loaded_at: Optional[float] = None
_load_failures = 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _row_to_ad(row) -> object:
    """
    Build one rtb.Ad from a servable_ads row.

    Imported lazily so this module can be imported by tooling that has no
    interest in the serving stack.

    NOTE ON budget_id: it is the CAMPAIGN id, not the ad id. Budgets are set per
    campaign, so spend must be counted per campaign — see budget.py. Keying
    Redis on ad_id lets a five-creative campaign spend five times its budget.
    """
    from .rtb import Ad

    keywords = row["target_keywords"]
    # asyncpg returns JSONB as str unless a codec is registered. db.py does not
    # register one, so decode here rather than assuming.
    if isinstance(keywords, str):
        keywords = json.loads(keywords)

    return Ad(
        ad_id=row["ad_id"],
        advertiser_id=row["advertiser_id"],
        creative_html=row["creative_html"],
        destination_url=row["destination_url"],
        target_cpm=float(row["target_cpm"]),
        floor_price=float(row["floor_price"]),
        target_device=row["target_device"],
        target_keywords=list(keywords or []),
        daily_budget_usd=float(row["daily_budget_usd"]),
        spent_today_usd=0.0,        # filled in by budget.filter_by_budget
        campaign_id=row["campaign_id"],
        created_at=row["created_at"],
    )


async def load_inventory(pool) -> int:
    """
    Refresh the snapshot from Postgres. Returns the number of ads loaded.
    Never raises — a refresh failure keeps the previous snapshot.
    """
    global _ads, _loaded_at, _load_failures

    try:
        rows = await pool.fetch("SELECT * FROM servable_ads")
    except Exception:
        _load_failures += 1
        log.exception("inventory refresh failed (%d consecutive); keeping %d ads",
                      _load_failures, len(_ads))
        return 0

    fresh = []
    for row in rows:
        try:
            fresh.append(_row_to_ad(row))
        except Exception:
            # One malformed row must not cost you the whole inventory. Same
            # reasoning as kafka_skip_broken_messages in kafka_sink.sql.
            log.exception("skipping unloadable ad %s", row.get("ad_id"))

    _ads = fresh
    _loaded_at = time.time()
    _load_failures = 0
    log.info("inventory loaded: %d servable ads", len(fresh))
    return len(fresh)


async def refresh_inventory_loop(pool, interval: Optional[int] = None) -> None:
    """
    Background refresh, started in the lifespan and cancelled on shutdown.

    PAUSE LATENCY: an ad paused in the API keeps serving until the next refresh.
    That is why invalidate() exists and why the default interval is 60s rather
    than the 300s used for CORS — an advertiser watching a runaway creative
    wants it stopped now, and "wait five minutes" is not an acceptable answer.
    """
    interval = interval or settings.inventory_refresh_seconds
    log.info("inventory refresh loop started — interval: %ds", interval)

    while True:
        await asyncio.sleep(interval)
        try:
            await load_inventory(pool)
        except asyncio.CancelledError:
            log.info("inventory refresh loop cancelled")
            raise
        except Exception:
            log.exception("inventory refresh loop error")


async def invalidate(pool) -> int:
    """
    Force an immediate reload. Called after any write that changes what is
    servable, so an advertiser sees their edit take effect on the next request
    instead of within the refresh window.

    Reloads everything rather than patching the one changed ad: the write may
    have changed campaign-level fields shared by several ads, and a full reload
    of a few hundred rows is cheaper than reasoning about which ones moved.
    """
    return await load_inventory(pool)


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

def current_inventory() -> list:
    """
    The live snapshot. Returns the list itself, not a copy — callers must not
    mutate it. get_eligible_ads builds a new filtered list, so it does not.
    """
    if _loaded_at is not None:
        return _ads

    # Cold cache.
    if settings.is_production:
        log.error("inventory never loaded — serving no ads")
        return []

    from .rtb import MOCK_ADS
    log.warning("inventory never loaded — falling back to %d MOCK_ADS (dev only)",
                len(MOCK_ADS))
    return MOCK_ADS


def status() -> dict:
    """For /health."""
    return {
        "loaded": _loaded_at is not None,
        "ads": len(_ads),
        "age_seconds": round(time.time() - _loaded_at, 1) if _loaded_at else None,
        "consecutive_failures": _load_failures,
    }


def _reset_for_tests() -> None:
    global _ads, _loaded_at, _load_failures
    _ads, _loaded_at, _load_failures = [], None, 0