# inventory.py — the live ad inventory, cached in process.

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

# Loading

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
            log.exception("skipping unloadable ad %s", row.get("ad_id"))

    _ads = fresh
    _loaded_at = time.time()
    _load_failures = 0
    log.info("inventory loaded: %d servable ads", len(fresh))
    return len(fresh)


async def refresh_inventory_loop(pool, interval: Optional[int] = None) -> None:
    """
    Background refresh, started in the lifespan and cancelled on shutdown.
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
    """
    return await load_inventory(pool)


# Read path

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