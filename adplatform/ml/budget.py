# budget.py — daily spend enforcement.
#   win_price   is a CPM — dollars per THOUSAND impressions
#   cost_usd    is what this ONE impression actually costs = win_price / 1000

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone

from ..settings import settings

log = logging.getLogger(__name__)


KEY_TTL_SECONDS = settings.budget_key_ttl_seconds

_INCR_SCRIPT = """
local v = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
if redis.call('TTL', KEYS[1]) < 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return v
"""


def budget_key(campaign_id: str, day: str | None = None) -> str:
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"budget:camp:{campaign_id}:{day}"


def impression_cost_usd(win_price_cpm: float) -> float:
    return win_price_cpm / 1000.0


class BudgetTracker:

    def __init__(self, redis_client):
        self.redis = redis_client
        self._script = None

    async def _ensure_script(self):
        if self._script is None:
            self._script = self.redis.register_script(_INCR_SCRIPT)
        return self._script

    async def get_spend(self, campaign_ids: list[str]) -> dict[str, float]:
        """
        Batch-fetch today's spend for many campaigns in ONE round trip.

        Deliberately not one GET per campaign: a 40-ad eligible set would mean
        up to 40 sequential round trips, which at 0.2ms each is 8ms of your 20ms
        budget spent waiting on a cache. Callers should de-duplicate to campaign
        ids first -- several creatives usually share one campaign, so the MGET is
        typically smaller than the candidate set.

        FAIL-OPEN on Redis errors. The alternative — refusing to bid — turns a
        cache outage into a total revenue outage, and any overspend during the
        window is bounded, visible in the impression log, and corrected by
        reconcile(). Failing closed protects a number that is already recoverable
        at the cost of the only thing that isn't: the auction you cannot re-run.
        """
        if not campaign_ids:
            return {}
        try:
            keys = [budget_key(c) for c in campaign_ids]
            values = await self.redis.mget(keys)
            return {
                cid: float(v) if v is not None else 0.0
                for cid, v in zip(campaign_ids, values)
            }
        except Exception:
            log.exception("budget read failed; failing open (spend treated as 0)")
            return {cid: 0.0 for cid in campaign_ids}

    # -- write path (after the auction, off the critical path) ---------------

    async def record_spend(self, campaign_id: str, cost_usd: float) -> float | None:
        """
        Increment after a win. Call via spawn(), not awaited inline — the bid
        response should not wait on it.

        A crash between the auction and this call loses the increment. That is
        acceptable precisely because reconcile() recomputes from the durable
        impression log; without reconciliation this would be a silent
        under-count that lets advertisers overspend.
        """
        if not campaign_id:
            log.error("record_spend called with no campaign_id (cost=%.6f) -- "
                      "spend NOT recorded", cost_usd)
            return None
        try:
            script = await self._ensure_script()
            new_total = await script(
                keys=[budget_key(campaign_id)], args=[cost_usd, KEY_TTL_SECONDS]
            )
            return float(new_total)
        except Exception:
            log.exception("budget increment failed for campaign=%s cost=%.6f",
                          campaign_id, cost_usd)
            return None

    async def reconcile(self, ch_client, day: str | None = None) -> dict[str, float]:
        """
        Recompute today's spend from ad_impressions and overwrite Redis.
        """
        import asyncio

        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        query = """
            SELECT campaign_id, sum(win_price) / 1000.0 AS spent_usd
            FROM ad_impressions
            WHERE toDate(ts) = {day:Date}
              AND campaign_id != ''
            GROUP BY campaign_id
        """
        try:
            rows = (await asyncio.to_thread(
                ch_client.query, query, {"day": day}
            )).result_rows
        except Exception:
            log.exception("budget reconciliation query failed")
            return {}

        corrected: dict[str, float] = {}
        for campaign_id, spent in rows:
            spent = float(spent)
            key = budget_key(campaign_id, day)
            try:
                previous = await self.redis.get(key)
                previous = float(previous) if previous else 0.0
                if abs(previous - spent) > 0.001:
                    log.info("budget drift campaign=%s redis=%.4f actual=%.4f",
                             campaign_id, previous, spent)
                await self.redis.set(key, spent, ex=KEY_TTL_SECONDS)
                corrected[campaign_id] = spent
            except Exception:
                log.exception("failed to reconcile budget for campaign=%s", campaign_id)

        return corrected


async def budget_reconcile_loop(tracker: BudgetTracker, ch_client,
                                interval_seconds: int | None = None) -> None:
    import asyncio

    interval_seconds = interval_seconds or settings.budget_reconcile_seconds

    while True:
        try:
            await tracker.reconcile(ch_client)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("budget reconcile loop iteration failed")
        await asyncio.sleep(interval_seconds)

# Stage 1 filter

async def filter_by_budget(ads: list, tracker: BudgetTracker) -> list:
    """
    Drop ads that have exhausted today's budget, and stamp today's spend onto
    the survivors.

    The stamp is not incidental. `spent_today_usd` is what features.py divides
    by `daily_budget_usd` to build the `budget_pacing` feature, and this is the
    only place on the serving path that ever learns what a campaign has spent.
    Leaving it at the 0.0 that inventory._row_to_ad sets means the model trains
    and serves on a feature that is constant zero in production while its unit
    tests -- which construct an ad with the field already populated -- pass.

    `replace()` rather than assignment: these Ad objects belong to the shared
    inventory snapshot, so writing to them would have every concurrent request
    mutating the same objects and would leave stale spend on them between
    refreshes.
    """
    if not ads:
        return []

    campaign_ids = list({ad.campaign_id for ad in ads if ad.campaign_id})

    spend = await tracker.get_spend(campaign_ids) if campaign_ids else {}

    kept = []
    for ad in ads:
        if not ad.campaign_id:
            log.warning("ad %s has no campaign_id; excluding from auction", ad.ad_id)
            continue
        spent = spend.get(ad.campaign_id, 0.0)
        if spent < ad.daily_budget_usd:
            kept.append(replace(ad, spent_today_usd=spent))

    return kept
