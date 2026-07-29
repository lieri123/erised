# budget.py — daily spend enforcement.
#
# Two numbers that must not be confused:
#
#   win_price   is a CPM — dollars per THOUSAND impressions
#   cost_usd    is what this ONE impression actually costs = win_price / 1000
#
# Budgets decrement by cost_usd. Decrementing by win_price overcharges by 1000x
# and exhausts a $500 daily budget in about 125 impressions. Everything in this
# module deals in cost_usd; the conversion happens once, in impression_cost_usd.

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..settings import settings

log = logging.getLogger(__name__)

# 48h so a key survives past its own day for reconciliation and debugging,
# then cleans itself up. Without a TTL you accumulate one key per ad per day
# forever.
KEY_TTL_SECONDS = settings.budget_key_ttl_seconds

# INCRBYFLOAT and EXPIRE as one atomic operation. Two round trips would leave a
# window where a crash between them creates an immortal key.
_INCR_SCRIPT = """
local v = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
if redis.call('TTL', KEYS[1]) < 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return v
"""


def budget_key(ad_id: str, day: str | None = None) -> str:
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"budget:{ad_id}:{day}"


def impression_cost_usd(win_price_cpm: float) -> float:
    """One impression costs 1/1000th of the CPM. This is the only place the
    conversion happens."""
    return win_price_cpm / 1000.0


class BudgetTracker:
    """
    Redis is a fast cache of spend, not the source of truth. The impression log
    is authoritative — every win writes a durable row carrying win_price, so
    actual spend is always recomputable. That is what makes the failure policy
    below safe.
    """

    def __init__(self, redis_client):
        self.redis = redis_client
        self._script = None

    async def _ensure_script(self):
        if self._script is None:
            self._script = self.redis.register_script(_INCR_SCRIPT)
        return self._script

    # -- read path (hot) ----------------------------------------------------

    async def get_spend(self, ad_ids: list[str]) -> dict[str, float]:
        """
        Batch-fetch today's spend for many ads in ONE round trip.

        Deliberately not one GET per ad: a 40-ad eligible set would mean 40
        sequential round trips, which at 0.2ms each is 8ms of your 20ms budget
        spent waiting on a cache.

        FAIL-OPEN on Redis errors. The alternative — refusing to bid — turns a
        cache outage into a total revenue outage, and any overspend during the
        window is bounded, visible in the impression log, and corrected by
        reconcile(). Failing closed protects a number that is already recoverable
        at the cost of the only thing that isn't: the auction you cannot re-run.
        """
        if not ad_ids:
            return {}
        try:
            keys = [budget_key(a) for a in ad_ids]
            values = await self.redis.mget(keys)
            return {
                ad_id: float(v) if v is not None else 0.0
                for ad_id, v in zip(ad_ids, values)
            }
        except Exception:
            log.exception("budget read failed; failing open (spend treated as 0)")
            return {ad_id: 0.0 for ad_id in ad_ids}

    # -- write path (after the auction, off the critical path) ---------------

    async def record_spend(self, ad_id: str, cost_usd: float) -> float | None:
        """
        Increment after a win. Call via spawn(), not awaited inline — the bid
        response should not wait on it.

        A crash between the auction and this call loses the increment. That is
        acceptable precisely because reconcile() recomputes from the durable
        impression log; without reconciliation this would be a silent
        under-count that lets advertisers overspend.
        """
        try:
            script = await self._ensure_script()
            new_total = await script(
                keys=[budget_key(ad_id)], args=[cost_usd, KEY_TTL_SECONDS]
            )
            return float(new_total)
        except Exception:
            log.exception("budget increment failed for ad=%s cost=%.6f", ad_id, cost_usd)
            return None

    # -- reconciliation (periodic) ------------------------------------------

    async def reconcile(self, ch_client, day: str | None = None) -> dict[str, float]:
        """
        Recompute today's spend from ad_impressions and overwrite Redis.

        Run every few minutes. This is what makes the fail-open read policy and
        the fire-and-forget write path safe: both can lose counts, and both get
        corrected here from the log that actually persisted.

        Note SUM(win_price) / 1000 — the log stores CPM, budgets are in dollars.
        """
        import asyncio

        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        query = """
            SELECT ad_id, sum(win_price) / 1000.0 AS spent_usd
            FROM ad_impressions
            WHERE toDate(ts) = {day:Date}
            GROUP BY ad_id
        """
        try:
            rows = (await asyncio.to_thread(
                ch_client.query, query, {"day": day}
            )).result_rows
        except Exception:
            log.exception("budget reconciliation query failed")
            return {}

        corrected: dict[str, float] = {}
        for ad_id, spent in rows:
            spent = float(spent)
            key = budget_key(ad_id, day)
            try:
                previous = await self.redis.get(key)
                previous = float(previous) if previous else 0.0
                if abs(previous - spent) > 0.001:
                    log.info("budget drift ad=%s redis=%.4f actual=%.4f",
                             ad_id, previous, spent)
                await self.redis.set(key, spent, ex=KEY_TTL_SECONDS)
                corrected[ad_id] = spent
            except Exception:
                log.exception("failed to reconcile budget for ad=%s", ad_id)

        return corrected


async def budget_reconcile_loop(tracker: BudgetTracker, ch_client,
                                interval_seconds: int | None = None) -> None:
    """Background task; add to the lifespan alongside the other refresh loops."""
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


# ---------------------------------------------------------------------------
# Stage 1 filter
# ---------------------------------------------------------------------------

async def filter_by_budget(ads: list, tracker: BudgetTracker) -> list:
    """
    Drop ads that have exhausted today's budget.

    Two concurrent auctions can both read spend below the cap and both win,
    pushing an advertiser slightly over. Every ad platform has this and none of
    them lock the hot path to prevent it — a distributed lock per bid would cost
    more latency than the overshoot costs money. The overshoot is bounded by
    (concurrent requests x cost per impression), which at CPM pricing is
    fractions of a cent.
    """
    if not ads:
        return []
    spend = await tracker.get_spend([ad.ad_id for ad in ads])
    return [ad for ad in ads if spend.get(ad.ad_id, 0.0) < ad.daily_budget_usd]