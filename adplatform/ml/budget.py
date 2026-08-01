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


def budget_key(campaign_id: str, day: str | None = None) -> str:
    """
    Redis key for one campaign's spend on one UTC day.

    KEYED ON CAMPAIGN, NOT AD. daily_budget_usd is a property of the campaign,
    so a campaign with three creatives must share one counter. Keying on ad_id
    gave each creative its own counter measured against the full campaign
    budget -- 3x overspend, every counter reporting itself healthy.

    The namespace is deliberately "budget:camp:" rather than the old "budget:".
    A campaign_id could otherwise collide with an ad_id from the pre-migration
    era and silently inherit its spend. Distinct prefixes make the cutover
    total: old keys are unreachable and expire on their own TTL.
    """
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"budget:camp:{campaign_id}:{day}"


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
            # An ad with no campaign cannot be charged to a budget. Loud, because
            # silently skipping the increment is how an advertiser serves for free.
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

    # -- reconciliation (periodic) ------------------------------------------

    async def reconcile(self, ch_client, day: str | None = None) -> dict[str, float]:
        """
        Recompute today's spend from ad_impressions and overwrite Redis.

        Run every few minutes. This is what makes the fail-open read policy and
        the fire-and-forget write path safe: both can lose counts, and both get
        corrected here from the log that actually persisted.

        Note SUM(win_price) / 1000 — the log stores CPM, budgets are in dollars.

        GROUPS BY campaign_id to match the enforcement key. Grouping by ad_id
        while enforcement reads campaign keys would be worse than not
        reconciling at all: it would write ad-keyed values nobody reads, and the
        campaign keys would never be rebuilt after a Redis restart -- so every
        campaign would silently reset to zero spend and serve a second full
        budget.

        Rows with an empty campaign_id are excluded. Those are impressions
        served before migration 003 added the column; they have no campaign to
        charge, and folding them into any single campaign would overstate it.
        They remain in the table for training, which does not care.
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

    # De-duplicate to campaigns before hitting Redis. Several creatives normally
    # share one campaign, so this makes the MGET smaller than the candidate set,
    # not larger.
    campaign_ids = list({ad.campaign_id for ad in ads if ad.campaign_id})

    spend = await tracker.get_spend(campaign_ids) if campaign_ids else {}

    kept = []
    for ad in ads:
        if not ad.campaign_id:
            # No campaign means no budget to check and no way to charge the
            # spend. Dropping it is the conservative choice: serving free
            # impressions is worse than not serving. In practice this cannot
            # happen -- servable_ads joins through campaigns -- so it means the
            # inventory row is malformed.
            log.warning("ad %s has no campaign_id; excluding from auction", ad.ad_id)
            continue
        if spend.get(ad.campaign_id, 0.0) < ad.daily_budget_usd:
            kept.append(ad)

    return kept