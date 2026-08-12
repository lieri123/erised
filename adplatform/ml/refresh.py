# refresh.py — background tasks that keep the serving process current.

from __future__ import annotations

import asyncio
import logging

from ..settings import settings
from .ctr_model import ctr_model
from .features import CtrStats

log = logging.getLogger(__name__)

STATS_REFRESH_SECONDS = settings.stats_refresh_seconds
MODEL_REFRESH_SECONDS = settings.model_refresh_seconds
LOOKBACK_DAYS = settings.stats_lookback_days

AGG_QUERY = """
SELECT ad_id, placement_id,
       sumMerge(impressions) AS imps,
       sumMerge(clicks)      AS clicks
FROM ctr_agg_pair
WHERE day >= today() - {lookback:UInt16}
GROUP BY ad_id, placement_id
"""

_stats = CtrStats()

def current_stats() -> CtrStats:
    return _stats


async def _load_stats(ch_client) -> CtrStats:
    rows = (await asyncio.to_thread(
        ch_client.query, AGG_QUERY, {"lookback": LOOKBACK_DAYS}
    )).result_rows

    pair: dict[tuple[str, str], tuple[int, int]] = {}
    ad: dict[str, tuple[int, int]] = {}
    placement: dict[str, tuple[int, int]] = {}
    total_imps = total_clicks = 0

    for ad_id, placement_id, imps, clicks in rows:
        imps, clicks = int(imps), int(clicks)
        pair[(ad_id, placement_id)] = (imps, clicks)
        ai, ac = ad.get(ad_id, (0, 0))
        ad[ad_id] = (ai + imps, ac + clicks)
        pi, pc = placement.get(placement_id, (0, 0))
        placement[placement_id] = (pi + imps, pc + clicks)
        total_imps += imps
        total_clicks += clicks

    global_ctr = (total_clicks / total_imps) if total_imps > 1000 else 0.010

    return CtrStats(
        global_ctr=global_ctr,
        ad_counts=ad,
        placement_counts=placement,
        pair_counts=pair,
    )


async def stats_refresh_loop(ch_client) -> None:
    global _stats
    while True:
        try:
            _stats = await _load_stats(ch_client)
            log.info("CTR stats refreshed: %d pairs, global_ctr=%.4f%%",
                     len(_stats.pair_counts), 100 * _stats.global_ctr)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("CTR stats refresh failed, keeping previous snapshot")
        await asyncio.sleep(STATS_REFRESH_SECONDS)


async def model_refresh_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(ctr_model.load)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("model reload check failed")
        await asyncio.sleep(MODEL_REFRESH_SECONDS)