# rtb_integration.py

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..settings import settings
from .ctr_model import ctr_model
from .features import CtrStats, RequestContext

from ..rtb import get_eligible_ads

EXPLORATION_EPSILON = settings.exploration_epsilon


@dataclass
class ScoredAd:
    ad: object
    predicted_ctr: float
    bid_value: float
    features: list[float]


@dataclass
class AuctionResult:
    winner: ScoredAd
    win_price: float          # CPM — dollars per THOUSAND impressions
    cost_usd: float           # what THIS impression costs = win_price / 1000
    impression_id: str
    is_exploration: bool
    serve_propensity: float
    model_version: str


async def score_ads(
    ads: list,
    ctx: RequestContext,
    stats: CtrStats,
) -> list[ScoredAd]:
    """
    Predict CTR for every eligible ad and convert to a bid value.
    """
    ctrs, vectors = ctr_model.predict_batch(ads, ctx, stats)
    return [
        ScoredAd(
            ad=ad,
            predicted_ctr=ctr,
            bid_value=round(ctr * ad.target_cpm, 6),
            features=vec,
        )
        for ad, ctr, vec in zip(ads, ctrs, vectors)
    ]


PRICE_TICK_CPM = 0.01


def clearing_cpm(winner: ScoredAd, runner_up_bid_value: float) -> float:
    """
    Convert a second-price result from bid_value units back into a CPM.
    """
    ctr = max(winner.predicted_ctr, 1e-9)
    price = (runner_up_bid_value / ctr) + PRICE_TICK_CPM
    price = max(price, winner.ad.floor_price)
    price = round(price, 6)
    return min(price, winner.ad.target_cpm)


def run_auction(
    scored_ads: list[ScoredAd],
    impression_id: str,
    epsilon: float = EXPLORATION_EPSILON,
    rng: random.Random | None = None,
) -> Optional[AuctionResult]:
    if not scored_ads:
        return None

    rng = rng or random
    n = len(scored_ads)

    def _result(winner: ScoredAd, cpm: float, explored: bool,
                propensity: float) -> AuctionResult:
        cpm = max(cpm, 0.0)
        return AuctionResult(
            winner=winner,
            win_price=cpm,
            cost_usd=cpm / 1000.0,
            impression_id=impression_id,
            is_exploration=explored,
            serve_propensity=propensity,
            model_version=ctr_model.model_version,
        )

    if n > 1 and rng.random() < epsilon:
        winner = rng.choice(scored_ads)
        return _result(winner, min(winner.ad.floor_price, winner.ad.target_cpm),
                       True, epsilon / n)

    ranked = sorted(scored_ads, key=lambda s: s.bid_value, reverse=True)
    winner = ranked[0]

    if len(ranked) >= 2:
        cpm = clearing_cpm(winner, ranked[1].bid_value)
    else:
        cpm = min(winner.ad.floor_price, winner.ad.target_cpm)

    propensity = (1.0 - epsilon) + (epsilon / n) if n > 1 else 1.0
    return _result(winner, cpm, False, propensity)


async def run_rtb(
    publisher_id: str,
    placement_id: str,
    device_type: str,
    page_keywords: list[str],
    impression_id: str,
    request_ts: datetime | None = None,
    stats: CtrStats | None = None,
    budget_tracker=None,
) -> tuple[Optional[AuctionResult], RequestContext]:
    """
    Full pipeline: filter -> score -> auction.
    """
    from .refresh import current_stats

    ctx = RequestContext.build(
        publisher_id=publisher_id,
        placement_id=placement_id,
        device_type=device_type,
        page_keywords=page_keywords,
        request_ts=request_ts,
    )

    eligible = await get_eligible_ads(
        device_type=ctx.device_type, page_keywords=list(ctx.page_keywords)
    )

    if budget_tracker is not None and eligible:
        from .budget import filter_by_budget
        eligible = await filter_by_budget(eligible, budget_tracker)

    if not eligible:
        return None, ctx

    scored = await score_ads(eligible, ctx, stats or current_stats())
    return run_auction(scored, impression_id=impression_id), ctx


def build_impression_event(result: AuctionResult, ctx: RequestContext) -> dict:
    """
    The row that goes to Kafka -> ClickHouse.ad_impressions.

    `features` is the vector that was actually scored. Log it verbatim. The
    temptation is to log the raw request instead and recompute features in the
    training job — that is exactly how train/serve skew gets in, and it is
    almost impossible to detect afterwards because both sides look correct in
    isolation.
    """
    from .features import FEATURE_VERSION

    winner = result.winner
    return {
        "impression_id": result.impression_id,
        "ts": ctx.request_ts.isoformat(),
        "publisher_id": ctx.publisher_id,
        "placement_id": ctx.placement_id,
        "ad_id": winner.ad.ad_id,
        "campaign_id": winner.ad.campaign_id or "",
        "advertiser_id": winner.ad.advertiser_id,
        "device_type": ctx.device_type,
        "feature_version": FEATURE_VERSION,
        "features": winner.features,
        "predicted_ctr": winner.predicted_ctr,
        "bid_value": winner.bid_value,
        "win_price": result.win_price,
        "cost_usd": result.cost_usd,
        "is_exploration": int(result.is_exploration),
        "serve_propensity": result.serve_propensity,
        "model_version": result.model_version,
    }