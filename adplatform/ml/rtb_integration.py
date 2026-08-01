# rtb_integration.py — replaces stages 2 and 3 of your existing rtb.py.
#
# Changes from the current version:
#   - score_ads() calls the real model and returns the logged feature vectors
#   - run_auction() takes an epsilon-greedy exploration branch
#   - the single-bidder floor price bug is fixed (see _floor_as_bid_value)
#   - every served impression carries its feature vector and serve propensity
#
# Keep get_eligible_ads() exactly as it is. Stage 1 is a hard filter and has
# nothing to do with the model.

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..settings import settings
from .ctr_model import ctr_model
from .features import CtrStats, RequestContext

# Stage 1 stays exactly as you wrote it — it is a hard filter with no model
# involvement. If your rtb.py is not importable as a package, change this to
# `from ..rtb import get_eligible_ads`.
from ..rtb import get_eligible_ads

# Fraction of auctions decided randomly instead of by the model.
#
# This is not a tuning knob you can set to 0 — it is what makes your training
# data usable. With pure exploitation you only ever observe clicks on ads your
# current ranking already favours, so a genuinely better ad that the model
# happens to underrate never gets served, never accumulates click data, and
# never gets a chance to correct the model. The bias is self-reinforcing and no
# amount of modelling recovers from it.
#
# 0.05-0.10 while you are collecting your first data, then taper. Consider
# Thompson sampling later, which explores proportionally to uncertainty rather
# than uniformly and so costs less revenue for the same information.
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

    bid_value = predicted_ctr * target_cpm is the line that makes relevance
    beat raw budget: an ad bidding $8 at 0.4% CTR loses to one bidding $4 at
    1.2%, because the second is worth more per impression to everyone involved.

    No await inside — XGBoost inference on a few dozen rows is tens of
    microseconds and releases the GIL internally. If your eligible set ever
    grows into the thousands, move this to a thread executor rather than
    letting it block the event loop.
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

    THE BUG THIS FIXES. Ranking happens on bid_value = predicted_ctr x CPM, but
    billing happens in CPM. The old code charged the runner-up's bid_value
    directly as though it were already a CPM:

        A: $5.00 CPM at 2% CTR -> bid_value 0.10   (wins)
        B: $8.00 CPM at 1% CTR -> bid_value 0.08

        old: win_price = 0.08  ->  A billed $0.08 CPM against a $5.00 bid

    That undercharges by roughly 1/ctr — a factor of 50 to 100 at realistic
    click rates.

    The correct clearing price is the lowest CPM at which the winner would still
    have beaten the runner-up. The winner wins while

        winner_ctr x cpm >= runner_up_bid_value

    so the break-even CPM is runner_up_bid_value / winner_ctr:

        new: 0.08 / 0.02 = $4.00 CPM, plus a tick

    Sanity check: at $4.01 and 2% CTR the winner's bid_value is 0.0802, just
    above B's 0.08. Correct, and below A's $5.00 maximum.

    Invariant worth knowing: the winner has the highest bid_value, so
    runner_up_bid_value <= winner_ctr x winner_cpm always holds and the
    break-even CPM can never exceed the winner's own target_cpm. Only the tick
    can push it over, which is why the cap below binds so rarely.
    """
    ctr = max(winner.predicted_ctr, 1e-9)
    price = (runner_up_bid_value / ctr) + PRICE_TICK_CPM
    price = max(price, winner.ad.floor_price)
    # Round BEFORE the cap, not after. round(x, 6) can round upward, so
    # capping first and rounding second can land fractionally above the
    # advertiser's stated maximum — a property test found a case billing
    # 1.29735 against a 1.2973499 cap. Trivial money, but "you charged above my
    # max bid" is not an argument worth having with an advertiser.
    price = round(price, 6)
    # A misconfigured ad can have floor_price above target_cpm; the advertiser's
    # own maximum always wins that argument.
    return min(price, winner.ad.target_cpm)


def run_auction(
    scored_ads: list[ScoredAd],
    impression_id: str,
    epsilon: float = EXPLORATION_EPSILON,
    rng: random.Random | None = None,
) -> Optional[AuctionResult]:
    """
    Second-price auction with an epsilon-greedy exploration branch.

    Ranking is on bid_value; billing is in CPM. Keeping those two units straight
    is the whole job of this function — see clearing_cpm.

    Exploration serves a uniformly random eligible ad at its floor CPM. Charging
    floor rather than second price keeps exploration cheap for the advertiser who
    did not win on merit, and the logged serve_propensity lets the training job
    weight those rows correctly.
    """
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
        # Nobody to price against — the floor is the price.
        cpm = min(winner.ad.floor_price, winner.ad.target_cpm)

    # The model-chosen winner is served whenever we do not explore, plus its
    # share of the exploration branch.
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

    Signature notes for the gateway:

    `impression_id` is passed IN, not generated here. The gateway needs the id
    before this call returns (it goes into the click URL, the Postgres row, and
    the no_fill response), and the ML pipeline needs the same id on the
    impression event so clicks can be joined back to it. Two uuids means the
    training join matches nothing.

    `request_ts` replaces the old `hour_of_day` int. features.py derives hour,
    day-of-week and weekend from one timestamp, so passing the raw datetime
    keeps them consistent. Pass the SERVER time, not the client's
    `timestamp_ms` — the client clock is attacker-controlled, and a publisher
    that skews it shifts a feature the model depends on. Your ±60s validator
    bounds the damage but there is no reason to take it at all.

    Returns the RequestContext alongside the result so the gateway can build the
    impression event without recomputing normalised keywords.
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

    # Budget check goes here, AFTER the cheap DB filter and BEFORE scoring:
    # one Redis round trip for the whole candidate set, and no model inference
    # wasted on ads that cannot be served anyway.
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
        # ctx.request_ts, not now() — this is the timestamp the hour_of_day and
        # day_of_week features were derived from. Using a fresh now() here puts
        # the logged row a few milliseconds off its own features, which is
        # harmless until a request straddles an hour boundary and the row
        # disagrees with itself.
        "ts": ctx.request_ts.isoformat(),
        "publisher_id": ctx.publisher_id,
        "placement_id": ctx.placement_id,
        "ad_id": winner.ad.ad_id,
        # Required by budget.reconcile(), which GROUPs BY campaign_id to rebuild
        # Redis from this log. Safe to emit before migration 003 is applied:
        # kafka_impressions sets input_format_skip_unknown_fields = 1, so an
        # unrecognised field is ignored rather than rejected. "" rather than
        # None because the ClickHouse column is a non-nullable LowCardinality
        # (String) and JSONEachRow will not coerce a null into it.
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