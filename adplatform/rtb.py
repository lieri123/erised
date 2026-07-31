# rtb.py — Real-Time Bidding engine, stage 1.
#
# RECOVERED FILE. Your uploaded rtb.py was 0 bytes; this is reconstructed from
# the version you pasted at the start of our conversation.
#
# One deliberate change: stages 2 (score_ads) and 3 (run_auction) have been
# REMOVED from this file. They now live in ml/rtb_integration.py, which adds
# model inference, exploration, and the corrected CPM pricing. Keeping the old
# copies here would give you two definitions of ScoredAd, AuctionResult,
# run_auction and run_rtb — whichever import ran last would silently win, and
# you would have no way to tell which pricing logic was actually live.
#
# What stays here: the Ad model, the mock inventory, and get_eligible_ads.
# Stage 1 is a hard filter with no model involvement and needed no changes.
#
# The original stage 2/3 code is preserved in git history — and if you want it
# back verbatim, it is in the first message of our conversation.
 
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Module, not `from .inventory import current_inventory`. Same reasoning as the
# db._pool note in gateway.py: going through the module attribute means tests
# (and any future hot-swap of the snapshot) see the live function rather than a
# reference captured at import time. inventory.py imports rtb only lazily, from
# inside function bodies, so there is no cycle here.
from . import inventory
 
 
# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
 
@dataclass
class Ad:
    ad_id:             str
    advertiser_id:     str
    creative_html:     str        # HTML markup — {{CLICK_URL}} replaced by gateway
    destination_url:   str        # Where user lands after clicking
    target_cpm:        float      # Max bid in USD per 1000 impressions
    floor_price:       float      # Min CPM — anything below this is excluded
    target_device:     str        # "all" | "mobile" | "desktop" | "tablet"
    target_keywords:   list[str]  = field(default_factory=list)
    daily_budget_usd:  float      = 500.0
    spent_today_usd:   float      = 0.0

    # Both supplied by the servable_ads view via inventory._row_to_ad.
    #
    # These were missing until now, which meant _row_to_ad raised TypeError on
    # EVERY row, the per-row `except` swallowed it as "skipping unloadable ad",
    # and load_inventory returned 0 ads on every refresh. Nothing failed loudly
    # because get_eligible_ads was still reading MOCK_ADS.
    #
    # campaign_id is the budget key. Budgets are set per campaign, so spend must
    # be counted per campaign — see the note in inventory._row_to_ad. budget.py
    # still keys on ad_id; that is a separate bug (a five-creative campaign can
    # spend five times its budget) and this field is what you need to fix it.
    #
    # created_at feeds the ad_age_days feature. features.py reads it with
    # getattr(ad, "created_at", None), so it degraded to 0.0 silently rather
    # than raising — one of your seventeen features was dead.
    campaign_id:       Optional[str]      = None
    created_at:        Optional[datetime] = None
 
    @property
    def has_budget(self) -> bool:
        # NOTE: spent_today_usd is never incremented anywhere, so this is
        # always True. Real enforcement now lives in ml/budget.py, which keeps
        # spend in Redis and filters in run_rtb. This property is kept only so
        # existing code does not break; treat it as vestigial.
        return self.spent_today_usd < self.daily_budget_usd
 
 
# ---------------------------------------------------------------------------
# Mock ad inventory — DEV COLD-CACHE FALLBACK ONLY.
#
# No longer the serving source. get_eligible_ads reads inventory.current_
# inventory(). The only remaining reader is inventory.current_inventory()
# itself, which falls back to this list when the snapshot has never loaded AND
# settings.is_production is False, so the stack boots and demos without
# Postgres. In production a cold cache serves nothing, deliberately.
#
# Do not add ads here expecting them to serve. Use POST /v1/campaigns and
# POST /v1/ads.
# ---------------------------------------------------------------------------
 
MOCK_ADS: list[Ad] = [
    Ad(
        ad_id="ad_001", advertiser_id="adv_acme_cloud",
        creative_html="""<div style="background:#4F46E5;color:#fff;padding:18px 22px;border-radius:10px;
            font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:space-between;gap:16px">
          <div>
            <div style="font-weight:600;font-size:15px">Acme Cloud Platform</div>
            <div style="opacity:.8;font-size:13px;margin-top:3px">99.99% uptime. Scale in seconds.</div>
          </div>
          <a href="{{CLICK_URL}}" style="background:#fff;color:#4F46E5;padding:8px 16px;border-radius:6px;
            text-decoration:none;font-size:13px;font-weight:500;white-space:nowrap">Learn more</a>
        </div>""",
        destination_url="https://acme-cloud.example.com",
        target_cpm=5.00, floor_price=1.00,
        target_device="all",
        target_keywords=["cloud", "infrastructure", "devops", "python", "software", "server"],
    ),
    Ad(
        ad_id="ad_002", advertiser_id="adv_shopx",
        creative_html="""<div style="background:#059669;color:#fff;padding:18px 22px;border-radius:10px;
            font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:space-between;gap:16px">
          <div>
            <div style="font-weight:600;font-size:15px">ShopX — 50% Off Everything</div>
            <div style="opacity:.8;font-size:13px;margin-top:3px">Free shipping. Limited time.</div>
          </div>
          <a href="{{CLICK_URL}}" style="background:#fff;color:#059669;padding:8px 16px;border-radius:6px;
            text-decoration:none;font-size:13px;font-weight:500;white-space:nowrap">Shop now</a>
        </div>""",
        destination_url="https://shopx.example.com",
        target_cpm=3.50, floor_price=0.50,
        target_device="all",
        target_keywords=["shopping", "deals", "ecommerce", "retail", "sale"],
    ),
    Ad(
        ad_id="ad_003", advertiser_id="adv_finco",
        creative_html="""<div style="background:#0369A1;color:#fff;padding:18px 22px;border-radius:10px;
            font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:space-between;gap:16px">
          <div>
            <div style="font-weight:600;font-size:15px">FinCo — Earn 5.2% APY</div>
            <div style="opacity:.8;font-size:13px;margin-top:3px">FDIC insured. No fees. No minimums.</div>
          </div>
          <a href="{{CLICK_URL}}" style="background:#fff;color:#0369A1;padding:8px 16px;border-radius:6px;
            text-decoration:none;font-size:13px;font-weight:500;white-space:nowrap">Open account</a>
        </div>""",
        destination_url="https://finco.example.com",
        target_cpm=8.00, floor_price=2.00,
        target_device="desktop",
        target_keywords=["finance", "investing", "savings", "banking", "money"],
    ),
    Ad(
        ad_id="ad_004", advertiser_id="adv_devtools",
        creative_html="""<div style="background:#18181b;color:#fff;padding:18px 22px;border-radius:10px;
            font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:space-between;gap:16px">
          <div>
            <div style="font-weight:600;font-size:15px">DevTools Pro</div>
            <div style="opacity:.8;font-size:13px;margin-top:3px">Debug faster. Ship with confidence. Free trial.</div>
          </div>
          <a href="{{CLICK_URL}}" style="background:#7F77DD;color:#fff;padding:8px 16px;border-radius:6px;
            text-decoration:none;font-size:13px;font-weight:500;white-space:nowrap">Try free</a>
        </div>""",
        destination_url="https://devtools.example.com",
        target_cpm=6.00, floor_price=1.50,
        target_device="desktop",
        target_keywords=["python", "developer", "programming", "tutorial", "code", "software", "debug"],
    ),
    Ad(
        ad_id="ad_005", advertiser_id="adv_learnpro",
        creative_html="""<div style="background:#7C3AED;color:#fff;padding:18px 22px;border-radius:10px;
            font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:space-between;gap:16px">
          <div>
            <div style="font-weight:600;font-size:15px">LearnPro — Tech Courses</div>
            <div style="opacity:.8;font-size:13px;margin-top:3px">Python, ML, Cloud. Self-paced. $19/mo.</div>
          </div>
          <a href="{{CLICK_URL}}" style="background:#fff;color:#7C3AED;padding:8px 16px;border-radius:6px;
            text-decoration:none;font-size:13px;font-weight:500;white-space:nowrap">Start learning</a>
        </div>""",
        destination_url="https://learnpro.example.com",
        target_cpm=4.50, floor_price=1.00,
        target_device="all",
        target_keywords=["python", "tutorial", "learning", "course", "programming", "ml", "cloud"],
    ),
]
 
 
# ---------------------------------------------------------------------------
# Stage 1 — Filter eligible ads
# ---------------------------------------------------------------------------
 
async def get_eligible_ads(
    device_type: str,
    page_keywords: list[str],
    floor_price: float = 0.0,
) -> list[Ad]:
    """
    Return ads that pass all hard targeting filters. Binary pass/fail — no
    scoring or ranking happens here.
 
    Budget is deliberately NOT checked here any more. ml/budget.py does it in
    run_rtb, one batched Redis read for the whole candidate set, because
    spent_today_usd on these objects is never updated.
 
    `page_keywords` is currently unused — keyword matching is a scoring signal
    in stage 2, not a hard filter. Kept in the signature because run_rtb passes
    it and you will likely want a "must match at least one keyword" mode later.
 
    SOURCE OF ADS: inventory.current_inventory(), the in-process snapshot
    refreshed from the servable_ads view every inventory_refresh_seconds. Not a
    query — see the header of inventory.py for why one round trip per auction is
    the wrong shape. Not MOCK_ADS either, which is what this function used to
    read: every ad an advertiser created through POST /v1/ads was loaded into
    that snapshot and then never served, because nothing called it.
 
    Status, date windows and advertiser-active are already applied by the view,
    so they are deliberately not re-checked here. Budget is applied later, in
    ml/budget.filter_by_budget, as one batched Redis read over the whole
    candidate set.
 
    No await left in the body. Kept async because run_rtb awaits it and because
    a future variant may need I/O; an async function with no suspension point
    costs one coroutine allocation and nothing else.
    """
    ads = inventory.current_inventory()
 
    return [
        ad for ad in ads
        if (ad.target_device == "all" or ad.target_device == device_type)
        and ad.target_cpm >= floor_price
    ]