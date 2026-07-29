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
 
import asyncio
from dataclasses import dataclass, field
 
 
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
 
    @property
    def has_budget(self) -> bool:
        # NOTE: spent_today_usd is never incremented anywhere, so this is
        # always True. Real enforcement now lives in ml/budget.py, which keeps
        # spend in Redis and filters in run_rtb. This property is kept only so
        # existing code does not break; treat it as vestigial.
        return self.spent_today_usd < self.daily_budget_usd
 
 
# ---------------------------------------------------------------------------
# Mock ad inventory
# Replace with: async Postgres query filtered by device, status, floor_price
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
 
    Production version:
        rows = await pg_pool.fetch('''
            SELECT * FROM ads
            WHERE status = 'active'
              AND (target_device = 'all' OR target_device = $1)
              AND target_cpm >= $2
        ''', device_type, floor_price)
        return [Ad(**r) for r in rows]
    """
    await asyncio.sleep(0.002)  # Simulate ~2ms DB round trip
 
    return [
        ad for ad in MOCK_ADS
        if (ad.target_device == "all" or ad.target_device == device_type)
        and ad.target_cpm >= floor_price
    ]