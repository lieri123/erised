# rtb.py — Real-Time Bidding engine, stage 1.

from datetime import datetime
from typing import Optional
from . import inventory
 
 
# Data model
 
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

    campaign_id:       Optional[str]      = None
    created_at:        Optional[datetime] = None
 
    @property
    def has_budget(self) -> bool:
        return self.spent_today_usd < self.daily_budget_usd
 
 
# Mock ad inventory — DEV COLD-CACHE FALLBACK ONLY.
 
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
 
 
# Stage 1 — Filter eligible ads
 
async def get_eligible_ads(
    device_type: str,
    page_keywords: list[str],
    floor_price: float = 0.0,
) -> list[Ad]:
    """
    Return ads that pass all hard targeting filters. Binary pass/fail — no
    scoring or ranking happens here.
    """
    ads = inventory.current_inventory()
 
    return [
        ad for ad in ads
        if (ad.target_device == "all" or ad.target_device == device_type)
        and ad.target_cpm >= floor_price
    ]