# seed_inventory.py — fill advertisers/campaigns/ads so bids can actually fill.
#
#   make seed
#   python -m scripts.seed_inventory --advertisers 12
#   python -m scripts.seed_inventory --dry-run          # print the plan, touch nothing
#   python -m scripts.seed_inventory --rotate-key       # issue a fresh publisher key
#   python -m scripts.seed_inventory --purge            # delete seeded rows first

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import date, datetime, timedelta, timezone

import asyncpg

from adplatform.auth import generate_api_key, hash_api_key, key_prefix
from adplatform.settings import settings

logging.basicConfig(
    level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("seed")

PREFIX = "seed_"
DEMO_PUBLISHER_ID = "pub_demo"
DEMO_DOMAIN = "demo.localhost"

RNG_SEED = 20260731


# Generated content

VERTICALS = [
    ("acme_cloud",   "Acme Cloud",        ["cloud", "devops", "kubernetes", "hosting"]),
    ("shopx",        "ShopX Retail",      ["shopping", "retail", "ecommerce", "deals"]),
    ("finco",        "FinCo Banking",     ["finance", "investing", "banking", "savings"]),
    ("devtools",     "DevTools Inc",      ["python", "programming", "ide", "debugging"]),
    ("learnpro",     "LearnPro Academy",  ["education", "courses", "training", "career"]),
    ("travelnow",    "TravelNow",         ["travel", "flights", "hotels", "vacation"]),
    ("fitlife",      "FitLife Health",    ["fitness", "health", "nutrition", "wellness"]),
    ("autohaus",     "AutoHaus Motors",   ["cars", "automotive", "ev", "leasing"]),
    ("greenenergy",  "Green Energy Co",   ["solar", "energy", "sustainability", "climate"]),
    ("gamerz",       "Gamerz Network",    ["gaming", "esports", "console", "streaming"]),
    ("homestead",    "Homestead Realty",  ["realestate", "mortgage", "housing", "rent"]),
    ("mediaflow",    "MediaFlow",         ["streaming", "video", "podcast", "music"]),
    ("securenet",    "SecureNet",         ["security", "vpn", "privacy", "encryption"]),
    ("foodbox",      "FoodBox Delivery",  ["food", "delivery", "recipes", "grocery"]),
    ("petpal",       "PetPal Supplies",   ["pets", "dogs", "cats", "petcare"]),
]

DEVICES = ["all", "all", "all", "mobile", "desktop", "tablet"]  # weighted to "all"


def normalise_keywords(words: list[str]) -> list[str]:
    """
    Match CampaignRequest.normalise_keywords in gateway.py: lowercased, deduped,
    sorted. The servable_ads view is read straight into scoring with no further
    cleanup, so anything inserted here must already be in that shape or keyword
    overlap silently under-counts.
    """
    return sorted({w.strip().lower() for w in words if w.strip()})


def creative(brand: str, headline: str, variant: str) -> str:
    """
    {{CLICK_URL}} is substituted by the gateway at serve time. An ad without it
    renders but is unclickable, which produces a training set with no positive
    labels — see train_ctr.py's 'no positive examples' exit.
    """
    return (
        f'<a href="{{{{CLICK_URL}}}}" class="ad ad--{variant}">'
        f'<span class="ad__brand">{brand}</span>'
        f'<span class="ad__headline">{headline}</span>'
        f"</a>"
    )


def build_plan(n_advertisers: int) -> list[dict]:
    rng = random.Random(RNG_SEED)
    now = datetime.now(timezone.utc)
    plan = []

    for i in range(n_advertisers):
        slug, name, kws = VERTICALS[i % len(VERTICALS)]
        if i >= len(VERTICALS):
            slug = f"{slug}_{i // len(VERTICALS)}"

        adv_id = f"{PREFIX}adv_{slug}"
        campaigns = []

        for c in range(rng.randint(1, 2)):
            cid = f"{PREFIX}camp_{slug}_{c}"
            target_cpm = round(rng.uniform(2.0, 14.0), 2)
            campaigns.append({
                "campaign_id": cid,
                "advertiser_id": adv_id,
                "name": f"{name} — {'brand' if c == 0 else 'performance'}",
                "daily_budget_usd": float(rng.choice([50, 100, 250, 500, 1000])),
                "target_cpm": target_cpm,
                "floor_price": round(target_cpm * rng.uniform(0.2, 0.6), 2),
                "target_device": rng.choice(DEVICES),
                "target_keywords": normalise_keywords(
                    rng.sample(kws, k=rng.randint(2, len(kws)))
                ),
                "start_date": None,
                "end_date": None,
                "ads": [
                    {
                        "ad_id": f"{PREFIX}ad_{slug}_{c}{chr(97 + a)}",
                        "campaign_id": cid,
                        "name": f"{name} creative {chr(65 + a)}",
                        "creative_html": creative(
                            name,
                            rng.choice([
                                "See what you're missing",
                                "Try it free for 30 days",
                                "Built for people who ship",
                                "Switch in under five minutes",
                                "Join 40,000 others",
                            ]),
                            chr(97 + a),
                        ),
                        "destination_url": f"https://{slug}.example.com/?utm_source=adplatform",
                        "created_at": now - timedelta(days=rng.randint(0, 90),
                                                      hours=rng.randint(0, 23)),
                    }
                    for a in range(rng.randint(2, 3))
                ],
            })

        plan.append({
            "advertiser_id": adv_id,
            "name": name,
            "contact_email": f"ads@{slug}.example.com",
            "campaigns": campaigns,
        })

    return plan


# Writing

async def insert_plan(conn: asyncpg.Connection, plan: list[dict]) -> tuple[int, int, int]:
    n_adv = n_camp = n_ad = 0

    async with conn.transaction():
        for adv in plan:
            await conn.execute(
                """
                INSERT INTO advertisers (advertiser_id, name, contact_email, active)
                VALUES ($1, $2, $3, TRUE)
                ON CONFLICT (advertiser_id) DO NOTHING
                """,
                adv["advertiser_id"], adv["name"], adv["contact_email"],
            )
            n_adv += 1

            for camp in adv["campaigns"]:
                await conn.execute(
                    """
                    INSERT INTO campaigns (campaign_id, advertiser_id, name,
                                           daily_budget_usd, target_cpm, floor_price,
                                           target_device, target_keywords,
                                           start_date, end_date, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, 'active')
                    ON CONFLICT (campaign_id) DO NOTHING
                    """,
                    camp["campaign_id"], camp["advertiser_id"], camp["name"],
                    camp["daily_budget_usd"], camp["target_cpm"], camp["floor_price"],
                    camp["target_device"],
                    json.dumps(camp["target_keywords"]),
                    camp["start_date"], camp["end_date"],
                )
                n_camp += 1

                for ad in camp["ads"]:
                    await conn.execute(
                        """
                        INSERT INTO ads (ad_id, campaign_id, name, creative_html,
                                         destination_url, status, created_at)
                        VALUES ($1, $2, $3, $4, $5, 'active', $6)
                        ON CONFLICT (ad_id) DO NOTHING
                        """,
                        ad["ad_id"], ad["campaign_id"], ad["name"],
                        ad["creative_html"], ad["destination_url"], ad["created_at"],
                    )
                    n_ad += 1

    return n_adv, n_camp, n_ad


async def ensure_publisher_key(conn: asyncpg.Connection, rotate: bool) -> str | None:
    """
    Make sure pub_demo exists and has a usable key. Returns the plaintext key if
    one was issued, else None — an existing key cannot be recovered, only
    replaced, because the table stores an HMAC and nothing else.
    """
    await conn.execute(
        """
        INSERT INTO publishers (publisher_id, domain, active)
        VALUES ($1, $2, TRUE)
        ON CONFLICT (publisher_id) DO UPDATE SET active = TRUE
        """,
        DEMO_PUBLISHER_ID, DEMO_DOMAIN,
    )

    existing = await conn.fetchval(
        """
        SELECT count(*) FROM api_keys
         WHERE owner_type = 'publisher' AND owner_id = $1
           AND active = TRUE AND revoked_at IS NULL
        """,
        DEMO_PUBLISHER_ID,
    )

    if existing and not rotate:
        return None

    if rotate and existing:
        await conn.execute(
            """
            UPDATE api_keys SET active = FALSE, revoked_at = now()
             WHERE owner_type = 'publisher' AND owner_id = $1 AND active = TRUE
            """,
            DEMO_PUBLISHER_ID,
        )
        log.info("revoked %d existing key(s) for %s", existing, DEMO_PUBLISHER_ID)

    raw = generate_api_key(test=not settings.is_production)
    await conn.execute(
        """
        INSERT INTO api_keys (key_id, owner_type, owner_id, key_hash, key_prefix,
                              name, active)
        VALUES ($1, 'publisher', $2, $3, $4, 'seeded demo key', TRUE)
        """,
        f"{PREFIX}key_{date.today():%Y%m%d}_{key_prefix(raw)[-6:]}",
        DEMO_PUBLISHER_ID, hash_api_key(raw), key_prefix(raw),
    )
    return raw


async def purge(conn: asyncpg.Connection) -> None:
    """Delete in FK order. Only touches rows this script created."""
    async with conn.transaction():
        for sql in (
            f"DELETE FROM ads WHERE ad_id LIKE '{PREFIX}%'",
            f"DELETE FROM campaigns WHERE campaign_id LIKE '{PREFIX}%'",
            f"DELETE FROM advertisers WHERE advertiser_id LIKE '{PREFIX}%'",
            f"DELETE FROM api_keys WHERE key_id LIKE '{PREFIX}%'",
        ):
            result = await conn.execute(sql)
            log.info("purge: %s", result)

# Entry point

async def main() -> int:
    ap = argparse.ArgumentParser(description="Seed demo inventory into Postgres.")
    ap.add_argument("--advertisers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without connecting")
    ap.add_argument("--rotate-key", action="store_true",
                    help="revoke pub_demo's keys and issue a fresh one")
    ap.add_argument("--purge", action="store_true",
                    help="delete previously seeded rows before inserting")
    args = ap.parse_args()

    if args.advertisers < 1:
        log.error("--advertisers must be at least 1")
        return 2

    if settings.is_production:
        log.error("refusing to seed with ENV=production")
        return 2

    plan = build_plan(args.advertisers)
    n_camp = sum(len(a["campaigns"]) for a in plan)
    n_ad = sum(len(c["ads"]) for a in plan for c in a["campaigns"])

    if args.dry_run:
        for adv in plan:
            print(f"{adv['advertiser_id']}  ({adv['name']})")
            for c in adv["campaigns"]:
                print(f"    {c['campaign_id']}  cpm={c['target_cpm']:>5.2f} "
                      f"floor={c['floor_price']:>5.2f} device={c['target_device']:<8} "
                      f"budget=${c['daily_budget_usd']:.0f} kw={c['target_keywords']}")
                for ad in c["ads"]:
                    age = (datetime.now(timezone.utc) - ad["created_at"]).days
                    print(f"        {ad['ad_id']:<34} age={age:>2}d")
        print(f"\n{len(plan)} advertisers, {n_camp} campaigns, {n_ad} ads (dry run — nothing written)")
        return 0

    try:
        conn = await asyncpg.connect(settings.database_url, timeout=10)
    except Exception as exc:
        log.error("cannot reach Postgres at %s: %s", settings.database_url, exc)
        log.error("is the stack up? try: make up")
        return 1

    try:
        if not await conn.fetchval("SELECT to_regclass('public.servable_ads') IS NOT NULL"):
            log.error("servable_ads view missing — run `make bootstrap` first")
            return 1

        if args.purge:
            await purge(conn)

        n_adv, n_camp, n_ad = await insert_plan(conn, plan)
        raw_key = await ensure_publisher_key(conn, rotate=args.rotate_key)

        servable = await conn.fetchval("SELECT count(*) FROM servable_ads")
    finally:
        await conn.close()

    log.info("seeded %d advertisers, %d campaigns, %d ads", n_adv, n_camp, n_ad)

    print()
    print(f"  servable_ads now returns {servable} rows.")
    print("  The gateway picks these up within INVENTORY_REFRESH_SECONDS "
          f"({settings.inventory_refresh_seconds}s), or immediately on restart.")
    print()

    if raw_key:
        print("  Publisher API key (shown once — only its HMAC is stored):")
        print()
        print(f"      {raw_key}")
        print()
    else:
        print(f"  {DEMO_PUBLISHER_ID} already has an active key. It cannot be")
        print("  reprinted; re-run with --rotate-key to issue a new one.")
        print()

    print("  Then:")
    print("      curl -s localhost:8000/v1/bid \\")
    print("        -H 'Authorization: Bearer <key>' \\")
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"placement_id":"plc_demo","device_type":"mobile",'
          '"page_url":"https://demo.localhost/a","page_keywords":["python"]}\'')
    print()

    if servable == 0:
        log.error("servable_ads is empty after seeding — check campaign status "
                  "and advertiser.active")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
