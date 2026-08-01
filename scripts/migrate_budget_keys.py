# migrate_budget_keys.py — carry today's spend from the old per-ad Redis keys
# to the new per-campaign keys.
#
#     python -m scripts.migrate_budget_keys --dry-run
#     python -m scripts.migrate_budget_keys
#
# WHY THIS EXISTS
# ---------------
# budget.py used to key spend on ad_id ("budget:<ad_id>:<day>") and now keys on
# campaign_id ("budget:camp:<campaign_id>:<day>"). Different namespace, so on
# deploy every campaign reads 0 spent and gets a SECOND full daily budget for
# whatever remains of the day. On a $500 campaign that is a $500 mistake, and
# it happens once per campaign, silently.
#
# reconcile() would eventually fix it -- but only for impressions that carry a
# campaign_id, which means only those served after migration 003. Everything
# from earlier today has campaign_id = '' and is excluded from the reconcile
# query by design. So the gap is real and this script closes it.
#
# WHEN YOU CAN SKIP IT
# --------------------
# Deploy just after 00:00 UTC and there is nothing to carry over: both
# namespaces are empty for the new day. That is the cleaner option if you can
# choose your deploy window.
#
# ORDER
# -----
#   1. apply sql/003_campaign_budgets.sql
#   2. deploy the new gateway
#   3. run this
#
# Running it BEFORE the deploy is harmless but pointless -- the old gateway keeps
# writing ad keys, so the campaign totals go stale immediately.

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

import asyncpg
import redis.asyncio as aioredis

from adplatform.ml.budget import KEY_TTL_SECONDS, budget_key
from adplatform.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("migrate")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--day", default=None,
                    help="UTC date as YYYY-MM-DD (default: today)")
    ap.add_argument("--delete-old", action="store_true",
                    help="delete the old ad keys after summing (they expire on "
                         "their own 48h TTL anyway)")
    args = ap.parse_args()

    day = args.day or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    pg = await asyncpg.connect(settings.database_url, timeout=10)

    try:
        # ad_id -> campaign_id for every ad we know about. Ads deleted since
        # they last served are simply absent, and their spend is unattributable.
        mapping = {
            row["ad_id"]: row["campaign_id"]
            for row in await pg.fetch("SELECT ad_id, campaign_id FROM ads")
        }

        # SCAN, not KEYS. KEYS blocks the server for the length of the scan, and
        # this runs against the same Redis the auction reads from.
        #
        # The pattern must exclude the new namespace: "budget:*" would match
        # "budget:camp:..." too and we would sum the new keys into themselves.
        totals: dict[str, float] = defaultdict(float)
        unmapped: list[str] = []
        old_keys: list[str] = []

        async for key in r.scan_iter(match=f"budget:*:{day}", count=500):
            if key.startswith("budget:camp:"):
                continue
            # budget:<ad_id>:<day> -- ad_id may itself contain colons, so strip
            # the known prefix and suffix rather than splitting.
            ad_id = key[len("budget:"):-(len(day) + 1)]
            raw = await r.get(key)
            if raw is None:
                continue
            spent = float(raw)
            old_keys.append(key)

            campaign_id = mapping.get(ad_id)
            if campaign_id is None:
                unmapped.append(f"{ad_id} (${spent:.4f})")
                continue
            totals[campaign_id] += spent

        if unmapped:
            log.warning("%d ad key(s) had no campaign in Postgres; their spend "
                        "cannot be attributed and is dropped:", len(unmapped))
            for u in unmapped[:10]:
                log.warning("    %s", u)

        if not totals:
            log.info("nothing to migrate for %s", day)
            return 0

        for campaign_id, spent in sorted(totals.items()):
            new_key = budget_key(campaign_id, day)
            existing = await r.get(new_key)
            existing = float(existing) if existing else 0.0

            # MAX, not sum. If the new gateway has already been running, the
            # campaign key holds real spend from post-deploy impressions -- some
            # of which may also appear in the ad keys if both code paths ran.
            # Adding would double-count. Taking the larger keeps the advertiser
            # protected without inventing spend.
            merged = max(existing, spent)

            if args.dry_run:
                log.info("would set %s = %.6f (ad-key sum %.6f, existing %.6f)",
                         new_key, merged, spent, existing)
            else:
                await r.set(new_key, merged, ex=KEY_TTL_SECONDS)
                log.info("set %s = %.6f", new_key, merged)

        if args.delete_old and not args.dry_run:
            for k in old_keys:
                await r.delete(k)
            log.info("deleted %d old ad-keyed budget key(s)", len(old_keys))

        log.info("%s%d campaign(s), $%.4f total carried over",
                 "DRY RUN: " if args.dry_run else "", len(totals), sum(totals.values()))
        return 0
    finally:
        await pg.close()
        await r.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
