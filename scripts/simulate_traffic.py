# simulate_traffic.py — generate a labelled dataset the CTR model can learn from.
#
#     python -m scripts.simulate_traffic --impressions 20000
#     python -m scripts.simulate_traffic --impressions 20000 --concurrency 32
#     python -m scripts.simulate_traffic --check          # readiness only, no traffic
#
# WHY A SIMULATOR AND NOT JUST TRAFFIC
# ------------------------------------
# train_ctr.py refuses to train on data that cannot teach it anything, and it is
# right to. Three separate gates have to be cleared:
#
#   * enough rows to split three ways (train / valid / calib)
#   * at least one positive label
#   * AUC > 0.55 on the calibration split
#
# The last one is the hard one, and it is why this file is more than a for-loop
# around curl. If clicks are random, no model can beat 0.55 no matter how many
# rows you generate, and training correctly refuses to promote. The data needs a
# LEARNABLE STRUCTURE: a true click probability that actually depends on the
# features the model gets to see.
#
# WHAT SIGNAL IS PLANTED
# ----------------------
# Each simulated click is drawn from a true CTR built out of exactly three
# things, all of them visible in the feature vector:
#
#   1. per-ad quality      -> shows up via ad_ctr_prior / pair_ctr_prior
#   2. keyword relevance   -> shows up via keyword_overlap, overlap_ratio
#   3. device effect       -> shows up via is_mobile / is_desktop / is_tablet
#
# and one thing that is NOT visible: per-user propensity. That is deliberate.
# Real CTR has irreducible noise, and a simulator where the model can reach
# AUC 0.99 is not testing anything — it is testing that you can fit a formula
# you wrote yourself. The hidden term keeps achievable AUC in the 0.65-0.80
# range, which is roughly what real display advertising looks like.
#
# TIME TRAVEL: THE ONE THING THAT WILL CONFUSE YOU
# ------------------------------------------------
# train_ctr.py's query ends with:
#
#     AND i.ts < now() - INTERVAL 2 HOUR
#
# LABEL_CUTOFF_HOURS = 2. An impression logged one minute ago is EXCLUDED from
# training, because its click may not have arrived yet and counting it as a
# negative would be a lie. That is correct behaviour, and it means traffic you
# generate right now is not trainable for two hours.
#
# So this script does NOT rely on the gateway's own clock for the ClickHouse
# rows. It writes impressions directly to ClickHouse with backdated timestamps
# spread over the last few days (--spread-days), so training can run
# immediately. The bid requests still go through the real gateway — the auction,
# budget checks, and Kafka path are all exercised — but the trainable rows are
# inserted separately with historical ts values.
#
# This is a deliberate, disclosed shortcut for local development. It is NOT how
# you would validate the pipeline in production, where you would generate load
# and wait out the cutoff.

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from adplatform.ml.features import FEATURE_VERSION, N_FEATURES
from adplatform.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("simulate")

RNG_SEED = 20260801

DEVICES = ["mobile", "mobile", "mobile", "desktop", "desktop", "tablet"]

KEYWORD_POOL = [
    "cloud", "devops", "kubernetes", "hosting", "shopping", "retail", "ecommerce",
    "deals", "finance", "investing", "banking", "savings", "python", "programming",
    "ide", "debugging", "education", "courses", "training", "career", "travel",
    "flights", "hotels", "vacation", "fitness", "health", "nutrition", "wellness",
    "cars", "automotive", "ev", "leasing", "solar", "energy", "gaming", "esports",
    "security", "vpn", "privacy", "food", "delivery", "recipes", "pets", "dogs",
]

# Device multipliers on the true CTR. Mobile over-indexes on display, which is
# both realistic and gives is_mobile/is_desktop something to separate.
DEVICE_EFFECT = {"mobile": 1.35, "desktop": 0.75, "tablet": 1.0}

BASE_CTR = 0.02


# ---------------------------------------------------------------------------
# The latent world
# ---------------------------------------------------------------------------

class World:
    """
    Hidden ground truth. The gateway never sees this; it is what CLICKS are
    drawn from, and what the model is being asked to rediscover from features.
    """

    def __init__(self, ad_ids: list[str], rng: random.Random):
        self.rng = rng
        # Log-normal ad quality: most ads mediocre, a few much better. A uniform
        # spread would be easier to learn than anything real.
        self.ad_quality = {
            ad_id: math.exp(rng.gauss(0.0, 0.55)) for ad_id in ad_ids
        }
        # Per-user propensity is INVISIBLE to the model — the irreducible noise
        # that keeps achievable AUC realistic instead of ~1.0.
        self.user_propensity: dict[str, float] = {}

    def propensity(self, user_id: str) -> float:
        if user_id not in self.user_propensity:
            self.user_propensity[user_id] = math.exp(self.rng.gauss(0.0, 0.45))
        return self.user_propensity[user_id]

    def true_ctr(self, ad_id: str, user_id: str, device: str, overlap: int) -> float:
        p = (
            BASE_CTR
            * self.ad_quality.get(ad_id, 1.0)
            * DEVICE_EFFECT.get(device, 1.0)
            * self.propensity(user_id)
            * (1.0 + 0.45 * min(overlap, 4))     # relevance, saturating
        )
        return min(p, 0.60)


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

async def preflight(client: httpx.AsyncClient, base: str) -> dict:
    """Fail early and specifically rather than after 20k pointless requests."""
    try:
        health = (await client.get(f"{base}/health", timeout=10)).json()
    except Exception as exc:
        raise SystemExit(f"gateway unreachable at {base}: {exc}\ntry: docker compose up -d")

    problems = []
    inv = health.get("inventory", {})
    if inv.get("ads", 0) == 0:
        problems.append("inventory is empty — run: docker compose run --rm bootstrap "
                        "python -m scripts.seed_inventory")
    if health.get("auth", {}).get("active_keys", 0) == 0:
        problems.append("no active API keys — re-run the seed script")
    for store, state in health.get("stores", {}).items():
        if state != "up":
            problems.append(f"{store} is {state}")

    if problems:
        raise SystemExit("not ready:\n  - " + "\n  - ".join(problems))

    log.info("gateway ok — %d ads, model=%s", inv.get("ads"), health.get("ctr_model"))
    return health


# ---------------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------------

async def one_request(client, base, key, world, rng, stats) -> dict | None:
    user_id = f"u_{rng.randint(1, 4000)}"
    device = rng.choice(DEVICES)
    keywords = rng.sample(KEYWORD_POOL, k=rng.randint(1, 5))

    body = {
        "publisher_id": "pub_demo",
        "placement_id": f"plc_{rng.randint(1, 6)}",
        "user_id": user_id,
        "device_type": device,
        "page_url": f"https://demo.localhost/article/{rng.randint(1, 200)}",
        "page_keywords": keywords,
        "timestamp_ms": int(time.time() * 1000),
    }

    try:
        r = await client.post(f"{base}/v1/bid", json=body,
                              headers={"X-API-Key": key}, timeout=15)
    except Exception:
        stats["errors"] += 1
        return None

    if r.status_code == 429:
        # BID_RATE_LIMIT defaults to 120/minute — two requests a second. The
        # gateway is behaving correctly; the simulator has to slow down.
        stats["rate_limited"] += 1
        return None
    if r.status_code == 204:
        stats["no_fill"] += 1
        return None
    if r.status_code != 200:
        stats["errors"] += 1
        if stats["errors"] <= 3:
            log.warning("bid failed %s: %s", r.status_code, r.text[:200])
        return None

    bid = r.json()
    stats["filled"] += 1

    # Overlap the model will see, recomputed here to drive the true CTR.
    overlap = len(set(keywords) & set(KEYWORD_POOL[:12]))
    p = world.true_ctr(bid["ad_id"], user_id, device, overlap)
    clicked = rng.random() < p

    if clicked:
        stats["clicks"] += 1
        # Follow the SIGNED url out of the creative — exercises the real
        # signature path rather than assuming it works.
        import re
        m = re.search(r'href="([^"]+)"', bid.get("ad_markup", ""))
        if m:
            try:
                cr = await client.get(m.group(1), follow_redirects=False, timeout=15)
                if cr.status_code == 403:
                    stats["click_403"] += 1
            except Exception:
                stats["errors"] += 1

    return {
        "impression_id": bid["impression_id"],
        "ad_id": bid["ad_id"],
        "device": device,
        "keywords": keywords,
        "user_id": user_id,
        "overlap": overlap,
        "clicked": clicked,
        "predicted_ctr": bid.get("predicted_ctr", 0.0),
        "win_price": bid.get("win_price", 0.0),
    }


async def run_traffic(n: int, concurrency: int, base: str, key: str,
                      world: World, rng: random.Random) -> list[dict]:
    stats = {"filled": 0, "no_fill": 0, "errors": 0, "clicks": 0,
             "click_403": 0, "rate_limited": 0}
    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        await preflight(client, base)

        async def worker(_):
            async with sem:
                row = await one_request(client, base, key, world, rng, stats)
                if row:
                    results.append(row)

        started = time.time()
        for chunk_start in range(0, n, 500):
            chunk = range(chunk_start, min(chunk_start + 500, n))
            await asyncio.gather(*(worker(i) for i in chunk))
            done = min(chunk_start + 500, n)
            rate = done / max(time.time() - started, 0.001)
            log.info("%d/%d  filled=%d clicks=%d 429=%d err=%d  (%.0f req/s)",
                     done, n, stats["filled"], stats["clicks"],
                     stats["rate_limited"], stats["errors"], rate)

            # Bail out rather than spend twenty minutes collecting 429s.
            if done >= 500 and stats["rate_limited"] > done * 0.5:
                raise SystemExit(
                    f"\n{stats['rate_limited']} of {done} requests were rate "
                    f"limited (429).\n\n"
                    f"BID_RATE_LIMIT defaults to 120/minute — two per second. "
                    f"Raise it for the gateway and retry:\n\n"
                    f"    docker compose down gateway\n"
                    f"    docker compose run -d --service-ports "
                    f"-e BID_RATE_LIMIT=10000/minute --name adplatform-gateway-1 gateway\n\n"
                    f"or add BID_RATE_LIMIT=10000/minute under the gateway's "
                    f"environment: block in docker-compose.yml and "
                    f"`docker compose up -d gateway`.")

    if stats["click_403"]:
        log.error("%d click(s) rejected with 403 — signed URL verification is "
                  "failing; clicks are NOT being recorded", stats["click_403"])
    if stats["filled"] == 0:
        raise SystemExit("no bids filled — check budgets and inventory")

    log.info("traffic done: %d filled, %d clicks (%.2f%% raw CTR), %d no-fill, "
             "%d rate-limited, %d errors",
             stats["filled"], stats["clicks"],
             100.0 * stats["clicks"] / max(stats["filled"], 1),
             stats["no_fill"], stats["rate_limited"], stats["errors"])
    return results


# ---------------------------------------------------------------------------
# Backdated rows for training
# ---------------------------------------------------------------------------

def feature_vector(row: dict, ts: datetime, ad_stats: dict) -> list[float]:
    """
    Build a vector in exactly FEATURE_NAMES order. Order is load-bearing —
    XGBoost consumes a positional array, and a reordered vector produces
    confident nonsense rather than an error.
    """
    kw = row["keywords"]
    n_ad_kw = 3
    overlap = row["overlap"]
    seen, clicks = ad_stats.get(row["ad_id"], (0, 0))

    return [
        float(ts.hour),
        float(ts.weekday()),
        1.0 if ts.weekday() >= 5 else 0.0,
        1.0 if row["device"] == "mobile" else 0.0,
        1.0 if row["device"] == "desktop" else 0.0,
        1.0 if row["device"] == "tablet" else 0.0,
        float(overlap),
        float(overlap) / n_ad_kw,
        float(n_ad_kw),
        float(len(kw)),
        float(row["win_price"]),
        (clicks + 1.0) / (seen + 100.0),          # ad_ctr_prior, smoothed
        0.02,                                      # placement_ctr_prior
        (clicks + 1.0) / (seen + 100.0),          # pair_ctr_prior
        math.log1p(seen),                          # pair_impressions_log
        float((datetime.now(timezone.utc) - ts).days % 90),
        0.1,                                       # budget_pacing
    ]


def insert_backdated(rows: list[dict], spread_days: float, dsn_host: str,
                     dsn_port: int) -> tuple[int, int]:
    """
    Write impressions and clicks straight to ClickHouse with historical
    timestamps, so they clear train_ctr's `ts < now() - 2 HOUR` cutoff
    immediately. See the header for why this shortcut exists.
    """
    import clickhouse_connect

    client = clickhouse_connect.get_client(host=dsn_host, port=dsn_port)
    now = datetime.now(timezone.utc)

    imp_rows, click_rows = [], []
    ad_stats: dict[str, tuple[int, int]] = {}

    # Oldest first, so the running ad_ctr_prior a row sees reflects only
    # EARLIER rows. Computing it over the whole dataset would leak the label
    # into the feature and produce a gorgeous, meaningless AUC.
    for i, row in enumerate(rows):
        age_hours = 3.0 + (spread_days * 24.0) * (1.0 - i / max(len(rows), 1))
        ts = now - timedelta(hours=age_hours)

        vec = feature_vector(row, ts, ad_stats)
        imp_id = row["impression_id"] if _is_uuid(row["impression_id"]) else str(uuid.uuid4())

        imp_rows.append([
            uuid.UUID(imp_id), ts, "pub_demo", "plc_1", row["ad_id"],
            "adv_sim", row["device"], FEATURE_VERSION,
            [float(x) for x in vec], 0.02, 5.0, 5.0, 0, 1.0, "simulated",
            "camp_sim",
        ])

        if row["clicked"]:
            click_rows.append([uuid.UUID(imp_id), ts + timedelta(seconds=30),
                               row["ad_id"], "plc_1"])

        seen, clicks = ad_stats.get(row["ad_id"], (0, 0))
        ad_stats[row["ad_id"]] = (seen + 1, clicks + (1 if row["clicked"] else 0))

    client.insert(
        "ad_impressions", imp_rows,
        column_names=["impression_id", "ts", "publisher_id", "placement_id",
                      "ad_id", "advertiser_id", "device_type", "feature_version",
                      "features", "predicted_ctr", "bid_value", "win_price",
                      "is_exploration", "serve_propensity", "model_version",
                      "campaign_id"],
    )
    if click_rows:
        client.insert("ad_clicks", click_rows,
                      column_names=["impression_id", "ts", "ad_id", "placement_id"])

    return len(imp_rows), len(click_rows)


def _is_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------

async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impressions", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default=None,
                    help="publisher key; defaults to $SIM_API_KEY")
    ap.add_argument("--spread-days", type=float, default=5.0)
    ap.add_argument("--no-backdate", action="store_true",
                    help="skip the ClickHouse insert; rely on real gateway rows "
                         "(then wait out the 2h label cutoff before training)")
    ap.add_argument("--clickhouse-host", default="localhost")
    ap.add_argument("--clickhouse-port", type=int, default=8123)
    ap.add_argument("--check", action="store_true", help="readiness check only")
    args = ap.parse_args()

    import os
    key = args.api_key or os.environ.get("SIM_API_KEY")
    if not key:
        log.error("no API key. Pass --api-key, or set SIM_API_KEY.")
        log.error("Get one: docker compose run --rm bootstrap python -m "
                  "scripts.seed_inventory --rotate-key")
        return 2

    if args.check:
        async with httpx.AsyncClient() as c:
            await preflight(c, args.base_url)
        log.info("ready to simulate")
        return 0

    rng = random.Random(RNG_SEED)

    async with httpx.AsyncClient() as c:
        health = await preflight(c, args.base_url)
    del health

    # Ad ids come from a probe round so quality is assigned to ads that exist.
    async with httpx.AsyncClient() as c:
        probe = []
        for _ in range(30):
            r = await one_request(c, args.base_url, key,
                                  World([], random.Random(1)), rng,
                                  {"filled": 0, "no_fill": 0, "errors": 0,
                                   "clicks": 0, "click_403": 0,
                                   "rate_limited": 0})
            if r:
                probe.append(r["ad_id"])
    world = World(sorted(set(probe)) or ["unknown"], random.Random(RNG_SEED))
    log.info("assigned latent quality to %d ads", len(world.ad_quality))

    rows = await run_traffic(args.impressions, args.concurrency,
                             args.base_url, key, world, rng)

    if args.no_backdate:
        log.info("--no-backdate: %d rows went through the gateway only. Wait 2h "
                 "(LABEL_CUTOFF_HOURS) before training.", len(rows))
        return 0

    n_imp, n_click = insert_backdated(rows, args.spread_days,
                                      args.clickhouse_host, args.clickhouse_port)
    ctr = 100.0 * n_click / max(n_imp, 1)
    log.info("inserted %d backdated impressions, %d clicks (%.2f%% CTR)",
             n_imp, n_click, ctr)

    if n_click < 100:
        log.warning("only %d positives — training may fail its gates. Consider "
                    "--impressions %d", n_click, args.impressions * 3)

    print()
    print("  Now train:")
    print("      docker compose run --rm bootstrap python -m adplatform.ml.train_ctr \\")
    print("        --days 30 --out /app/models \\")
    print("        --dsn clickhouse://default@clickhouse:8123/default")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
