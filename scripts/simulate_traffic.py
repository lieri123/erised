# simulate_traffic.py — generate a labelled dataset the CTR model can learn from.

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
from pathlib import Path

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

DEVICE_EFFECT = {"mobile": 1.35, "desktop": 0.75, "tablet": 1.0}

BASE_CTR = 0.02


# The latent world

class World:
    """
    Hidden ground truth. The gateway never sees this; it is what CLICKS are
    drawn from, and what the model is being asked to rediscover from features.
    """

    def __init__(self, ad_ids: list[str], rng: random.Random):
        self.rng = rng
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


# Readiness

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


# Latency

def new_stats() -> dict:
    """One place that knows the shape, so the probe loop and the real run cannot
    drift apart. They did: adding latency_ms to one and not the other is a
    KeyError 30 requests into a run that already spun up the whole stack."""
    return {"filled": 0, "no_fill": 0, "errors": 0, "clicks": 0,
            "click_403": 0, "rate_limited": 0, "latency_ms": []}


def percentile(ordered: list[float], q: float) -> float:
    """Nearest-rank, on an already-sorted list. No numpy in the driver, and at
    20k samples the difference from an interpolating percentile is well under
    the run-to-run variance."""
    if not ordered:
        return float("nan")
    k = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[k - 1]


def latency_summary(samples: list[float], wall_seconds: float,
                    concurrency: int) -> dict:
    ordered = sorted(samples)
    n = len(ordered)
    if not n:
        return {}
    return {
        "responses": n,
        "concurrency": concurrency,
        "wall_seconds": round(wall_seconds, 2),
        "throughput_rps": round(n / max(wall_seconds, 1e-3), 1),
        "p50_ms": round(percentile(ordered, 50), 2),
        "p95_ms": round(percentile(ordered, 95), 2),
        "p99_ms": round(percentile(ordered, 99), 2),
        "max_ms": round(ordered[-1], 2),
        "mean_ms": round(sum(ordered) / n, 2),
    }


def log_latency(summary: dict) -> None:
    if not summary:
        log.warning("no latency samples — every request failed in transport?")
        return
    log.info("")
    log.info("/v1/bid latency, %d responses at concurrency %d",
             summary["responses"], summary["concurrency"])
    for label in ("p50", "p95", "p99", "max", "mean"):
        log.info("  %-5s %8.2f ms", label, summary[f"{label}_ms"])
    log.info("  %.0f req/s sustained over %.1fs",
             summary["throughput_rps"], summary["wall_seconds"])
    log.info("")


# Traffic

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

    # Timed around the bid POST only. The click GET below is a different
    # endpoint doing different work, and folding the two together produces a
    # number that describes neither.
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{base}/v1/bid", json=body,
                              headers={"X-API-Key": key}, timeout=15)
    except Exception:
        # No response, so no latency worth recording — a timeout would enter as
        # a suspiciously round 15000ms and sit on top of p99.
        stats["errors"] += 1
        return None
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # 429s are rejected by the rate limiter before an auction runs. They are
    # real responses but they are not the work being measured, and at any
    # meaningful volume they would drag p50 toward slowapi's overhead.
    if r.status_code != 429:
        stats["latency_ms"].append(elapsed_ms)

    if r.status_code == 429:
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

    overlap = len(set(keywords) & set(KEYWORD_POOL[:12]))
    p = world.true_ctr(bid["ad_id"], user_id, device, overlap)
    clicked = rng.random() < p

    if clicked:
        stats["clicks"] += 1
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
                      world: World, rng: random.Random) -> tuple[list[dict], dict]:
    stats = new_stats()
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

    wall = time.time() - started

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

    summary = latency_summary(stats["latency_ms"], wall, concurrency)
    log_latency(summary)
    return results, summary


# Backdated rows for training

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
    ap.add_argument("--latency-out", default=None,
                    help="write the latency summary as JSON, e.g. "
                         "docs/loadtest/run.json")
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

    async with httpx.AsyncClient() as c:
        probe = []
        for _ in range(30):
            r = await one_request(c, args.base_url, key,
                                  World([], random.Random(1)), rng,
                                  new_stats())
            if r:
                probe.append(r["ad_id"])
    world = World(sorted(set(probe)) or ["unknown"], random.Random(RNG_SEED))
    log.info("assigned latent quality to %d ads", len(world.ad_quality))

    rows, latency = await run_traffic(args.impressions, args.concurrency,
                                      args.base_url, key, world, rng)

    if args.latency_out and latency:
        import json
        out = Path(args.latency_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "requested_impressions": args.impressions,
            **latency,
        }, indent=2))
        log.info("latency summary written to %s", out)

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
