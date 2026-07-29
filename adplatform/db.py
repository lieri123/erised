# db.py — Postgres persistence via asyncpg.
#
# Division of labour between the three stores:
#   Postgres    — one row by key, transactional. Powers the click lookup.
#   ClickHouse  — scan-and-aggregate over everything. Powers training + reporting.
#   Redis       — hot counters. Powers budget enforcement.
#
# This file is the Postgres half. It deliberately does NOT store the feature
# vector; that goes to ClickHouse via Kafka.

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import asyncpg

from .settings import settings

log = logging.getLogger("db")

# Module-level. gateway.py reads this as `db._pool`, NOT via
# `from db import _pool` — that would bind a snapshot of None at import time.
_pool: Optional[asyncpg.Pool] = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS publishers (
    publisher_id  TEXT PRIMARY KEY,
    domain        TEXT UNIQUE,
    api_key_hash  TEXT,
    active        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS impressions (
    impression_id    TEXT PRIMARY KEY,
    ts               TIMESTAMPTZ NOT NULL DEFAULT now(),

    publisher_id     TEXT NOT NULL,
    placement_id     TEXT NOT NULL,
    user_id          TEXT,
    device_type      TEXT,
    page_url         TEXT,
    page_keywords    JSONB NOT NULL DEFAULT '[]'::jsonb,

    ad_id            TEXT,
    advertiser_id    TEXT,
    -- Read back by /v1/click. Storing it here is what lets the click endpoint
    -- drop its `redirect` query parameter, which was an open redirect.
    destination_url  TEXT,

    win_price        DOUBLE PRECISION,   -- CPM
    cost_usd         DOUBLE PRECISION,   -- this impression = win_price / 1000
    predicted_ctr    DOUBLE PRECISION,

    filled           BOOLEAN NOT NULL DEFAULT FALSE,
    clicked          BOOLEAN NOT NULL DEFAULT FALSE,
    clicked_at       TIMESTAMPTZ,
    converted        BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS impressions_publisher_ts_idx
    ON impressions (publisher_id, ts DESC);
CREATE INDEX IF NOT EXISTS impressions_ad_ts_idx
    ON impressions (ad_id, ts DESC);

CREATE TABLE IF NOT EXISTS conversions (
    id                BIGSERIAL PRIMARY KEY,
    impression_id     TEXT NOT NULL REFERENCES impressions(impression_id),
    ad_id             TEXT,
    advertiser_id     TEXT,
    user_id           TEXT,
    conversion_type   TEXT,
    conversion_value  DOUBLE PRECISION,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversions_impression_idx
    ON conversions (impression_id);
"""

# Seed rows so CORS and the demo keys work on a fresh database.
SEED = """
INSERT INTO publishers (publisher_id, domain, active) VALUES
    ('pub_demo',     'http://localhost:3000',      TRUE),
    ('pub_techblog', 'https://techblog.example',   TRUE),
    ('pub_newssite', 'https://newssite.example',   TRUE)
ON CONFLICT (publisher_id) DO NOTHING;
"""


async def init_db() -> Optional[asyncpg.Pool]:
    """
    Create the pool and ensure the schema exists. Returns the pool, and also
    sets the module-level `_pool`.

    Does NOT raise if Postgres is unreachable — the gateway logs a warning and
    runs with CORS restricted to dev origins. Failing to boot because the
    database is briefly down is worse than booting degraded.
    """
    global _pool
    try:
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=settings.pg_min_size,
            max_size=settings.pg_max_size,
            command_timeout=settings.pg_command_timeout,
        )
        async with _pool.acquire() as conn:
            await conn.execute(SCHEMA)
            await conn.execute(SEED)
        log.info("Postgres connected, schema ready")
        return _pool
    except Exception as e:
        log.error("Postgres unavailable (%s) — running without persistence", e)
        _pool = None
        return None


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Impressions
# ---------------------------------------------------------------------------

async def save_impression(record: dict[str, Any]) -> None:
    """
    Insert one impression. Called via spawn() so it never blocks the bid
    response.

    ON CONFLICT DO NOTHING because retries and duplicate events are cheaper to
    absorb than to prevent, and impression_id is already unique per auction.
    """
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO impressions (
                    impression_id, publisher_id, placement_id, user_id,
                    device_type, page_url, page_keywords,
                    ad_id, advertiser_id, destination_url,
                    win_price, cost_usd, predicted_ctr,
                    filled, clicked, converted
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
                ON CONFLICT (impression_id) DO NOTHING
                """,
                record["impression_id"],
                record["publisher_id"],
                record["placement_id"],
                record.get("user_id"),
                record.get("device_type"),
                record.get("page_url"),
                json.dumps(record.get("page_keywords", [])),
                record.get("ad_id"),
                record.get("advertiser_id"),
                record.get("destination_url"),
                record.get("win_price"),
                record.get("cost_usd"),
                record.get("predicted_ctr"),
                bool(record.get("filled", False)),
                bool(record.get("clicked", False)),
                bool(record.get("converted", False)),
            )
    except Exception:
        log.exception("save_impression failed for %s", record.get("impression_id"))


async def get_impression(impression_id: str) -> Optional[dict[str, Any]]:
    """Fetch one impression by id. Returns None if absent or Postgres is down."""
    if _pool is None:
        return None
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM impressions WHERE impression_id = $1", impression_id
            )
        if row is None:
            return None
        rec = dict(row)
        # asyncpg returns JSONB as a string unless a codec is registered.
        if isinstance(rec.get("page_keywords"), str):
            rec["page_keywords"] = json.loads(rec["page_keywords"])
        return rec
    except Exception:
        log.exception("get_impression failed for %s", impression_id)
        return None


async def mark_clicked(impression_id: str) -> bool:
    """
    Mark an impression clicked. Returns True only if THIS call was the one that
    flipped the flag.

    The `AND clicked = FALSE` in the WHERE clause plus RETURNING makes the
    check-and-set atomic in a single statement. Reading `clicked` first and
    then updating — which is what the gateway used to do — lets two concurrent
    clicks on the same impression both see FALSE and both emit a click event.
    A double-counted click is a corrupted training label, and it inflates the
    measured CTR of whichever ad got double-clicked.
    """
    if _pool is None:
        return False
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE impressions
                   SET clicked = TRUE, clicked_at = now()
                 WHERE impression_id = $1 AND clicked = FALSE
             RETURNING impression_id
                """,
                impression_id,
            )
        return row is not None
    except Exception:
        log.exception("mark_clicked failed for %s", impression_id)
        return False


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

async def save_conversion(event: dict[str, Any]) -> None:
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO conversions (
                        impression_id, ad_id, advertiser_id, user_id,
                        conversion_type, conversion_value
                    ) VALUES ($1,$2,$3,$4,$5,$6)
                    """,
                    event["impression_id"],
                    event.get("ad_id"),
                    event.get("advertiser_id"),
                    event.get("user_id"),
                    event.get("conversion_type"),
                    event.get("conversion_value"),
                )
                await conn.execute(
                    "UPDATE impressions SET converted = TRUE WHERE impression_id = $1",
                    event["impression_id"],
                )
    except Exception:
        log.exception("save_conversion failed for %s", event.get("impression_id"))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

async def get_publisher_stats(publisher_id: str) -> dict[str, Any]:
    """
    Last 30 days for one publisher.

    Postgres is the wrong engine for this once you have real volume — it is a
    full scan of every impression row. Move it to ClickHouse when the query
    starts taking seconds. Fine for now.

    revenue_usd sums cost_usd, not win_price. win_price is a CPM; summing it
    would report roughly 1000x actual revenue.
    """
    empty = {
        "publisher_id": publisher_id, "impressions": 0, "clicks": 0,
        "conversions": 0, "ctr": 0.0, "revenue_usd": 0.0,
    }
    if _pool is None:
        return empty
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    count(*) FILTER (WHERE filled)              AS impressions,
                    count(*) FILTER (WHERE clicked)             AS clicks,
                    count(*) FILTER (WHERE converted)           AS conversions,
                    COALESCE(sum(cost_usd), 0)                  AS revenue_usd
                FROM impressions
                WHERE publisher_id = $1
                  AND ts > now() - INTERVAL '30 days'
                """,
                publisher_id,
            )
        impressions = int(row["impressions"] or 0)
        clicks = int(row["clicks"] or 0)
        return {
            "publisher_id": publisher_id,
            "impressions":  impressions,
            "clicks":       clicks,
            "conversions":  int(row["conversions"] or 0),
            "ctr":          round(clicks / impressions, 6) if impressions else 0.0,
            "revenue_usd":  round(float(row["revenue_usd"] or 0.0), 6),
        }
    except Exception:
        log.exception("get_publisher_stats failed for %s", publisher_id)
        return empty