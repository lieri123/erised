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


async def get_advertiser_stats(advertiser_id: str) -> dict[str, Any]:
    """
    Last 30 days for one advertiser. Mirrors get_publisher_stats — same
    caveat about Postgres being the wrong engine once volume is real.
    """
    empty = {
        "advertiser_id": advertiser_id, "impressions": 0, "clicks": 0,
        "conversions": 0, "ctr": 0.0, "spend_usd": 0.0,
    }
    if _pool is None:
        return empty
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    count(*) FILTER (WHERE filled)    AS impressions,
                    count(*) FILTER (WHERE clicked)   AS clicks,
                    count(*) FILTER (WHERE converted) AS conversions,
                    COALESCE(sum(cost_usd), 0)        AS spend_usd
                FROM impressions
                WHERE advertiser_id = $1
                  AND ts > now() - INTERVAL '30 days'
                """,
                advertiser_id,
            )
        impressions = int(row["impressions"] or 0)
        clicks = int(row["clicks"] or 0)
        return {
            "advertiser_id": advertiser_id,
            "impressions":   impressions,
            "clicks":        clicks,
            "conversions":   int(row["conversions"] or 0),
            "ctr":           round(clicks / impressions, 6) if impressions else 0.0,
            "spend_usd":     round(float(row["spend_usd"] or 0.0), 6),
        }
    except Exception:
        log.exception("get_advertiser_stats failed for %s", advertiser_id)
        return empty


# ---------------------------------------------------------------------------
# Campaigns / ads
# ---------------------------------------------------------------------------

async def create_campaign(fields: dict[str, Any]) -> Optional[str]:
    """Returns the campaign_id on success, None if Postgres is unavailable or the insert fails."""
    if _pool is None:
        return None
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO campaigns (
                    campaign_id, advertiser_id, name, daily_budget_usd,
                    target_cpm, floor_price, target_device, target_keywords,
                    start_date, end_date
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                """,
                fields["campaign_id"],
                fields["advertiser_id"],
                fields["name"],
                fields["daily_budget_usd"],
                fields["target_cpm"],
                fields.get("floor_price", 0.0),
                fields.get("target_device", "all"),
                json.dumps(fields.get("target_keywords", [])),
                fields.get("start_date"),
                fields.get("end_date"),
            )
        return fields["campaign_id"]
    except Exception:
        log.exception("create_campaign failed for %s", fields.get("campaign_id"))
        return None


async def list_campaigns(advertiser_id: str) -> list[dict[str, Any]]:
    if _pool is None:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT campaign_id, name, daily_budget_usd, target_cpm,
                       floor_price, target_device, target_keywords,
                       start_date, end_date, status, created_at
                  FROM campaigns
                 WHERE advertiser_id = $1
                 ORDER BY created_at DESC
                """,
                advertiser_id,
            )
        out = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("target_keywords"), str):
                d["target_keywords"] = json.loads(d["target_keywords"])
            out.append(d)
        return out
    except Exception:
        log.exception("list_campaigns failed for %s", advertiser_id)
        return []


async def create_ad(fields: dict[str, Any]) -> Optional[str]:
    """Returns the ad_id on success, None otherwise. Caller (gateway) already
    verified the advertiser owns campaign_id before calling this."""
    if _pool is None:
        return None
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ads (ad_id, campaign_id, name, creative_html, destination_url)
                VALUES ($1,$2,$3,$4,$5)
                """,
                fields["ad_id"],
                fields["campaign_id"],
                fields.get("name"),
                fields["creative_html"],
                fields["destination_url"],
            )
        return fields["ad_id"]
    except Exception:
        log.exception("create_ad failed for %s", fields.get("ad_id"))
        return None


async def set_status(
    table: str, id_col: str, id_val: str, status: str, advertiser_id: str
) -> bool:
    """
    Shared by /v1/campaigns/{id}/status and /v1/ads/{id}/status.

    `table` and `id_col` come from the call sites in gateway.py, never from the
    request, so this is not a SQL-injection surface despite the f-string.

    Ownership is checked in the same statement rather than as a prior SELECT:
    for `ads`, ownership is via the parent campaign, so the WHERE clause
    differs per table and doing it as one UPDATE...RETURNING avoids a
    check-then-act race with a concurrent status change.
    """
    if _pool is None:
        return False
    try:
        async with _pool.acquire() as conn:
            if table == "campaigns":
                row = await conn.fetchrow(
                    f"""
                    UPDATE campaigns SET status = $1
                     WHERE {id_col} = $2 AND advertiser_id = $3
                 RETURNING {id_col}
                    """,
                    status, id_val, advertiser_id,
                )
            elif table == "ads":
                row = await conn.fetchrow(
                    f"""
                    UPDATE ads SET status = $1
                     WHERE {id_col} = $2
                       AND campaign_id IN (
                           SELECT campaign_id FROM campaigns WHERE advertiser_id = $3
                       )
                 RETURNING {id_col}
                    """,
                    status, id_val, advertiser_id,
                )
            else:
                raise ValueError(f"set_status: unknown table {table!r}")
        return row is not None
    except Exception:
        log.exception("set_status(%s, %s=%s) failed", table, id_col, id_val)
        return False


# ---------------------------------------------------------------------------
# Publishers / advertisers / api keys (admin provisioning)
# ---------------------------------------------------------------------------

async def create_publisher(publisher_id: str, domain: str, name: Optional[str]) -> bool:
    if _pool is None:
        return False
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO publishers (publisher_id, domain, active)
                VALUES ($1, $2, TRUE)
                ON CONFLICT (publisher_id) DO NOTHING
                RETURNING publisher_id
                """,
                publisher_id, domain,
            )
        return row is not None
    except Exception:
        log.exception("create_publisher failed for %s", publisher_id)
        return False


async def create_advertiser(
    advertiser_id: str, name: str, contact_email: Optional[str]
) -> bool:
    if _pool is None:
        return False
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO advertisers (advertiser_id, name, contact_email, active)
                VALUES ($1, $2, $3, TRUE)
                ON CONFLICT (advertiser_id) DO NOTHING
                RETURNING advertiser_id
                """,
                advertiser_id, name, contact_email,
            )
        return row is not None
    except Exception:
        log.exception("create_advertiser failed for %s", advertiser_id)
        return False


async def insert_api_key(
    key_id: str,
    publisher_id: Optional[str],
    key_hash: str,
    key_prefix: str,
    name: Optional[str] = None,
    owner_type: str = "publisher",
    owner_id: Optional[str] = None,
) -> bool:
    """
    publisher_id is the historical positional argument (owner_type="publisher"
    call sites pass the publisher id there). For owner_type="advertiser",
    callers pass publisher_id=None and owner_id=<advertiser_id> explicitly.
    """
    resolved_owner_id = owner_id if owner_type == "advertiser" else publisher_id
    if not resolved_owner_id:
        log.error("insert_api_key: no owner id resolved (owner_type=%s)", owner_type)
        return False
    if _pool is None:
        return False
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO api_keys (key_id, owner_type, owner_id, key_hash, key_prefix, name)
                VALUES ($1,$2,$3,$4,$5,$6)
                """,
                key_id, owner_type, resolved_owner_id, key_hash, key_prefix, name,
            )
        return True
    except Exception:
        log.exception("insert_api_key failed for %s", key_id)
        return False


async def list_api_keys(publisher_id: str) -> list[dict[str, Any]]:
    """Metadata only — key_hash is never returned."""
    if _pool is None:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT key_id, key_prefix, name, active, revoked_at,
                       last_used_at, created_at
                  FROM api_keys
                 WHERE owner_type = 'publisher' AND owner_id = $1
                 ORDER BY created_at DESC
                """,
                publisher_id,
            )
        return [dict(r) for r in rows]
    except Exception:
        log.exception("list_api_keys failed for %s", publisher_id)
        return []


async def touch_api_key(key_id: str) -> None:
    """
    Stamp last_used_at. Called from gateway.note_key_usage via spawn(), already
    throttled by auth.should_write_last_used to one write per key per
    last_used_write_interval (default 300s).

    That throttle is why this can be a plain UPDATE with no upsert or contention
    handling: at most one write per key per five minutes, and a lost race just
    means the timestamp is a few minutes stale. last_used_at is an operational
    breadcrumb for "is this key still in use before I revoke it", not an audit
    log — do not build billing or security decisions on it.

    Swallows its own exceptions like every other spawn() target here. A failed
    timestamp write must never surface as a failed bid.

    NOTE: gateway.py:307 called this function before it existed, which raised
    AttributeError while evaluating the argument to spawn() — i.e. on the
    caller's stack, before the task was ever created, so the fire-and-forget
    error isolation did not apply. Every authenticated endpoint returned 500;
    only /health worked, because it does not authenticate.
    """
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                "UPDATE api_keys SET last_used_at = now() WHERE key_id = $1",
                key_id,
            )
    except Exception:
        log.exception("touch_api_key failed for %s", key_id)


async def revoke_api_key(key_id: str) -> Optional[str]:
    """Marks the key revoked and returns its key_hash so the caller can evict
    it from the in-memory auth cache immediately. None if unknown/already revoked."""
    if _pool is None:
        return None
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE api_keys
                   SET active = FALSE, revoked_at = now()
                 WHERE key_id = $1 AND revoked_at IS NULL
             RETURNING key_hash
                """,
                key_id,
            )
        return row["key_hash"] if row else None
    except Exception:
        log.exception("revoke_api_key failed for %s", key_id)
        return None