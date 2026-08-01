#!/usr/bin/env python3
"""
bootstrap.py — bring a fresh stack to the point where the gateway can serve.

Runs as a one-shot container in docker compose, before the gateway starts.
Idempotent: safe to run on every `docker compose up`.

Order is not arbitrary:

  1. Kafka topics FIRST. ClickHouse's Kafka-engine tables start consuming the
     instant they are created. Pointing one at a topic that does not exist yet
     works (Redpanda auto-creates) but you get an auto-created topic with
     default settings instead of the partition count you wanted, and it is
     invisible until you wonder why throughput is flat.

  2. Postgres schema. The gateway applies db.SCHEMA itself on startup, but
     bootstrap runs first and the advertiser-side migration (002) needs the base
     tables to reference. Applying SCHEMA here from the same constant the
     gateway uses means there is one definition, not two that drift.

  3. ClickHouse schema.sql, THEN kafka_sink.sql. The materialized views in
     kafka_sink.sql have `TO ad_impressions` / `TO ad_clicks` targets. Creating
     a MV whose destination does not exist fails outright, so the order matters.

Everything waits for its dependency with a bounded retry rather than trusting
compose healthchecks alone — a container being "healthy" and a database being
ready to accept a CREATE TABLE are not the same moment.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adplatform.db import SCHEMA as PG_BASE_SCHEMA  # noqa: E402
from adplatform.settings import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] bootstrap - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bootstrap")

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

# num_partitions on `impressions` is the one number here worth thinking about.
# ClickHouse consumes with kafka_num_consumers=1, so extra partitions buy you
# nothing today — but partitions cannot be reduced later, only added, and
# repartitioning a live topic means resetting consumer offsets. Three is cheap
# insurance for a single-node dev stack that might become three nodes.
TOPICS = {
    "impressions": 3,
    "clicks": 3,
    "conversions": 1,
}

RETRY_ATTEMPTS = 30
RETRY_DELAY = 2.0


async def _retry(what: str, fn):
    """Call an async fn until it stops raising, or give up loudly."""
    last: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt == 1 or attempt % 5 == 0:
                log.info("waiting for %s (attempt %d/%d): %s",
                         what, attempt, RETRY_ATTEMPTS, e)
            await asyncio.sleep(RETRY_DELAY)
    raise RuntimeError(f"{what} never became available: {last}")


# ---------------------------------------------------------------------------
# SQL splitting
# ---------------------------------------------------------------------------

def split_sql(text: str) -> list[str]:
    """
    Split a .sql file into individual statements.

    ClickHouse's HTTP interface takes one statement per request, so schema.sql
    has to be split. Comments must be stripped BEFORE splitting on `;` —
    schema.sql and kafka_sink.sql both contain commented-out example queries
    that end in a semicolon, and splitting first would emit a statement that is
    nothing but a comment fragment.

    Deliberately simple: strips `--` line comments and `/* */` blocks, then
    splits on `;`. That is correct for these two files and would break on a
    string literal containing a semicolon. If you ever add one, replace this
    with sqlglot rather than making the regex cleverer.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        # Trailing comment on a code line, e.g. "  kafka_num_consumers = 1, -- one"
        line = re.sub(r"--.*$", "", line)
        lines.append(line)
    joined = "\n".join(lines)
    return [s.strip() for s in joined.split(";") if s.strip()]


# ---------------------------------------------------------------------------
# 1. Kafka topics
# ---------------------------------------------------------------------------

async def ensure_topics() -> None:
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic

    async def connect():
        admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap)
        await admin.start()
        return admin

    admin = await _retry(f"Redpanda at {settings.kafka_bootstrap}", connect)
    try:
        existing = set(await admin.list_topics())
        wanted = [
            NewTopic(name=name, num_partitions=parts, replication_factor=1)
            for name, parts in TOPICS.items()
            if name not in existing
        ]
        if not wanted:
            log.info("topics already present: %s", ", ".join(sorted(TOPICS)))
            return
        await admin.create_topics(wanted)
        log.info("created topics: %s", ", ".join(t.name for t in wanted))
    except Exception as e:
        # A concurrent bootstrap (two `compose up` in a row) races here.
        if "already exists" in str(e).lower():
            log.info("topics already exist (race), continuing")
        else:
            raise
    finally:
        await admin.close()


# ---------------------------------------------------------------------------
# 2. Postgres
# ---------------------------------------------------------------------------

async def apply_postgres() -> None:
    import asyncpg

    async def connect():
        return await asyncpg.connect(settings.database_url, timeout=5)

    conn = await _retry("Postgres", connect)
    try:
        # asyncpg executes a multi-statement string in one call as long as there
        # are no bound parameters, so these files need no splitting.
        await conn.execute(PG_BASE_SCHEMA)
        log.info("applied base schema (adplatform.db.SCHEMA)")

        migration = SQL_DIR / "002_advertiser_side.sql"
        if migration.exists():
            await conn.execute(migration.read_text())
            log.info("applied %s", migration.name)
        else:
            log.warning("%s not found — advertiser tables will be missing",
                        migration.name)

        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables
             WHERE schemaname = 'public' ORDER BY tablename
        """)
        log.info("postgres tables: %s",
                 ", ".join(r["tablename"] for r in tables) or "(none)")
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# 3. ClickHouse
# ---------------------------------------------------------------------------

def _ch_client():
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        database=settings.clickhouse_database,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
    )


async def apply_clickhouse() -> None:
    async def connect():
        return await asyncio.to_thread(_ch_client)

    client = await _retry(
        f"ClickHouse at {settings.clickhouse_host}:{settings.clickhouse_port}",
        connect,
    )

    # Order matters. schema.sql creates the destination tables, kafka_sink.sql
    # creates the consumers and MVs that write into them, and numbered
    # migrations then alter both. 003 drops and recreates kafka_impressions, so
    # it MUST run after kafka_sink.sql, not before.
    #
    # Every file here is idempotent (IF EXISTS / IF NOT EXISTS), which is what
    # lets bootstrap re-run on every `docker compose up` without a migrations
    # table. That stops being true the moment someone writes a migration with a
    # bare ALTER -- at which point this list should be replaced with a real
    # migration runner that records what it has applied.
    for filename in ("schema.sql", "kafka_sink.sql", "003_campaign_budgets.sql"):
        path = SQL_DIR / filename
        if not path.exists():
            log.error("%s missing — skipping", filename)
            continue
        statements = split_sql(path.read_text())
        log.info("applying %s (%d statements)", filename, len(statements))
        for i, stmt in enumerate(statements, 1):
            try:
                await asyncio.to_thread(client.command, stmt)
            except Exception as e:
                head = " ".join(stmt.split())[:90]
                log.error("statement %d of %s failed: %s\n  %s",
                          i, filename, e, head)
                raise

    rows = (await asyncio.to_thread(
        client.query, "SELECT name FROM system.tables WHERE database = {db:String} ORDER BY name",
        {"db": settings.clickhouse_database},
    )).result_rows
    log.info("clickhouse objects: %s", ", ".join(r[0] for r in rows) or "(none)")


# ---------------------------------------------------------------------------

async def main() -> int:
    log.info("bootstrap starting — %s", settings.describe())
    try:
        await ensure_topics()
        await apply_postgres()
        await apply_clickhouse()
    except Exception:
        log.exception("BOOTSTRAP FAILED")
        return 1

    log.info("bootstrap complete — topics, postgres and clickhouse are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
