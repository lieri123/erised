# settings.py — every piece of environment-dependent configuration, in one place.
#
# Why a module and not os.getenv() scattered around:
#   The same code has to run in three environments — your laptop (localhost),
#   docker compose (service hostnames), and the Hetzner box (real hosts). Each
#   hardcoded "localhost:8123" is a place that silently connects to nothing in
#   two of those three.
#
# Read once at import, frozen afterwards. Nothing here should ever be mutated at
# runtime; if you need a value to change while running, it belongs in the
# database with a refresh loop, not here.
#
# Precedence: real environment variable > .env file > default below.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader. Deliberately does not overwrite variables that are
    already set — in compose and systemd the real environment wins, and a stale
    .env silently overriding it is a miserable afternoon.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


def _str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from None


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(key)
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # -- service identity ---------------------------------------------------
    env: str = field(default_factory=lambda: _str("ENV", "development"))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO"))

    # The base URL publishers' browsers can reach. This ends up inside every
    # click URL embedded in a creative, so getting it wrong means every click
    # in production points at localhost.
    public_base_url: str = field(
        default_factory=lambda: _str("PUBLIC_BASE_URL", "http://localhost:8000")
    )

    # -- Postgres -----------------------------------------------------------
    database_url: str = field(
        default_factory=lambda: _str(
            "DATABASE_URL",
            "postgresql://adplatform:adplatform@localhost:5432/adplatform",
        )
    )
    pg_min_size: int = field(default_factory=lambda: _int("PG_MIN_SIZE", 2))
    pg_max_size: int = field(default_factory=lambda: _int("PG_MAX_SIZE", 10))
    pg_command_timeout: int = field(
        default_factory=lambda: _int("PG_COMMAND_TIMEOUT", 5)
    )

    # -- Kafka / Redpanda ---------------------------------------------------
    kafka_bootstrap: str = field(
        default_factory=lambda: _str("KAFKA_BOOTSTRAP", "localhost:19092")
    )
    kafka_linger_ms: int = field(default_factory=lambda: _int("KAFKA_LINGER_MS", 5))
    kafka_request_timeout_ms: int = field(
        default_factory=lambda: _int("KAFKA_REQUEST_TIMEOUT_MS", 5000)
    )

    # -- ClickHouse ---------------------------------------------------------
    clickhouse_host: str = field(
        default_factory=lambda: _str("CLICKHOUSE_HOST", "localhost")
    )
    clickhouse_port: int = field(default_factory=lambda: _int("CLICKHOUSE_PORT", 8123))
    clickhouse_database: str = field(
        default_factory=lambda: _str("CLICKHOUSE_DATABASE", "default")
    )
    clickhouse_user: str = field(
        default_factory=lambda: _str("CLICKHOUSE_USER", "default")
    )
    clickhouse_password: str = field(
        default_factory=lambda: _str("CLICKHOUSE_PASSWORD", "")
    )

    # -- Redis --------------------------------------------------------------
    redis_url: str = field(
        default_factory=lambda: _str("REDIS_URL", "redis://localhost:6379")
    )

    # -- CORS ---------------------------------------------------------------
    cors_refresh_interval: int = field(
        default_factory=lambda: _int("CORS_REFRESH_INTERVAL", 300)
    )
    # Dev origins are configurable so production can ship with an empty list
    # rather than permanently trusting localhost.
    dev_origins: tuple[str, ...] = field(
        default_factory=lambda: _csv(
            "DEV_ORIGINS",
            (
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
            ),
        )
    )

    # -- model + stats refresh ----------------------------------------------
    model_dir: str = field(default_factory=lambda: _str("MODEL_DIR", "models/current"))
    model_refresh_seconds: int = field(
        default_factory=lambda: _int("MODEL_REFRESH_SECONDS", 60)
    )
    stats_refresh_seconds: int = field(
        default_factory=lambda: _int("STATS_REFRESH_SECONDS", 300)
    )
    stats_lookback_days: int = field(
        default_factory=lambda: _int("STATS_LOOKBACK_DAYS", 30)
    )

    # -- auction ------------------------------------------------------------
    # Not zero. See the note in ml/rtb_integration.py — this is what keeps the
    # training data from being a closed loop on the model's own past opinions.
    exploration_epsilon: float = field(
        default_factory=lambda: _float("EXPLORATION_EPSILON", 0.08)
    )

    # -- budgets ------------------------------------------------------------
    budget_key_ttl_seconds: int = field(
        default_factory=lambda: _int("BUDGET_KEY_TTL_SECONDS", 60 * 60 * 48)
    )
    budget_reconcile_seconds: int = field(
        default_factory=lambda: _int("BUDGET_RECONCILE_SECONDS", 300)
    )

    # -- rate limits --------------------------------------------------------
    bid_rate_limit: str = field(
        default_factory=lambda: _str("BID_RATE_LIMIT", "120/minute")
    )
    stats_rate_limit: str = field(
        default_factory=lambda: _str("STATS_RATE_LIMIT", "30/minute")
    )

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    def describe(self) -> str:
        """One-line summary for the startup log. Never logs secrets."""
        return (
            f"env={self.env} pg={_redact(self.database_url)} "
            f"kafka={self.kafka_bootstrap} "
            f"clickhouse={self.clickhouse_host}:{self.clickhouse_port} "
            f"redis={_redact(self.redis_url)} base_url={self.public_base_url}"
        )


def _redact(url: str) -> str:
    """Strip credentials out of a connection URL before it reaches a log line."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level convenience. Import this, not the individual values —
# `from .settings import settings` keeps one object; `from .settings import
# database_url` would snapshot a string and reintroduce exactly the stale-binding
# problem that bit db._pool.
settings = get_settings()