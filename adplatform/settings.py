# settings.py — every piece of environment-dependent configuration, in one place.
from __future__ import annotations

import os
import tempfile
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
    """
    Comma-separated list from the environment.
    """
    raw = os.getenv(key)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: _str("ENV", "development"))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO"))

    public_base_url: str = field(
        default_factory=lambda: _str("PUBLIC_BASE_URL", "http://localhost:8000")
    )

    click_url_ttl_seconds: int = field(
        default_factory=lambda: _int("CLICK_URL_TTL_SECONDS", 60 * 60 * 24)
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
    # PLAINTEXT is correct on a compose network and wrong on anything routable.
    # Redpanda on EC2 inside the VPC can stay PLAINTEXT; MSK cannot.
    kafka_security_protocol: str = field(
        default_factory=lambda: _str("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").upper()
    )
    kafka_sasl_mechanism: str = field(
        default_factory=lambda: _str("KAFKA_SASL_MECHANISM", "PLAIN").upper()
    )
    kafka_sasl_username: str = field(
        default_factory=lambda: _str("KAFKA_SASL_USERNAME", "")
    )
    kafka_sasl_password: str = field(
        default_factory=lambda: _str("KAFKA_SASL_PASSWORD", "")
    )
    # Empty means "use the system trust store", which is right for MSK public
    # endpoints and wrong for a self-signed Redpanda cert.
    kafka_ssl_cafile: str = field(default_factory=lambda: _str("KAFKA_SSL_CAFILE", ""))

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

    # -- auth / admin ---------------------------------------------------------
   
    api_key_pepper: str = field(
        default_factory=lambda: _str("API_KEY_PEPPER", "dev-only-insecure-pepper")
    )
    api_key_refresh_seconds: int = field(
        default_factory=lambda: _int("API_KEY_REFRESH_SECONDS", 60)
    )
    last_used_write_interval: int = field(
        default_factory=lambda: _int("LAST_USED_WRITE_INTERVAL", 300)
    )
    admin_token: str = field(default_factory=lambda: _str("ADMIN_TOKEN", ""))

    bootstrap_api_key: str = field(default_factory=lambda: _str("BOOTSTRAP_API_KEY", ""))
    bootstrap_publisher_id: str = field(
        default_factory=lambda: _str("BOOTSTRAP_PUBLISHER_ID", "pub_demo")
    )

    # -- inventory ------------------------------------------------------------
    inventory_refresh_seconds: int = field(
        default_factory=lambda: _int("INVENTORY_REFRESH_SECONDS", 60)
    )

    # -- model artifacts ----------------------------------------------------
    # "local" reads a directory (a bind mount in compose). "s3" reads a
    # current.json pointer and downloads the version it names. See
    # adplatform/ml/artifacts.py for why the pointer exists.
    model_artifact_backend: str = field(
        default_factory=lambda: _str("MODEL_ARTIFACT_BACKEND", "local")
    )
    model_dir: str = field(default_factory=lambda: _str("MODEL_DIR", "models/current"))
    model_s3_bucket: str = field(default_factory=lambda: _str("MODEL_S3_BUCKET", ""))
    model_s3_prefix: str = field(
        default_factory=lambda: _str("MODEL_S3_PREFIX", "models")
    )
    # Task-local scratch. Fargate gives each task an ephemeral volume; a fresh
    # task downloads once at boot and then only on promotion.
    model_cache_dir: str = field(
        default_factory=lambda: _str(
            "MODEL_CACHE_DIR", str(Path(tempfile.gettempdir()) / "erised-models")
        )
    )
    aws_region: str = field(
        default_factory=lambda: _str("AWS_REGION", _str("AWS_DEFAULT_REGION", "us-east-1"))
    )

    # -- stats refresh ------------------------------------------------------
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

    def validate_for_production(self) -> list[str]:
        """
        Config problems that are silent at runtime but should be fatal at boot.
        Called once, in the lifespan, before anything else starts. Returns an
        empty list outside production — a dev box is allowed to run on defaults.
        """
        if not self.is_production:
            return []

        problems = []
        if self.api_key_pepper == "dev-only-insecure-pepper" or not self.api_key_pepper:
            problems.append(
                "API_KEY_PEPPER is unset or still the insecure default — "
                "every stored api_key hash is forgeable by anyone who read this repo"
            )
        if self.bootstrap_api_key:
            problems.append(
                "BOOTSTRAP_API_KEY is set in production — remove it, it is a "
                "standing credential outside the normal key table"
            )
        if not self.admin_token:
            problems.append(
                "ADMIN_TOKEN is unset — /admin/* will 503 for everyone, "
                "including you"
            )
        if not self.clickhouse_password:
            problems.append(
                "CLICKHOUSE_PASSWORD is empty — port 8123 is unauthenticated"
            )
        if self.model_artifact_backend.lower() == "local":
            problems.append(
                "MODEL_ARTIFACT_BACKEND=local in production — with more than one "
                "replica each task loads whatever its own filesystem happens to "
                "hold, and they will silently serve different models"
            )
        if self.model_artifact_backend.lower() == "s3" and not self.model_s3_bucket:
            problems.append("MODEL_ARTIFACT_BACKEND=s3 but MODEL_S3_BUCKET is empty")
        if self.kafka_security_protocol.startswith("SASL") and not self.kafka_sasl_username:
            problems.append(
                f"KAFKA_SECURITY_PROTOCOL={self.kafka_security_protocol} but "
                "KAFKA_SASL_USERNAME is empty — the producer will fail to connect "
                "and events will be dropped silently"
            )
        return problems

    def describe(self) -> str:
        """One-line summary for the startup log. Never logs secrets."""
        return (
            f"env={self.env} pg={_redact(self.database_url)} "
            f"kafka={self.kafka_bootstrap}/{self.kafka_security_protocol} "
            f"model={self.model_artifact_backend} "
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


settings = get_settings()