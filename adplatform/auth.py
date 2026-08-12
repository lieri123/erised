# auth.py — publisher API key authentication.

from __future__ import annotations

import asyncio
import hmac
import logging
import secrets
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Optional

from fastapi import Header, HTTPException, Request

from .settings import settings

log = logging.getLogger("auth")

KEY_PREFIX_LIVE = "pk_live_"
KEY_PREFIX_TEST = "pk_test_"

PREFIX_STORE_LEN = 16

# Key generation and hashing

def generate_api_key(test: bool = False) -> str:
    """
    Mint a new key. Returned in plaintext exactly once, at creation; after that
    only the hash exists anywhere and the key is unrecoverable.
    """
    prefix = KEY_PREFIX_TEST if test else KEY_PREFIX_LIVE
    return prefix + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    return hmac.new(
        settings.api_key_pepper.encode(),
        api_key.encode(),
        sha256,
    ).hexdigest()


def key_prefix(api_key: str) -> str:
    """The identifiable, safe-to-log fragment stored alongside the hash."""
    return api_key[:PREFIX_STORE_LEN]


# The in-memory cache


@dataclass(frozen=True)
class Principal:

    owner_id: str
    key_id: str
    key_prefix: str
    owner_type: str = "publisher"
    domain: Optional[str] = None

    @property
    def publisher_id(self) -> str:
        return self.owner_id

    @property
    def advertiser_id(self) -> str:
        return self.owner_id

    def __str__(self) -> str:
        return f"{self.owner_type}:{self.owner_id}({self.key_prefix}…)"


Publisher = Principal

_keys: dict[str, Principal] = {}

_loaded_at: Optional[float] = None


def _bootstrap_keys() -> dict[str, Publisher]:
    raw = settings.bootstrap_api_key
    if not raw:
        return {}
    if settings.is_production:
        log.error("BOOTSTRAP_API_KEY is set but ignored in production")
        return {}
    log.warning("bootstrap API key active (development only) — publisher=%s",
                settings.bootstrap_publisher_id)
    return {
        hash_api_key(raw): Principal(
            owner_id=settings.bootstrap_publisher_id,
            key_id="bootstrap",
            key_prefix=key_prefix(raw),
            owner_type="publisher",
        )
    }


async def load_keys_from_db(pool) -> int:
    """
    Pull every live key into memory. Returns the number loaded.
    """
    global _keys, _loaded_at

    try:
        rows = await pool.fetch(
            """
            SELECT k.key_id, k.key_hash, k.key_prefix, k.owner_type, k.owner_id,
                   p.domain
              FROM api_keys k
         LEFT JOIN publishers  p ON k.owner_type = 'publisher'
                                AND p.publisher_id = k.owner_id
         LEFT JOIN advertisers d ON k.owner_type = 'advertiser'
                                AND d.advertiser_id = k.owner_id
             WHERE k.active = TRUE
               AND k.revoked_at IS NULL
               AND COALESCE(p.active, d.active, FALSE) = TRUE
            """
        )
    except Exception:
        log.exception("API key refresh failed; keeping %d cached keys", len(_keys))
        return 0

    fresh = {
        row["key_hash"]: Principal(
            owner_id=row["owner_id"],
            key_id=row["key_id"],
            key_prefix=row["key_prefix"],
            owner_type=row["owner_type"],
            domain=row["domain"],
        )
        for row in rows
    }
    fresh.update(_bootstrap_keys())

    _keys = fresh
    _loaded_at = time.time()
    log.info("API keys loaded: %d active", len(rows))
    return len(rows)


async def refresh_keys_loop(pool, interval: Optional[int] = None) -> None:
    """
    Background refresh. Same shape as cors.refresh_origins_loop — started in the
    lifespan, cancelled on shutdown.
    """
    interval = interval or settings.api_key_refresh_seconds
    log.info("API key refresh loop started — interval: %ds", interval)

    while True:
        await asyncio.sleep(interval)
        try:
            await load_keys_from_db(pool)
        except asyncio.CancelledError:
            log.info("API key refresh loop cancelled")
            raise
        except Exception:
            log.exception("API key refresh loop error")


def add_key_now(api_key: str, principal: Principal) -> None:
    """
    Activate a freshly created key without waiting for the next refresh, so a
    publisher who just signed up can make their first call immediately.
    """
    _keys[hash_api_key(api_key)] = principal
    log.info("API key activated immediately: %s", principal)


def revoke_key_now(key_hash: str) -> bool:
    """Drop a key from the live cache. Call alongside the DB update, not instead of it."""
    removed = _keys.pop(key_hash, None)
    if removed:
        log.warning("API key revoked immediately: %s", removed)
    return removed is not None


def cache_status() -> dict:
    """For /health. Never exposes hashes."""
    return {
        "loaded": _loaded_at is not None,
        "active_keys": len(_keys),
        "age_seconds": round(time.time() - _loaded_at, 1) if _loaded_at else None,
    }

_last_used_written: dict[str, float] = {}


def should_write_last_used(key_id: str) -> bool:
    now = time.time()
    previous = _last_used_written.get(key_id, 0.0)
    if now - previous < settings.last_used_write_interval:
        return False
    _last_used_written[key_id] = now
    return True

# The FastAPI dependency

async def _authenticate(
    request: Request,
    x_api_key: Optional[str],
    expected_type: str,
) -> Principal:
    if _loaded_at is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication temporarily unavailable",
            headers={"Retry-After": "5"},
        )

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    principal = _keys.get(hash_api_key(x_api_key))
    if principal is None:
        log.warning("rejected API key %s… from %s",
                    x_api_key[:8], request.client.host if request.client else "?")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if principal.owner_type != expected_type:
        log.warning("wrong key type: %s called a %s endpoint", principal, expected_type)
        raise HTTPException(
            status_code=403,
            detail=f"This endpoint requires an {expected_type} API key",
        )

    request.state.principal = principal
    return principal


async def require_publisher(
    request: Request,
    x_api_key: Optional[str] = Header(None),
) -> Principal:
    """Publisher-side endpoints: /v1/bid, /v1/conversion, /v1/stats."""
    return await _authenticate(request, x_api_key, "publisher")


async def require_advertiser(
    request: Request,
    x_api_key: Optional[str] = Header(None),
) -> Principal:
    """Advertiser-side endpoints: campaign and ad management, spend reporting."""
    return await _authenticate(request, x_api_key, "advertiser")


async def require_admin(
    x_admin_token: Optional[str] = Header(None),
) -> None:
    expected = settings.admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="Admin API not configured")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token")