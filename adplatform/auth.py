# auth.py — publisher API key authentication.
#
# Replaces the hardcoded VALID_API_KEYS dict in gateway.py.
# ---------------------------------------------------------------------------
# WHY A CACHE
# ---------------------------------------------------------------------------
# Even a fast hash does not save you if you hit Postgres on every bid. One round
# trip is 1-2ms against a 20ms budget, and it couples ad serving to database
# availability — Postgres hiccups and you stop bidding.
#
# So the same pattern as cors.py: load the whole key set into memory at startup,
# refresh in the background, and do an O(1) dict lookup on the hot path. The
# entire working set is one row per publisher; even 100k publishers is a few MB.
#
# ---------------------------------------------------------------------------
# FAIL CLOSED — the opposite of budget.py
# ---------------------------------------------------------------------------
# budget.py fails OPEN on Redis errors: treat spend as zero and keep bidding,
# because a cache outage becoming a revenue outage is worse than a bounded,
# recoverable overspend.
#
# Auth fails CLOSED. The asymmetry is not inconsistency — it is the same
# reasoning applied to a different failure. An overspend is money, visible in
# the impression log, and correctable by reconcile(). Unauthorized access is
# not correctable; you cannot un-serve data to someone who already read it.
#
# Concretely: if the cache has NEVER loaded we return 503, not 401. A 401 tells
# an integrating publisher "your key is wrong" and sends them hunting through
# their config for a problem that is on our side. 503 says "we are broken, retry".
#
# If a refresh FAILS we keep the previous snapshot rather than emptying it —
# same as cors.py. An empty set would lock out every publisher because Postgres
# blipped.

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

# How much of the key is stored in plaintext for identification. Enough to
# answer "which key is this?" in a support ticket or a log line, far too little
# to be useful to anyone who reads it.
PREFIX_STORE_LEN = 16


# ---------------------------------------------------------------------------
# Key generation and hashing
# ---------------------------------------------------------------------------

def generate_api_key(test: bool = False) -> str:
    """
    Mint a new key. Returned in plaintext exactly once, at creation; after that
    only the hash exists anywhere and the key is unrecoverable.

    The visible prefix is deliberate. `pk_live_` lets secret scanners (GitHub's,
    and your own CI) recognise a leaked credential in a commit, and lets you
    tell at a glance whether the key someone pasted into a bug report is a
    production one that needs immediate rotation.
    """
    prefix = KEY_PREFIX_TEST if test else KEY_PREFIX_LIVE
    return prefix + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """
    HMAC-SHA256 of the key under the server pepper. Deterministic, so it can be
    used as a lookup index — which is the whole point. A per-row salt (what
    bcrypt does) would force a linear scan comparing against every stored hash,
    turning an O(1) dict lookup into an O(n) walk on your hot path.

    The pepper is what a deterministic hash gives up and buys back: no per-row
    salt, but a database dump is useless without the application secret.
    """
    return hmac.new(
        settings.api_key_pepper.encode(),
        api_key.encode(),
        sha256,
    ).hexdigest()


def key_prefix(api_key: str) -> str:
    """The identifiable, safe-to-log fragment stored alongside the hash."""
    return api_key[:PREFIX_STORE_LEN]


# ---------------------------------------------------------------------------
# The in-memory cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Principal:
    """
    Who the caller is. Attached to request.state and used for authorisation.

    owner_type is the load-bearing field: publishers call /v1/bid, advertisers
    call /v1/campaigns, and a key for one must not work on the other. Checking
    it in the dependency rather than in each handler means a new endpoint cannot
    forget to check.
    """

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


# The old name, kept so existing imports and annotations keep working.
Publisher = Principal


# hash -> Principal. Replaced wholesale on refresh, never mutated in place, so a
# request reading it mid-refresh sees a consistent snapshot.
_keys: dict[str, Principal] = {}

# None until the first successful load. This is what distinguishes "no such key"
# (401) from "we have never managed to read the key table" (503).
_loaded_at: Optional[float] = None


def _bootstrap_keys() -> dict[str, Publisher]:
    """
    A single key from the environment, so a fresh install can make its first
    call before any publisher row exists. Refuses to activate in production —
    a hardcoded credential that works in prod is how the dict you are replacing
    got there in the first place.
    """
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

    Joined against publishers so a suspended publisher's keys stop working even
    if nobody remembered to revoke them individually — deactivating the account
    is the action people actually take.
    """
    global _keys, _loaded_at

    try:
        # LEFT JOIN both owner tables and require that the relevant one is
        # active. An INNER JOIN on publishers would silently drop every
        # advertiser key, which is the kind of bug that looks like "auth is
        # broken for advertisers" and takes an hour to find.
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
        # Keep the previous snapshot. An empty dict here would 401 every
        # publisher on the platform because of one bad query.
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

    REVOCATION LATENCY: a revoked key keeps working until the next refresh.
    That window is why revoke_key_now() exists, and why the interval defaults to
    60s rather than the 300s used for CORS — a stolen credential is a worse
    problem than a publisher waiting to go live.
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


# ---------------------------------------------------------------------------
# last_used_at, without a write per request
# ---------------------------------------------------------------------------
#
# "Track when each key was last used" sounds free and is not: at bid volume it
# is one UPDATE per request against a row every request also reads. Postgres
# ends up doing more work maintaining a timestamp nobody reads in real time than
# it does serving the actual application.
#
# So: remember in memory when we last wrote, and write at most once per key per
# window. Resolution drops to ~5 minutes, which is all anyone wants it for
# ("is this key still in use, can I delete it?").

_last_used_written: dict[str, float] = {}


def should_write_last_used(key_id: str) -> bool:
    now = time.time()
    previous = _last_used_written.get(key_id, 0.0)
    if now - previous < settings.last_used_write_interval:
        return False
    _last_used_written[key_id] = now
    return True


# ---------------------------------------------------------------------------
# The FastAPI dependency
# ---------------------------------------------------------------------------

async def _authenticate(
    request: Request,
    x_api_key: Optional[str],
    expected_type: str,
) -> Principal:
    """
    Authenticate the caller. Use as a dependency:

        async def bid(publisher: Publisher = Depends(require_publisher)): ...

    A dependency rather than a helper called inside the handler, because
    FastAPI resolves it before the body is parsed and it shows up in the OpenAPI
    schema. The old authenticate() ran after Pydantic had already validated the
    payload — unauthenticated callers got free input validation.
    """
    if _loaded_at is None:
        # Never loaded. Not the caller's fault; do not blame their key.
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
        # Log the prefix, never the key. Logs get shipped to third parties and
        # sit in files far longer than anyone intends.
        log.warning("rejected API key %s… from %s",
                    x_api_key[:8], request.client.host if request.client else "?")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if principal.owner_type != expected_type:
        # A real, live key being used on the wrong side of the marketplace.
        # 403 not 401: the credential is valid, it is just not authorised here,
        # and telling them to check their key would send them down the wrong path.
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
    """
    Guards the provisioning endpoints. A shared static token is the right amount
    of machinery for endpoints only you call; it is emphatically not enough once
    other humans need access, at which point this becomes real accounts with
    audit logging.

    compare_digest, not ==. String comparison short-circuits on the first
    differing byte, which leaks the correct prefix to anyone who can measure
    response times precisely enough. Cheap to avoid, embarrassing to explain.
    """
    expected = settings.admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="Admin API not configured")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token")