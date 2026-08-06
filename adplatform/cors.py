# cors.py — Dynamic CORS middleware

import asyncio
import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .settings import settings

log = logging.getLogger("cors")

DEV_ORIGINS: set[str] = set(settings.dev_origins)

REGISTERED_ORIGINS: set[str] = set(DEV_ORIGINS)

REFRESH_INTERVAL_SECONDS = settings.cors_refresh_interval


# DB loading

async def load_origins_from_db(pool) -> int:
    """
    Pull every active publisher domain from Postgres into REGISTERED_ORIGINS.
    Returns the number of domains loaded.
    """
    try:
        rows = await pool.fetch("""
            SELECT domain FROM publishers
            WHERE active = TRUE
              AND domain IS NOT NULL
              AND domain != ''
        """)

        fresh_domains = {row["domain"] for row in rows}

        REGISTERED_ORIGINS.clear()
        REGISTERED_ORIGINS.update(DEV_ORIGINS)       # always keep dev origins
        REGISTERED_ORIGINS.update(fresh_domains)

        log.info(
            f"CORS origins refreshed — "
            f"{len(fresh_domains)} publisher domains + {len(DEV_ORIGINS)} dev origins"
        )
        return len(fresh_domains)

    except Exception as e:
        log.error(f"Failed to load CORS origins from DB: {e}")
        log.warning("Keeping existing CORS origin set until next refresh")
        return 0


async def add_origin(pool, domain: str):
    """
    Add a single origin immediately without waiting for the next refresh cycle.
    Call this right after inserting a new publisher row so their tag works instantly.
    """
    if not domain.startswith(("http://", "https://")):
        raise ValueError(f"Domain must start with http:// or https://: {domain}")

    REGISTERED_ORIGINS.add(domain)
    log.info(f"CORS origin added immediately: {domain}")

# Background refresh loop

async def refresh_origins_loop(pool):
    """
    Background task started at server startup. Re-pulls publisher domains
    from Postgres every REFRESH_INTERVAL_SECONDS.

    Failures are logged but never crash the loop — the server keeps running
    with whatever set of origins it had before the failure.
    """
    log.info(f"CORS refresh loop started — interval: {REFRESH_INTERVAL_SECONDS}s")

    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        try:
            count = await load_origins_from_db(pool)
            log.debug(f"CORS refresh complete — {len(REGISTERED_ORIGINS)} total origins ({count} from DB)")
        except asyncio.CancelledError:
            log.info("CORS refresh loop cancelled")
            break
        except Exception as e:
            # Log and continue — never let a refresh failure kill the server
            log.error(f"CORS refresh loop error: {e}")


# The middleware

class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """
    CORS middleware that only allows registered publisher origins.

    On every request:
      1. Read the Origin header the browser sends
      2. Check if it's in REGISTERED_ORIGINS
      3. If yes  → echo it back in Access-Control-Allow-Origin
      4. If no   → send no CORS headers → browser blocks the response
    """

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")

        if request.method == "OPTIONS":
            if origin and origin in REGISTERED_ORIGINS:
                return Response(
                    status_code=204,
                    headers={
                        "Access-Control-Allow-Origin":  origin,
                        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
                        "Access-Control-Max-Age":       "86400",   # cache preflight 24h
                        "Vary":                         "Origin",
                    }
                )
            else:
                log.warning(f"Preflight rejected for unregistered origin: {origin}")
                return Response(status_code=204)

        response = await call_next(request)

        if origin and origin in REGISTERED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"]   = origin
            response.headers["Access-Control-Allow-Methods"]  = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"]  = "Content-Type, X-API-Key"
            response.headers["Access-Control-Expose-Headers"] = "X-Impression-Id, X-Latency-Ms"
            response.headers["Vary"]                          = "Origin"

        elif origin:
            log.warning(f"CORS blocked unregistered origin: {origin}")

        return response