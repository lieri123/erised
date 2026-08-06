# cors.py — Dynamic CORS middleware
#
# Why this exists:
#   The static CORSMiddleware(allow_origins=["*"]) lets any website call your
#   API from a browser. That means anyone could embed your JS tag, burn through
#   your infrastructure, and you couldn't stop them at the browser level.
#
#   This middleware only echoes Access-Control-Allow-Origin back to origins that
#   are registered publishers in your database. Unregistered origins get no
#   CORS headers — the browser blocks the response from reaching their JS.
#
# Important distinction:
#   CORS only stops browsers. curl, Postman, and scripts don't send an Origin
#   header and don't enforce CORS. The X-API-Key check in each route is what
#   stops non-browser clients. These two layers protect against different threats:
#
#     Unauthorized browser (no API key, wrong origin) → blocked by CORS
#     Unauthorized script (no API key)                → blocked by X-API-Key
#     Authorized publisher (valid key, registered origin) → passes both
#
# How origins stay fresh:
#   On startup:  load_origins_from_db() pulls all active publisher domains
#   Every 5min:  refresh_origins_loop() re-pulls so new publishers activate
#                without a server restart
#   On signup:   new publisher row inserted → active within 5 minutes max
#
# If Postgres is unavailable:
#   Falls back to DEV_ORIGINS (localhost only) and logs a warning.
#   The server keeps running — CORS just becomes more restrictive.

import asyncio
import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .settings import settings

log = logging.getLogger("cors")

# ---------------------------------------------------------------------------
# In-memory origin set
#
# This is the live set the middleware checks on every request.
# It is populated from Postgres on startup and refreshed every 5 minutes.
# Never hardcode publisher domains here — add them to the publishers table.
# ---------------------------------------------------------------------------

# Always-allowed origins — local development only
# In production you may want to remove localhost entirely
# Set DEV_ORIGINS="" in the production environment to drop localhost entirely.
DEV_ORIGINS: set[str] = set(settings.dev_origins)

# Live set — populated from DB at runtime, never edited directly in code
REGISTERED_ORIGINS: set[str] = set(DEV_ORIGINS)

# How often (seconds) to re-pull origins from Postgres
REFRESH_INTERVAL_SECONDS = settings.cors_refresh_interval


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------

async def load_origins_from_db(pool) -> int:
    """
    Pull every active publisher domain from Postgres into REGISTERED_ORIGINS.
    Returns the number of domains loaded.

    The publishers table looks like:
        publisher_id  TEXT PRIMARY KEY
        domain        TEXT NOT NULL UNIQUE   -- e.g. "https://techblog.com"
        active        BOOLEAN DEFAULT TRUE

    A publisher going inactive (active=FALSE) will be removed from REGISTERED_ORIGINS
    on the next refresh cycle — their tag stops working within 5 minutes.
    """
    try:
        rows = await pool.fetch("""
            SELECT domain FROM publishers
            WHERE active = TRUE
              AND domain IS NOT NULL
              AND domain != ''
        """)

        fresh_domains = {row["domain"] for row in rows}

        # Rebuild the set — don't mutate while iterating
        REGISTERED_ORIGINS.clear()
        REGISTERED_ORIGINS.update(DEV_ORIGINS)       # always keep dev origins
        REGISTERED_ORIGINS.update(fresh_domains)

        log.info(
            f"CORS origins refreshed — "
            f"{len(fresh_domains)} publisher domains + {len(DEV_ORIGINS)} dev origins"
        )
        return len(fresh_domains)

    except Exception as e:
        # If Postgres is down, keep whatever is in memory — don't wipe it
        log.error(f"Failed to load CORS origins from DB: {e}")
        log.warning("Keeping existing CORS origin set until next refresh")
        return 0


async def add_origin(pool, domain: str):
    """
    Add a single origin immediately without waiting for the next refresh cycle.
    Call this right after inserting a new publisher row so their tag works instantly.

    Usage:
        await add_origin(db_pool, "https://newpublisher.com")
    """
    if not domain.startswith(("http://", "https://")):
        raise ValueError(f"Domain must start with http:// or https://: {domain}")

    REGISTERED_ORIGINS.add(domain)
    log.info(f"CORS origin added immediately: {domain}")

# ---------------------------------------------------------------------------
# Background refresh loop
# ---------------------------------------------------------------------------

async def refresh_origins_loop(pool):
    """
    Background task started at server startup. Re-pulls publisher domains
    from Postgres every REFRESH_INTERVAL_SECONDS.

    Failures are logged but never crash the loop — the server keeps running
    with whatever set of origins it had before the failure.

    Cancel this task in the lifespan shutdown block:
        refresh_task.cancel()
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


# ---------------------------------------------------------------------------
# The middleware itself
# ---------------------------------------------------------------------------

class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """
    CORS middleware that only allows registered publisher origins.

    On every request:
      1. Read the Origin header the browser sends
      2. Check if it's in REGISTERED_ORIGINS
      3. If yes  → echo it back in Access-Control-Allow-Origin
      4. If no   → send no CORS headers → browser blocks the response

    OPTIONS (preflight):
      Browsers send a preflight OPTIONS request before the real POST to check
      if CORS will allow it. We return 204 immediately with CORS headers if
      the origin is registered, or 204 with no headers if it isn't.
      Without handling OPTIONS, the browser never sends the actual bid request.
    """

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")

        # --- Preflight (OPTIONS) ---
        # Browser sends this before every cross-origin POST to ask:
        # "will you allow my real request?" We answer here.
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
                # Unregistered origin — return 204 but with no CORS headers
                # Browser sees no Allow-Origin and blocks the real request
                log.warning(f"Preflight rejected for unregistered origin: {origin}")
                return Response(status_code=204)

        # --- Actual request ---
        response = await call_next(request)

        if origin and origin in REGISTERED_ORIGINS:
            # Registered publisher — allow the browser to read the response
            response.headers["Access-Control-Allow-Origin"]   = origin
            response.headers["Access-Control-Allow-Methods"]  = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"]  = "Content-Type, X-API-Key"
            response.headers["Access-Control-Expose-Headers"] = "X-Impression-Id, X-Latency-Ms"
            response.headers["Vary"]                          = "Origin"

        elif origin:
            # Origin present but not registered — log it so you can investigate
            # Could be: publisher forgot to register, typo in domain, bad actor
            log.warning(f"CORS blocked unregistered origin: {origin}")

        # No origin header = not a browser request (curl, script, etc.)
        # X-API-Key check in the route handles those

        return response