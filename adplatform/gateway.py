# gateway.py — Ad Platform API
#
# COMPLETE FILE. Replace your existing gateway.py with this wholesale.
#
# Merged from your original plus every fix we worked through:
#   1. db._pool read as an attribute, not an import-time snapshot
#   2. spawn() instead of bare asyncio.create_task
#   3. /v1/click no longer takes a redirect parameter (open redirect closed)
#   4. one impression_id, minted before the auction
#   5. server time instead of client timestamp_ms for features
#   6. budget enforcement + spend recording
#   7. win_price logged as CPM, budget decremented by cost_usd

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field, field_validator, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import db
from . import __version__ as VERSION
from . import inventory
from .auth import (
    Principal,
    Publisher,
    require_advertiser,
    cache_status,
    generate_api_key,
    hash_api_key,
    key_prefix,
    load_keys_from_db,
    refresh_keys_loop,
    require_admin,
    require_publisher,
    revoke_key_now,
    add_key_now,
    should_write_last_used,
)
from .cors import DynamicCORSMiddleware, load_origins_from_db, refresh_origins_loop
from .db import (
    get_impression,
    get_publisher_stats,
    init_db,
    mark_clicked,
    save_conversion,
    save_impression,
)
from .events import close_kafka, dropped_count, init_kafka, is_connected, publish_event
from .ml.budget import BudgetTracker, budget_reconcile_loop
from .ml.ctr_model import ctr_model
from .ml.refresh import current_stats, model_refresh_loop, stats_refresh_loop
from .ml.rtb_integration import build_impression_event, run_rtb
from .inventory import invalidate as invalidate_inventory
from .inventory import load_inventory, refresh_inventory_loop
from .settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gateway")

limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Background task registry
#
# The event loop keeps only a WEAK reference to a running task, so a bare
# asyncio.create_task with no strong reference elsewhere can be garbage
# collected mid-execution — dropping impressions and clicks non-deterministically
# under load. It also swallows exceptions. spawn() fixes both.
# ---------------------------------------------------------------------------

_background_tasks: set[asyncio.Task] = set()


def _on_task_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("background task %s failed: %r", task.get_name(), exc, exc_info=exc)


def spawn(coro, *, name: str | None = None) -> asyncio.Task:
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)
    return task


async def drain_background_tasks(timeout: float = 5.0) -> None:
    if not _background_tasks:
        return
    log.info("draining %d background tasks", len(_background_tasks))
    _, pending = await asyncio.wait(set(_background_tasks), timeout=timeout)
    if pending:
        log.warning("%d background tasks unfinished after %.1fs", len(pending), timeout)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting ad platform gateway — %s", settings.describe())

    # Refuse to boot on an insecure production config. Every item this checks is
    # silent at runtime: a default pepper does not raise, it just means the key
    # hashes in your database are forgeable by anyone who read this repo.
    problems = settings.validate_for_production()
    if problems:
        for problem in problems:
            log.critical("FATAL CONFIG: %s", problem)
        raise RuntimeError(f"refusing to start: {len(problems)} config problem(s)")

    await init_db()
    await init_kafka()

    # db._pool as an ATTRIBUTE. `from db import _pool as db_pool` binds a
    # snapshot taken at import time, when it is still None — which is why your
    # CORS never loaded from the database and the warning fired every startup.
    if db._pool:
        await load_origins_from_db(db._pool)
    else:
        log.warning("No DB pool available — CORS limited to dev origins only")

    # ClickHouse and Redis are optional at boot so you can run the gateway
    # locally without the full stack. Missing ClickHouse means the CTR stats
    # stay empty (baseline still serves); missing Redis means budgets are not
    # enforced, which is logged loudly.
    app.state.ch = None
    app.state.redis = None
    app.state.budget = None

    try:
        import clickhouse_connect
        app.state.ch = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            database=settings.clickhouse_database,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
        )
        log.info("ClickHouse connected at %s:%s",
                 settings.clickhouse_host, settings.clickhouse_port)
    except Exception as e:
        log.warning("ClickHouse unavailable (%s) — CTR stats will stay empty", e)

    try:
        import redis.asyncio as aioredis
        app.state.redis = aioredis.from_url(
            settings.redis_url, decode_responses=True)
        await app.state.redis.ping()
        app.state.budget = BudgetTracker(app.state.redis)
        log.info("Redis connected — budget enforcement active")
    except Exception as e:
        log.error("Redis unavailable (%s) — BUDGETS ARE NOT ENFORCED", e)

    ctr_model.load()   # synchronous first load; no-ops if models/current absent

    tasks: list[asyncio.Task] = []
    if db._pool is not None:
        await load_keys_from_db(db._pool)
        # Synchronous first load, like ctr_model.load(): the first bid request
        # should not be served from an empty inventory while a loop waits out
        # its first sleep.
        await load_inventory(db._pool)
        tasks.append(asyncio.create_task(
            refresh_origins_loop(db._pool), name="cors-refresh"))
        tasks.append(asyncio.create_task(
            refresh_keys_loop(db._pool), name="key-refresh"))
        tasks.append(asyncio.create_task(
            refresh_inventory_loop(db._pool), name="inventory-refresh"))
    else:
        # No pool means the key cache never loads, and require_publisher will
        # return 503 rather than 401 for every request. That is correct: with no
        # key table we genuinely cannot tell a valid key from an invalid one, and
        # saying "invalid key" would send publishers debugging our outage.
        log.error("no DB pool — API key auth unavailable, /v1/* will return 503")
    if app.state.ch is not None:
        tasks.append(asyncio.create_task(
            stats_refresh_loop(app.state.ch), name="ctr-stats-refresh"))
    tasks.append(asyncio.create_task(model_refresh_loop(), name="ctr-model-refresh"))
    if app.state.budget is not None and app.state.ch is not None:
        tasks.append(asyncio.create_task(
            budget_reconcile_loop(app.state.budget, app.state.ch),
            name="budget-reconcile"))

    log.info("Gateway ready (ctr model: %s)", ctr_model.model_version)
    try:
        yield
    finally:
        log.info("Shutting down")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await drain_background_tasks()
        await close_kafka()
        if app.state.redis is not None:
            await app.state.redis.aclose()


app = FastAPI(
    title="Ad Platform API",
    version=VERSION,
    description="Real-time bidding gateway with CTR-model ad selection.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(DynamicCORSMiddleware)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BidRequest(BaseModel):
    publisher_id:   str           = Field(..., min_length=1, max_length=64)
    placement_id:   str           = Field(..., min_length=1, max_length=64)
    page_url:       str           = Field(..., max_length=2048)
    page_referrer:  Optional[str] = Field(None, max_length=2048)
    user_id:        str           = Field(..., min_length=1, max_length=128)
    device_type:    str           = Field("desktop")
    viewport_width: Optional[int] = Field(None, ge=0, le=10000)
    timestamp_ms:   int           = Field(...)
    page_keywords:  list[str]     = Field(default_factory=list)

    @field_validator("device_type")
    @classmethod
    def validate_device(cls, v):
        if v not in {"desktop", "mobile", "tablet"}:
            raise ValueError("device_type must be desktop, mobile, or tablet")
        return v

    @field_validator("timestamp_ms")
    @classmethod
    def validate_timestamp(cls, v):
        if abs(int(time.time() * 1000) - v) > 60_000:
            raise ValueError("timestamp_ms is more than 60s from server time")
        return v


class WinResponse(BaseModel):
    impression_id: str
    ad_id:         str
    ad_markup:     str
    win_price:     float = Field(..., description="Clearing price in CPM (USD per 1000)")
    predicted_ctr: float


class NoFillResponse(BaseModel):
    impression_id: str
    message: str = "no_fill"


class ConversionRequest(BaseModel):
    impression_id:    str             = Field(...)
    conversion_type:  str             = Field("purchase", max_length=64)
    conversion_value: Optional[float] = Field(None, ge=0)


class StatsResponse(BaseModel):
    publisher_id: str
    impressions:  int
    clicks:       int
    conversions:  int
    ctr:          float
    revenue_usd:  float


# ---------------------------------------------------------------------------
# Auth
#
# The hardcoded VALID_API_KEYS dict is gone. Keys now live in the api_keys
# table, hashed with a server pepper, cached in memory and refreshed in the
# background. See auth.py for why HMAC-SHA256 rather than argon2, and why this
# fails closed while budget.py fails open.
#
# authenticate() is replaced by the require_publisher dependency. The difference
# matters: FastAPI resolves dependencies BEFORE parsing the request body, so
# unauthenticated callers no longer get free Pydantic validation of whatever
# they send. It also puts the scheme in the OpenAPI docs.
# ---------------------------------------------------------------------------


def note_key_usage(publisher: Publisher) -> None:
    """Fire-and-forget last_used_at, throttled to one write per key per window."""
    if publisher.key_id != "bootstrap" and should_write_last_used(publisher.key_id):
        spawn(db.touch_api_key(publisher.key_id), name="touch-key")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info("%s %s -> %s (%sms)", request.method, request.url.path,
             response.status_code, ms)
    response.headers["X-Latency-Ms"] = str(ms)
    return response


# ---------------------------------------------------------------------------
# POST /v1/bid
# ---------------------------------------------------------------------------

@app.post("/v1/bid", tags=["Serving"],
          summary="Run a real-time auction and return a winning ad")
@limiter.limit(settings.bid_rate_limit)
async def bid(
    request: Request,
    bid_request: BidRequest,
    publisher: Publisher = Depends(require_publisher),
):
    t0 = time.perf_counter()
    # Identity comes from the KEY, never from the body. bid_request.publisher_id
    # is caller-supplied and is not trusted for anything.
    publisher_id = publisher.publisher_id
    note_key_usage(publisher)

    # A key that says pub_a sending publisher_id=pub_b is either a broken
    # integration or someone probing for a confused-deputy bug. Neither should
    # be silently normalised — the publisher needs to know their tag is
    # misconfigured, and if it is probing you want it in the logs.
    claimed = getattr(bid_request, "publisher_id", None)
    if claimed and claimed != publisher_id:
        log.warning("publisher_id mismatch: key=%s body=%s", publisher_id, claimed)
        raise HTTPException(
            status_code=403,
            detail="publisher_id does not match the authenticated key",
        )

    # Mint the id ONCE, before the auction. It goes into the click URL, the
    # Postgres row, and the ClickHouse impression event. Two uuids means the
    # training join matches nothing and the model never sees a positive.
    impression_id = str(uuid.uuid4())

    # Server time, not bid_request.timestamp_ms. The client clock is
    # attacker-controlled and hour_of_day is a model feature.
    request_ts = datetime.now(timezone.utc)

    result, ctx = await run_rtb(
        publisher_id=publisher_id,
        placement_id=bid_request.placement_id,
        device_type=bid_request.device_type,
        page_keywords=bid_request.page_keywords,
        impression_id=impression_id,
        request_ts=request_ts,
        budget_tracker=request.app.state.budget,
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    base_event = {
        "impression_id": impression_id,
        "publisher_id":  publisher_id,
        "placement_id":  bid_request.placement_id,
        "user_id":       bid_request.user_id,
        "device_type":   bid_request.device_type,
        "page_url":      bid_request.page_url,
        "page_keywords": bid_request.page_keywords,
        "timestamp_ms":  bid_request.timestamp_ms,
        "server_ts":     request_ts.isoformat(),
    }

    if result is None:
        log.info("no_fill | publisher=%s | %sms", publisher_id, latency_ms)
        spawn(publish_event("impressions", {**base_event, "filled": False}),
              name="nofill-event")
        return JSONResponse(status_code=200, content={
            "impression_id": impression_id, "message": "no_fill"})

    winner = result.winner
    log.info(
        "win | ad=%s cpm=$%.4f cost=$%.6f ctr=%.5f model=%s%s publisher=%s %sms",
        winner.ad.ad_id, result.win_price, result.cost_usd, winner.predicted_ctr,
        result.model_version, " EXPLORE" if result.is_exploration else "",
        publisher_id, latency_ms,
    )

    # Decrement by cost_usd, NOT win_price. win_price is a CPM; this single
    # impression costs one thousandth of it. Using win_price here would burn a
    # $500 daily budget in about 125 impressions.
    if request.app.state.budget is not None:
        spawn(request.app.state.budget.record_spend(winner.ad.ad_id, result.cost_usd),
              name="record-spend")

    # Postgres row — powers the click lookup. destination_url is what lets
    # /v1/click drop the redirect query parameter.
    spawn(save_impression({
        **base_event,
        "ad_id":           winner.ad.ad_id,
        "advertiser_id":   winner.ad.advertiser_id,
        "destination_url": winner.ad.destination_url,
        "win_price":       result.win_price,
        "cost_usd":        result.cost_usd,
        "predicted_ctr":   winner.predicted_ctr,
        "clicked":         False,
        "converted":       False,
        "filled":          True,
    }), name="save-impression")

    # ClickHouse row — carries the feature vector for training. Deliberately
    # separate from the Postgres write: different shape, lifetime, and consumer.
    spawn(publish_event("impressions", {
        **build_impression_event(result, ctx),
        "user_id":  bid_request.user_id,
        "page_url": bid_request.page_url,
    }), name="impression-event")

    ad_markup = winner.ad.creative_html.replace(
        "{{CLICK_URL}}", f"{settings.public_base_url}/v1/click?id={impression_id}")

    response = JSONResponse(content={
        "impression_id": impression_id,
        "ad_id":         winner.ad.ad_id,
        "ad_markup":     ad_markup,
        "win_price":     result.win_price,
        "predicted_ctr": winner.predicted_ctr,
    })
    response.headers["X-Impression-Id"] = impression_id
    return response


# ---------------------------------------------------------------------------
# GET /v1/click
# ---------------------------------------------------------------------------

@app.get("/v1/click", tags=["Tracking"],
         summary="Record a click then redirect to the advertiser")
async def track_click(id: str = Query(..., description="impression_id")):
    """
    No `redirect` parameter. The destination is read from the impression row,
    so there is nothing for an attacker to point elsewhere.

    The old version accepted any http(s) URL as a query parameter and redirected
    to it even when the impression_id did not exist — a fully open redirect
    requiring no valid id at all.
    """
    impression = await get_impression(id)
    if impression is None:
        log.warning("click on unknown impression | id=%s", id)
        raise HTTPException(status_code=404, detail="Unknown impression")

    destination = impression.get("destination_url")
    if not destination or not destination.startswith(("http://", "https://")):
        log.error("impression %s has no usable destination_url", id)
        raise HTTPException(status_code=500, detail="Missing destination")

    # Atomic check-and-set. mark_clicked flips the flag in a single UPDATE with
    # `AND clicked = FALSE`, and returns True only if THIS call was the one that
    # changed it. Awaited rather than spawned, because the return value gates
    # whether we emit the event.
    #
    # The previous read-then-write version let two concurrent clicks on the same
    # impression both observe clicked=False and both publish. A double-counted
    # click is a corrupted training label and inflates that ad's measured CTR.
    if not await mark_clicked(id):
        log.info("duplicate click ignored | id=%s", id)
        return RedirectResponse(url=destination, status_code=302)

    # No `label` field here. The training label comes from the LEFT JOIN in
    # train_ctr.py, because a label needs the non-clicks too. A click stream on
    # its own is 100% positives and cannot train anything.
    spawn(publish_event("clicks", {
        "impression_id": id,
        "ad_id":         impression["ad_id"],
        "placement_id":  impression["placement_id"],
        "clicked_at_ms": int(time.time() * 1000),
    }), name="click-event")

    log.info("click | impression=%s | ad=%s", id, impression["ad_id"])
    return RedirectResponse(url=destination, status_code=302)


# ---------------------------------------------------------------------------
# GET /v1/impression — render confirmation pixel
# ---------------------------------------------------------------------------

@app.get("/v1/impression", tags=["Tracking"],
         summary="Confirm ad rendered in browser (1x1 pixel)")
async def impression_pixel(id: str = Query(..., description="impression_id")):
    log.info("impression rendered | id=%s", id)
    gif = bytes([
        0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00,
        0x80, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x21,
        0xf9, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44,
        0x01, 0x00, 0x3b,
    ])
    return Response(content=gif, media_type="image/gif")


# ---------------------------------------------------------------------------
# POST /v1/conversion
# ---------------------------------------------------------------------------

@app.post("/v1/conversion", tags=["Tracking"],
          summary="Record a post-click conversion")
async def track_conversion(
    conversion: ConversionRequest,
    publisher: Publisher = Depends(require_publisher),
):
    note_key_usage(publisher)

    impression = await get_impression(conversion.impression_id)
    if impression is None:
        raise HTTPException(status_code=404, detail="Impression not found")

    # Any valid key used to be able to post a conversion against ANY impression,
    # and the 404-vs-200 difference let one publisher enumerate another's
    # impression ids. Return 404 rather than 403 on a mismatch: 403 confirms the
    # impression exists, which is the leak.
    if impression.get("publisher_id") != publisher.publisher_id:
        log.warning("cross-publisher conversion attempt: key=%s impression=%s",
                    publisher.publisher_id, conversion.impression_id)
        raise HTTPException(status_code=404, detail="Impression not found")

    conversion_event = {
        "impression_id":    conversion.impression_id,
        "ad_id":            impression["ad_id"],
        "advertiser_id":    impression["advertiser_id"],
        "user_id":          impression["user_id"],
        "conversion_type":  conversion.conversion_type,
        "conversion_value": conversion.conversion_value,
        "converted_at_ms":  int(time.time() * 1000),
    }

    spawn(save_conversion(conversion_event), name="save-conversion")
    spawn(publish_event("conversions", conversion_event), name="conversion-event")

    log.info("conversion | impression=%s type=%s value=%s",
             conversion.impression_id, conversion.conversion_type,
             conversion.conversion_value)
    return {"status": "ok", "impression_id": conversion.impression_id}


# ---------------------------------------------------------------------------
# GET /v1/stats
# ---------------------------------------------------------------------------

@app.get("/v1/stats", response_model=StatsResponse, tags=["Reporting"],
         summary="Publisher performance stats (last 30 days)")
@limiter.limit(settings.stats_rate_limit)
async def publisher_stats(
    request: Request,
    publisher: Publisher = Depends(require_publisher),
):
    # No publisher_id parameter, by design. The moment reporting takes an id
    # from the caller you own an authorisation check on every field you return,
    # and one missed check is a cross-tenant data leak.
    note_key_usage(publisher)
    return await get_publisher_stats(publisher.publisher_id)


# ---------------------------------------------------------------------------
# Advertiser API — the demand side.
#
# Every endpoint here derives advertiser_id from the API KEY, never from a path
# or body parameter. The moment an id is caller-supplied you owe an ownership
# check on every field you touch, and one missed check is a cross-tenant write.
# ---------------------------------------------------------------------------

class CampaignRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    daily_budget_usd: float = Field(..., gt=0, le=1_000_000)
    target_cpm: float = Field(..., gt=0, le=1000,
                              description="Max CPM — dollars per 1000 impressions")
    floor_price: float = Field(0.0, ge=0, le=1000)
    target_device: str = Field("all", pattern="^(all|mobile|desktop|tablet)$")
    target_keywords: list[str] = Field(default_factory=list, max_length=100)
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    @field_validator("target_keywords")
    @classmethod
    def normalise_keywords(cls, v: list[str]) -> list[str]:
        # Normalised HERE, once, on write. features.extract_features lowercases
        # ad keywords on every scoring call to compensate for inventory that was
        # not normalised; doing it at the boundary means storage and scoring
        # agree and the hot path does less work.
        return sorted({k.strip().lower() for k in v if k.strip()})

    @model_validator(mode="after")
    def check_consistency(self):
        if self.floor_price > self.target_cpm:
            raise ValueError("floor_price cannot exceed target_cpm — the campaign "
                             "could never win an auction")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date is before start_date")
        return self


class AdRequest(BaseModel):
    campaign_id: str
    name: Optional[str] = None
    creative_html: str = Field(..., min_length=1, max_length=20_000)
    destination_url: str

    @field_validator("creative_html")
    @classmethod
    def must_have_click_macro(cls, v: str) -> str:
        # Without the macro the creative renders but every click is untracked:
        # no click row, no training label, and the advertiser is billed for
        # impressions whose performance is invisible. Rejecting at write time
        # beats discovering it in a week of flat CTR.
        if "{{CLICK_URL}}" not in v:
            raise ValueError("creative_html must contain {{CLICK_URL}} — without it "
                             "clicks cannot be tracked or attributed")
        return v

    @field_validator("destination_url")
    @classmethod
    def must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("destination_url must start with http:// or https://")
        return v


@app.post("/v1/campaigns", tags=["Advertiser"], status_code=201,
          summary="Create a campaign")
async def create_campaign(
    body: CampaignRequest,
    advertiser: Principal = Depends(require_advertiser),
):
    note_key_usage(advertiser)
    campaign_id = f"camp_{uuid.uuid4().hex[:12]}"

    created = await db.create_campaign({
        "campaign_id": campaign_id,
        "advertiser_id": advertiser.advertiser_id,
        **body.model_dump(),
    })
    if created is None:
        raise HTTPException(status_code=400, detail="Could not create campaign")

    # No inventory reload: a campaign with no ads is not servable, so nothing
    # about the auction changed.
    log.info("campaign created: %s by %s ($%.2f/day, $%.2f CPM)",
             campaign_id, advertiser.advertiser_id,
             body.daily_budget_usd, body.target_cpm)
    return {"campaign_id": campaign_id, "status": "active", **body.model_dump(mode="json")}


@app.get("/v1/campaigns", tags=["Advertiser"], summary="List your campaigns")
async def list_campaigns(advertiser: Principal = Depends(require_advertiser)):
    note_key_usage(advertiser)
    return {"campaigns": await db.list_campaigns(advertiser.advertiser_id)}


@app.post("/v1/ads", tags=["Advertiser"], status_code=201,
          summary="Create an ad under a campaign")
async def create_ad(
    body: AdRequest,
    advertiser: Principal = Depends(require_advertiser),
):
    note_key_usage(advertiser)

    # Ownership check before the insert. The FK would accept any existing
    # campaign_id, including another advertiser's, which would let anyone attach
    # a creative to a competitor's budget.
    owned = {c["campaign_id"] for c in await db.list_campaigns(advertiser.advertiser_id)}
    if body.campaign_id not in owned:
        raise HTTPException(status_code=404, detail="Unknown campaign_id")

    ad_id = f"ad_{uuid.uuid4().hex[:12]}"
    if await db.create_ad({"ad_id": ad_id, **body.model_dump()}) is None:
        raise HTTPException(status_code=400, detail="Could not create ad")

    # THIS one changes what is servable, so reload now rather than leaving the
    # advertiser to wonder for a minute whether their ad went live.
    await invalidate_inventory(db._pool)
    log.info("ad created: %s under %s", ad_id, body.campaign_id)
    return {"ad_id": ad_id, "campaign_id": body.campaign_id, "status": "active"}


@app.patch("/v1/campaigns/{campaign_id}/status", tags=["Advertiser"],
           summary="Pause, resume, or archive a campaign")
async def set_campaign_status(
    campaign_id: str,
    status: str = Query(..., pattern="^(active|paused|archived)$"),
    advertiser: Principal = Depends(require_advertiser),
):
    note_key_usage(advertiser)
    if not await db.set_status("campaigns", "campaign_id", campaign_id,
                               status, advertiser.advertiser_id):
        raise HTTPException(status_code=404, detail="Unknown campaign_id")
    await invalidate_inventory(db._pool)
    return {"campaign_id": campaign_id, "status": status}


@app.patch("/v1/ads/{ad_id}/status", tags=["Advertiser"],
           summary="Pause, resume, or archive a single creative")
async def set_ad_status(
    ad_id: str,
    status: str = Query(..., pattern="^(active|paused|archived)$"),
    advertiser: Principal = Depends(require_advertiser),
):
    note_key_usage(advertiser)
    if not await db.set_status("ads", "ad_id", ad_id, status,
                               advertiser.advertiser_id):
        raise HTTPException(status_code=404, detail="Unknown ad_id")
    await invalidate_inventory(db._pool)
    return {"ad_id": ad_id, "status": status}


@app.get("/v1/advertiser/stats", tags=["Advertiser"],
         summary="Your delivery and spend (last 30 days)")
@limiter.limit(settings.stats_rate_limit)
async def advertiser_stats(
    request: Request,
    advertiser: Principal = Depends(require_advertiser),
):
    note_key_usage(advertiser)
    return await db.get_advertiser_stats(advertiser.advertiser_id)


# ---------------------------------------------------------------------------
# Provisioning — POST /admin/publishers, /admin/publishers/{id}/keys, ...
#
# Guarded by X-Admin-Token, which is a static shared secret. That is the right
# amount of machinery for endpoints only you call, and definitively not enough
# once other people need access — at that point this becomes real accounts with
# an audit trail.
# ---------------------------------------------------------------------------

class CreatePublisherRequest(BaseModel):
    publisher_id: str = Field(..., min_length=3, max_length=64,
                              pattern=r"^[a-z0-9_]+$")
    domain: str = Field(..., description="Origin including scheme, e.g. https://blog.example")
    name: Optional[str] = None

    @field_validator("domain")
    @classmethod
    def check_scheme(cls, v: str) -> str:
        # cors.add_origin raises on a bare hostname. Catching it here turns a
        # 500 into a readable 422 that says which field is wrong.
        if not v.startswith(("http://", "https://")):
            raise ValueError("domain must include the scheme, e.g. https://blog.example")
        return v.rstrip("/")


@app.post("/admin/publishers", tags=["Admin"], status_code=201,
          summary="Create a publisher and issue its first API key",
          dependencies=[Depends(require_admin)])
async def admin_create_publisher(body: CreatePublisherRequest):
    """
    The plaintext key is in this response and nowhere else, ever. It is not
    logged, not stored, and not recoverable — only its HMAC is persisted. A lost
    key is rotated, not retrieved, which is the property that makes the hash
    worth having.
    """
    if db._pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    if not await db.create_publisher(body.publisher_id, body.domain, body.name):
        raise HTTPException(status_code=409,
                            detail="publisher_id or domain already exists")

    api_key = generate_api_key(test=not settings.is_production)
    key_id = f"key_{uuid.uuid4().hex[:16]}"

    if not await db.insert_api_key(key_id, body.publisher_id,
                                   hash_api_key(api_key), key_prefix(api_key),
                                   name="initial"):
        raise HTTPException(status_code=500, detail="Failed to store API key")

    # Both caches updated in-process so the publisher's very first request
    # works, rather than failing for up to a minute while they conclude the
    # integration is broken.
    add_key_now(api_key, Publisher(publisher_id=body.publisher_id, key_id=key_id,
                                   key_prefix=key_prefix(api_key),
                                   domain=body.domain))
    await add_origin(db._pool, body.domain)

    log.info("publisher created: %s (%s) key=%s",
             body.publisher_id, body.domain, key_prefix(api_key))
    return {
        "publisher_id": body.publisher_id,
        "domain":       body.domain,
        "key_id":       key_id,
        "api_key":      api_key,
        "warning":      "Store this key now — it cannot be retrieved again.",
    }


@app.post("/admin/publishers/{publisher_id}/keys", tags=["Admin"], status_code=201,
          summary="Issue an additional key (rotation)",
          dependencies=[Depends(require_admin)])
async def admin_create_key(publisher_id: str, name: Optional[str] = None):
    """
    Issue a second live key so a publisher can rotate without downtime: deploy
    the new key, confirm traffic on it, then revoke the old one. This is the
    whole reason keys are a separate table rather than a column on publishers.
    """
    if db._pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    api_key = generate_api_key(test=not settings.is_production)
    key_id = f"key_{uuid.uuid4().hex[:16]}"

    if not await db.insert_api_key(key_id, publisher_id, hash_api_key(api_key),
                                   key_prefix(api_key), name=name or "rotated"):
        raise HTTPException(status_code=404, detail="Unknown publisher_id")

    add_key_now(api_key, Publisher(publisher_id=publisher_id, key_id=key_id,
                                   key_prefix=key_prefix(api_key)))
    return {"key_id": key_id, "api_key": api_key,
            "warning": "Store this key now — it cannot be retrieved again."}


class CreateAdvertiserRequest(BaseModel):
    advertiser_id: str = Field(..., min_length=3, max_length=64,
                               pattern=r"^[a-z0-9_]+$")
    name: str = Field(..., min_length=1, max_length=200)
    contact_email: Optional[str] = None


@app.post("/admin/advertisers", tags=["Admin"], status_code=201,
          summary="Create an advertiser and issue its first API key",
          dependencies=[Depends(require_admin)])
async def admin_create_advertiser(body: CreateAdvertiserRequest):
    if db._pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    if not await db.create_advertiser(body.advertiser_id, body.name, body.contact_email):
        raise HTTPException(status_code=409, detail="advertiser_id already exists")

    api_key = generate_api_key(test=not settings.is_production)
    key_id = f"key_{uuid.uuid4().hex[:16]}"

    if not await db.insert_api_key(key_id, None, hash_api_key(api_key),
                                   key_prefix(api_key), name="initial",
                                   owner_type="advertiser",
                                   owner_id=body.advertiser_id):
        raise HTTPException(status_code=500, detail="Failed to store API key")

    add_key_now(api_key, Principal(owner_id=body.advertiser_id, key_id=key_id,
                                   key_prefix=key_prefix(api_key),
                                   owner_type="advertiser"))
    log.info("advertiser created: %s key=%s", body.advertiser_id, key_prefix(api_key))
    return {
        "advertiser_id": body.advertiser_id,
        "key_id": key_id,
        "api_key": api_key,
        "warning": "Store this key now — it cannot be retrieved again.",
    }


@app.get("/admin/publishers/{publisher_id}/keys", tags=["Admin"],
         summary="List key metadata (never the keys themselves)",
         dependencies=[Depends(require_admin)])
async def admin_list_keys(publisher_id: str):
    return {"publisher_id": publisher_id, "keys": await db.list_api_keys(publisher_id)}


@app.delete("/admin/keys/{key_id}", tags=["Admin"],
            summary="Revoke a key immediately",
            dependencies=[Depends(require_admin)])
async def admin_revoke_key(key_id: str):
    """
    Writes revoked_at AND evicts from the live cache. Doing only the DB write
    leaves the key working until the next refresh; doing only the eviction means
    it comes back on the next refresh. Both, in that order.
    """
    key_hash = await db.revoke_api_key(key_id)
    if key_hash is None:
        raise HTTPException(status_code=404, detail="Unknown or already-revoked key_id")
    revoke_key_now(key_hash)
    return {"key_id": key_id, "revoked": True}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

async def _probe_postgres() -> str:
    if db._pool is None:
        return "down"
    try:
        async with db._pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return "up"
    except Exception:
        return "error"


async def _probe_redis() -> str:
    if app.state.redis is None:
        return "down"
    try:
        await app.state.redis.ping()
        return "up"
    except Exception:
        return "error"


async def _probe_clickhouse() -> str:
    if app.state.ch is None:
        return "down"
    try:
        await asyncio.to_thread(app.state.ch.command, "SELECT 1")
        return "up"
    except Exception:
        return "error"


@app.get("/health", tags=["System"], summary="Health check")
async def health(deep: bool = False):
    """
    Cheap by default — reports the connection state the lifespan established,
    with no I/O. A load balancer polling this every second must not open a
    Postgres connection every second.

    `?deep=true` actually round-trips to each store. Use it after `compose up`
    and in scripts/verify_stack.sh; do not point a health check at it.

    `stores` is the thing to read: all four must say "up" before the pipeline is
    whole. A gateway with clickhouse down still serves ads perfectly — it just
    serves them from the baseline model forever and never learns anything, which
    is exactly the failure that is easy to miss.
    """
    if deep:
        pg, redis_state, ch = await asyncio.gather(
            _probe_postgres(), _probe_redis(), _probe_clickhouse()
        )
    else:
        pg = "up" if db._pool is not None else "down"
        redis_state = "up" if app.state.redis is not None else "down"
        ch = "up" if app.state.ch is not None else "down"

    stores = {
        "postgres":   pg,
        "redis":      redis_state,
        "clickhouse": ch,
        # is_connected() reflects whether the producer started, not whether the
        # broker is reachable right now. dropped_count climbing while this says
        # "up" means the broker went away after startup.
        "kafka":      "up" if is_connected() else "down",
    }
    degraded = [name for name, state in stores.items() if state != "up"]

    return {
        "status":       "ok" if not degraded else "degraded",
        "service":      "ad-platform-gateway",
        "version":      VERSION,
        "env":          settings.env,
        "stores":       stores,
        "degraded":     degraded,
        "ctr_model":    ctr_model.model_version,
        "ctr_stats":    _stats_summary(),
        "budgets":      "enforced" if app.state.budget else "NOT ENFORCED",
        "events_dropped": dropped_count(),
        "auth":         cache_status(),
        "inventory":    inventory.status(),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }


def _stats_summary() -> dict:
    """
    Whether the CTR stats snapshot has actually loaded. "0 pairs" here after a
    few minutes of traffic means stats_refresh_loop is failing or ClickHouse is
    empty — the gateway will keep serving from the baseline and say nothing.
    """
    stats = current_stats()
    return {
        "pairs":      len(stats.pair_counts),
        "global_ctr": round(stats.global_ctr, 6),
    }