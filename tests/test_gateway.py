# tests/test_gateway.py — the HTTP surface.
#
# Until this file existed the suite tested everything BELOW the gateway and
# nothing at it: 928 lines of routing, auth wiring, and response shaping whose
# only check was `scripts/check_imports` proving the module imports.
#
# The header of gateway.py lists seven bugs that were live in this file at one
# time or another — an open redirect on /v1/click, background tasks garbage
# collected mid-flight, features built from a client-supplied timestamp, budgets
# not enforced, and a missing import that made POST /admin/publishers return 500
# *after* writing the publisher row and issuing its key. Every one of them is a
# request away from being caught. None of them were.
#
# The stores are faked rather than mocked at the driver level, because what
# needs testing here is the gateway's own logic — which principal a key maps to,
# which errors turn into which status codes, what gets spawned and what does
# not. Postgres' own behaviour is not under test.

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import pytest

from adplatform import auth, db, gateway, inventory
from adplatform.auth import Principal, hash_api_key
from adplatform.rtb import Ad
from adplatform.settings import settings
from adplatform.signing import build_click_url, sign_click

PUBLISHER_KEY = "pk_test_publisher_key_for_tests"
ADVERTISER_KEY = "pk_test_advertiser_key_for_tests"
ADMIN_TOKEN = "test-admin-token"


# --- fakes -----------------------------------------------------------------

class FakePool:
    """Enough asyncpg surface for the handlers that touch the pool directly."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []

    async def fetch(self, _query, *args):
        return self.rows

    async def fetchrow(self, _query, *args):
        return self.rows[0] if self.rows else None

    async def fetchval(self, _query, *args):
        return 1

    async def execute(self, query, *args):
        self.executed.append((query, args))

    @asynccontextmanager
    async def acquire(self):
        yield self


class Spy:
    """Records calls to a coroutine the gateway fires via spawn()."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    @property
    def count(self) -> int:
        return len(self.calls)


def make_ad(ad_id="ad_test", cpm=5.0, device="all", campaign="camp_test"):
    return Ad(
        ad_id=ad_id,
        advertiser_id="adv_test",
        creative_html='<a href="{{CLICK_URL}}">buy</a>',
        destination_url="https://advertiser.example.com/landing",
        target_cpm=cpm,
        floor_price=1.0,
        target_device=device,
        target_keywords=["python"],
        daily_budget_usd=500.0,
        campaign_id=campaign,
        created_at=datetime.now(timezone.utc),
    )


def impression_row(**over):
    row = {
        "impression_id": "imp_1",
        "publisher_id": "pub_test",
        "placement_id": "place_1",
        "user_id": "user_1",
        "ad_id": "ad_test",
        "advertiser_id": "adv_test",
        "destination_url": "https://advertiser.example.com/landing",
        "page_keywords": ["python"],
        "clicked": False,
        "converted": False,
    }
    row.update(over)
    return row


# --- harness ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def wired(monkeypatch):
    """
    Stand the app up without its lifespan.

    The lifespan opens Postgres, Kafka, Redis and ClickHouse connections, so
    running it would make every test in this file an integration test. Instead
    the module-level state it would have populated is set directly, which is
    also the only way to exercise the *unpopulated* cases — auth returning 503
    before the key cache loads has no other route to it.
    """
    monkeypatch.setattr(db, "_pool", FakePool(), raising=False)

    # settings is a frozen dataclass; object.__setattr__ is the only way in, and
    # require_admin reads the attribute per request so this takes effect.
    object.__setattr__(settings, "admin_token", ADMIN_TOKEN)

    monkeypatch.setattr(auth, "_keys", {
        hash_api_key(PUBLISHER_KEY): Principal(
            owner_id="pub_test", key_id="key_pub", key_prefix="pk_test_publishe",
            owner_type="publisher", domain="https://pub.example",
        ),
        hash_api_key(ADVERTISER_KEY): Principal(
            owner_id="adv_test", key_id="key_adv", key_prefix="pk_test_advertis",
            owner_type="advertiser",
        ),
    })
    monkeypatch.setattr(auth, "_loaded_at", time.time())
    monkeypatch.setattr(auth, "_last_used_written", {})

    gateway.app.state.redis = None
    gateway.app.state.ch = None
    gateway.app.state.budget = None

    # Fire-and-forget writes: silenced by default, individual tests re-patch
    # with a Spy when they assert on them.
    monkeypatch.setattr(gateway, "save_impression", Spy())
    monkeypatch.setattr(gateway, "publish_event", Spy())
    monkeypatch.setattr(gateway, "save_conversion", Spy())
    monkeypatch.setattr(db, "touch_api_key", Spy())

    inventory._reset_for_tests()
    inventory._ads = [make_ad()]
    inventory._loaded_at = 1.0
    yield
    inventory._reset_for_tests()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=gateway.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://testserver") as c:
        yield c


async def drain():
    """Let spawn()ed tasks run before asserting on their side effects."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def bid_body(**over):
    body = {
        "publisher_id": "pub_test",
        "placement_id": "place_1",
        "page_url": "https://pub.example/article",
        "user_id": "user_1",
        "device_type": "desktop",
        "timestamp_ms": int(time.time() * 1000),
        "page_keywords": ["python"],
    }
    body.update(over)
    return body


def pub_headers():
    return {"X-API-Key": PUBLISHER_KEY}


def adv_headers():
    return {"X-API-Key": ADVERTISER_KEY}


# --- authentication --------------------------------------------------------

class TestAuthentication:
    """
    owner_type is checked in the dependency, not the handler. That is the
    property under test: a publisher key must not reach an advertiser endpoint
    even though both are valid keys in the same table.
    """

    async def test_missing_key_is_401(self, client):
        r = await client.post("/v1/bid", json=bid_body())
        assert r.status_code == 401
        assert r.headers.get("WWW-Authenticate") == "ApiKey"

    async def test_unknown_key_is_401(self, client):
        r = await client.post("/v1/bid", json=bid_body(),
                              headers={"X-API-Key": "pk_test_not_a_real_key"})
        assert r.status_code == 401

    async def test_advertiser_key_cannot_bid(self, client):
        r = await client.post("/v1/bid", json=bid_body(), headers=adv_headers())
        assert r.status_code == 403

    async def test_publisher_key_cannot_create_a_campaign(self, client):
        r = await client.post("/v1/campaigns", headers=pub_headers(), json={
            "name": "c", "daily_budget_usd": 10.0, "target_cpm": 5.0})
        assert r.status_code == 403

    async def test_503_before_the_key_cache_loads(self, client, monkeypatch):
        """
        With no key table loaded we cannot tell a valid key from an invalid one.
        401 would send a publisher debugging their own integration during our
        outage; 503 with Retry-After says whose problem it is.
        """
        monkeypatch.setattr(auth, "_loaded_at", None)
        r = await client.post("/v1/bid", json=bid_body(), headers=pub_headers())
        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "5"

    async def test_a_revoked_key_stops_working_immediately(self, client):
        auth.revoke_key_now(hash_api_key(PUBLISHER_KEY))
        r = await client.post("/v1/bid", json=bid_body(), headers=pub_headers())
        assert r.status_code == 401


# --- POST /v1/bid ----------------------------------------------------------

class TestBid:

    async def test_a_win_returns_a_creative_and_a_signed_click_url(self, client):
        r = await client.post("/v1/bid", json=bid_body(), headers=pub_headers())
        assert r.status_code == 200

        body = r.json()
        assert body["ad_id"] == "ad_test"
        assert body["win_price"] >= 0.0
        assert 0.0 < body["predicted_ctr"] <= 1.0
        assert uuid.UUID(body["impression_id"])
        assert r.headers["X-Impression-Id"] == body["impression_id"]

        # The macro must be substituted, or clicks are untrackable.
        assert "{{CLICK_URL}}" not in body["ad_markup"]
        assert f"/v1/click?id={body['impression_id']}" in body["ad_markup"]
        assert "sig=" in body["ad_markup"] and "exp=" in body["ad_markup"]

    async def test_body_publisher_id_must_match_the_key(self, client):
        """
        The body field is advisory; the key is authoritative. Trusting the body
        would let any publisher log impressions against another's account.
        """
        r = await client.post("/v1/bid", headers=pub_headers(),
                              json=bid_body(publisher_id="pub_someone_else"))
        assert r.status_code == 403

    async def test_empty_inventory_is_a_200_no_fill_not_an_error(self, client):
        """A no-fill is a normal auction outcome. A 4xx/5xx would have publisher
        ad tags reporting errors for an ordinary empty candidate set."""
        inventory._ads = []
        r = await client.post("/v1/bid", json=bid_body(), headers=pub_headers())
        assert r.status_code == 200
        assert r.json()["message"] == "no_fill"
        assert uuid.UUID(r.json()["impression_id"])

    async def test_no_fill_still_logs_an_event(self, client, monkeypatch):
        """Unfilled requests are the denominator of fill rate. Dropping them
        makes fill rate unmeasurable and always 100%."""
        events = Spy()
        monkeypatch.setattr(gateway, "publish_event", events)
        inventory._ads = []

        await client.post("/v1/bid", json=bid_body(), headers=pub_headers())
        await drain()

        assert events.count == 1
        topic, payload = events.calls[0][0]
        assert topic == "impressions"
        assert payload["filled"] is False

    async def test_a_win_persists_the_impression(self, client, monkeypatch):
        """The click path reads destination_url from this row; without it every
        click 404s."""
        saves = Spy()
        monkeypatch.setattr(gateway, "save_impression", saves)

        r = await client.post("/v1/bid", json=bid_body(), headers=pub_headers())
        await drain()

        assert saves.count == 1
        record = saves.calls[0][0][0]
        assert record["impression_id"] == r.json()["impression_id"]
        assert record["destination_url"] == "https://advertiser.example.com/landing"
        assert record["filled"] is True

    async def test_device_targeting_is_honoured_through_the_endpoint(self, client):
        inventory._ads = [make_ad(ad_id="ad_mobile", device="mobile")]
        r = await client.post("/v1/bid", headers=pub_headers(),
                              json=bid_body(device_type="desktop"))
        assert r.json()["message"] == "no_fill"

    @pytest.mark.parametrize("device", ["watch", "", "DESKTOP"])
    async def test_unknown_device_type_is_rejected(self, client, device):
        r = await client.post("/v1/bid", headers=pub_headers(),
                              json=bid_body(device_type=device))
        assert r.status_code == 422

    async def test_a_stale_client_clock_is_rejected(self, client):
        """
        Features are built from server time, but a wildly wrong client clock
        still signals a broken or hostile integration and the field is logged.
        """
        r = await client.post("/v1/bid", headers=pub_headers(),
                              json=bid_body(timestamp_ms=int(time.time() * 1000) - 600_000))
        assert r.status_code == 422


# --- GET /v1/click ---------------------------------------------------------

class TestClick:
    """
    Every rejection returns the same body. The log records which check failed;
    the response never does, because telling a forger whether the expiry or the
    signature failed tells them which half to work on.
    """

    @pytest.fixture(autouse=True)
    def _impression(self, monkeypatch):
        self.impression = impression_row()
        monkeypatch.setattr(gateway, "get_impression",
                            Spy(result=self.impression))
        self.marked = Spy(result=True)
        monkeypatch.setattr(gateway, "mark_clicked", self.marked)

    async def test_a_valid_signature_redirects_to_the_advertiser(self, client):
        sig, exp = sign_click("imp_1")
        r = await client.get(f"/v1/click?id=imp_1&exp={exp}&sig={sig}")
        assert r.status_code == 302
        assert r.headers["location"] == "https://advertiser.example.com/landing"

    async def test_an_unsigned_click_is_refused(self, client):
        r = await client.get("/v1/click?id=imp_1")
        assert r.status_code == 403

    async def test_a_tampered_signature_is_refused(self, client):
        sig, exp = sign_click("imp_1")
        r = await client.get(f"/v1/click?id=imp_1&exp={exp}&sig={sig[:-1]}X")
        assert r.status_code == 403

    async def test_a_signature_for_another_impression_is_refused(self, client):
        """The signature covers impression_id, so it cannot be replayed onto a
        different one."""
        sig, exp = sign_click("imp_other")
        r = await client.get(f"/v1/click?id=imp_1&exp={exp}&sig={sig}")
        assert r.status_code == 403

    async def test_an_expired_url_is_refused(self, client):
        sig, exp = sign_click("imp_1", ttl_seconds=-10)
        r = await client.get(f"/v1/click?id=imp_1&exp={exp}&sig={sig}")
        assert r.status_code == 403

    async def test_every_rejection_looks_identical(self, client):
        sig, exp = sign_click("imp_1")
        expired_sig, expired_exp = sign_click("imp_1", ttl_seconds=-10)

        bodies = set()
        for url in (
            "/v1/click?id=imp_1",
            f"/v1/click?id=imp_1&exp={exp}&sig={sig[:-1]}X",
            f"/v1/click?id=imp_1&exp={expired_exp}&sig={expired_sig}",
        ):
            r = await client.get(url)
            assert r.status_code == 403
            bodies.add(r.text)

        assert len(bodies) == 1, f"rejection reasons leak to the caller: {bodies}"

    async def test_an_unknown_impression_is_404(self, client, monkeypatch):
        monkeypatch.setattr(gateway, "get_impression", Spy(result=None))
        sig, exp = sign_click("imp_missing")
        r = await client.get(f"/v1/click?id=imp_missing&exp={exp}&sig={sig}")
        assert r.status_code == 404

    async def test_a_duplicate_click_redirects_without_double_counting(
        self, client, monkeypatch
    ):
        """
        The user must still land on the advertiser — a browser back-button hit
        is not their problem — but the click may only be counted once.
        """
        events = Spy()
        monkeypatch.setattr(gateway, "publish_event", events)
        monkeypatch.setattr(gateway, "mark_clicked", Spy(result=False))

        sig, exp = sign_click("imp_1")
        r = await client.get(f"/v1/click?id=imp_1&exp={exp}&sig={sig}")
        await drain()

        assert r.status_code == 302
        assert r.headers["location"] == "https://advertiser.example.com/landing"
        assert events.count == 0, "duplicate click was counted"

    async def test_the_destination_comes_from_the_row_not_the_request(self, client):
        """
        The open redirect this endpoint used to have came from a `redirect`
        query parameter. There must be no way to steer the 302 from the URL.
        """
        sig, exp = sign_click("imp_1")
        r = await client.get(
            f"/v1/click?id=imp_1&exp={exp}&sig={sig}"
            "&redirect=https://evil.example&destination_url=https://evil.example"
        )
        assert r.status_code == 302
        assert "evil.example" not in r.headers["location"]

    async def test_a_non_http_destination_is_refused(self, client, monkeypatch):
        """A javascript: or data: destination in the row must not be handed to
        the browser."""
        monkeypatch.setattr(gateway, "get_impression", Spy(
            result=impression_row(destination_url="javascript:alert(1)")))
        sig, exp = sign_click("imp_1")
        r = await client.get(f"/v1/click?id=imp_1&exp={exp}&sig={sig}")
        assert r.status_code == 500
        assert "javascript" not in r.text

    async def test_a_click_url_built_by_the_gateway_verifies(self, client):
        """Closes the loop: whatever build_click_url emits, /v1/click accepts."""
        url = build_click_url("imp_1", base_url="http://testserver")
        r = await client.get(url.replace("http://testserver", ""))
        assert r.status_code == 302


# --- POST /v1/conversion ---------------------------------------------------

class TestConversion:

    async def test_a_conversion_on_your_own_impression_is_recorded(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(gateway, "get_impression", Spy(result=impression_row()))
        saves = Spy()
        monkeypatch.setattr(gateway, "save_conversion", saves)

        r = await client.post("/v1/conversion", headers=pub_headers(), json={
            "impression_id": "imp_1", "conversion_type": "purchase",
            "conversion_value": 42.0})
        await drain()

        assert r.status_code == 200
        assert saves.count == 1

    async def test_another_publishers_impression_is_404_not_403(
        self, client, monkeypatch
    ):
        """
        404 rather than 403 on purpose: 403 would confirm the impression_id
        exists, turning this endpoint into an oracle for enumerating other
        publishers' inventory.
        """
        monkeypatch.setattr(gateway, "get_impression", Spy(
            result=impression_row(publisher_id="pub_someone_else")))
        saves = Spy()
        monkeypatch.setattr(gateway, "save_conversion", saves)

        r = await client.post("/v1/conversion", headers=pub_headers(),
                              json={"impression_id": "imp_1"})
        await drain()

        assert r.status_code == 404
        assert saves.count == 0, "a cross-publisher conversion was written"

    async def test_an_unknown_impression_is_404(self, client, monkeypatch):
        monkeypatch.setattr(gateway, "get_impression", Spy(result=None))
        r = await client.post("/v1/conversion", headers=pub_headers(),
                              json={"impression_id": "nope"})
        assert r.status_code == 404


# --- GET /v1/stats ---------------------------------------------------------

class TestStats:

    async def test_stats_are_scoped_to_the_key(self, client, monkeypatch):
        """
        There is no publisher_id parameter, by design. This asserts the handler
        passes the authenticated id and nothing else — the moment reporting
        takes an id from the caller, every field needs its own authorisation
        check.
        """
        seen = []

        async def fake_stats(publisher_id):
            seen.append(publisher_id)
            return {"publisher_id": publisher_id, "impressions": 1, "clicks": 0,
                    "conversions": 0, "ctr": 0.0, "revenue_usd": 0.0}

        monkeypatch.setattr(gateway, "get_publisher_stats", fake_stats)

        r = await client.get("/v1/stats?publisher_id=pub_someone_else",
                             headers=pub_headers())

        assert r.status_code == 200
        assert seen == ["pub_test"], "a query parameter influenced whose stats were read"


# --- /admin ----------------------------------------------------------------

class TestAdmin:

    async def test_a_wrong_admin_token_is_401(self, client):
        r = await client.post("/admin/publishers",
                              headers={"X-Admin-Token": "wrong"},
                              json={"publisher_id": "pub_new",
                                    "domain": "https://new.example"})
        assert r.status_code == 401

    async def test_a_missing_admin_token_is_401(self, client):
        r = await client.post("/admin/publishers",
                              json={"publisher_id": "pub_new",
                                    "domain": "https://new.example"})
        assert r.status_code == 401

    async def test_a_publisher_key_is_not_an_admin_token(self, client):
        r = await client.post("/admin/publishers",
                              headers={"X-Admin-Token": PUBLISHER_KEY},
                              json={"publisher_id": "pub_new",
                                    "domain": "https://new.example"})
        assert r.status_code == 401

    async def test_the_admin_api_503s_when_no_token_is_configured(self, client):
        """Unset ADMIN_TOKEN must lock the door, not leave it open."""
        object.__setattr__(settings, "admin_token", "")
        try:
            r = await client.post("/admin/publishers",
                                  headers={"X-Admin-Token": "anything"},
                                  json={"publisher_id": "pub_new",
                                        "domain": "https://new.example"})
            assert r.status_code == 503
        finally:
            object.__setattr__(settings, "admin_token", ADMIN_TOKEN)

    async def test_domain_must_carry_a_scheme(self, client):
        """The domain is registered as a CORS origin, and an origin without a
        scheme never matches."""
        r = await client.post("/admin/publishers",
                              headers={"X-Admin-Token": ADMIN_TOKEN},
                              json={"publisher_id": "pub_new",
                                    "domain": "new.example"})
        assert r.status_code == 422

    async def test_creating_a_publisher_returns_a_working_key_once(
        self, client, monkeypatch
    ):
        """
        The regression for the missing `add_origin` import: this endpoint
        returned 500 *after* writing the publisher row and issuing its key,
        leaving a publisher that exists but cannot be reached from a browser.
        """
        monkeypatch.setattr(db, "create_publisher", Spy(result=True))
        monkeypatch.setattr(db, "insert_api_key", Spy(result=True))
        monkeypatch.setattr(gateway, "add_origin", Spy(result=None))

        r = await client.post("/admin/publishers",
                              headers={"X-Admin-Token": ADMIN_TOKEN},
                              json={"publisher_id": "pub_new",
                                    "domain": "https://new.example"})

        assert r.status_code == 201, r.text
        issued = r.json()["api_key"]
        assert issued.startswith("pk_")

        # The key must work on the very first request, not after the next
        # refresh tick.
        probe = await client.post("/v1/bid", json=bid_body(publisher_id="pub_new"),
                                  headers={"X-API-Key": issued})
        assert probe.status_code != 401

    async def test_a_duplicate_publisher_is_409(self, client, monkeypatch):
        monkeypatch.setattr(db, "create_publisher", Spy(result=False))
        r = await client.post("/admin/publishers",
                              headers={"X-Admin-Token": ADMIN_TOKEN},
                              json={"publisher_id": "pub_dupe",
                                    "domain": "https://dupe.example"})
        assert r.status_code == 409


# --- /health ---------------------------------------------------------------

class TestHealth:

    async def test_health_reports_degraded_rather_than_failing(self, client):
        """
        /health must answer while stores are down — it is what the orchestrator
        reads to decide whether the process is worth keeping.
        """
        r = await client.get("/health")
        assert r.status_code == 200

        body = r.json()
        assert body["status"] == "degraded"
        assert body["budgets"] == "NOT ENFORCED"
        assert set(body["stores"]) == {"postgres", "redis", "clickhouse", "kafka"}
        assert body["ctr_model"] == "baseline"

    async def test_health_needs_no_key(self, client):
        assert (await client.get("/health")).status_code == 200
