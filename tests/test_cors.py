# tests/test_cors.py — origin registration and the middleware that reads it.
#
# cors.py was one of three modules with no tests at all, which mattered because
# it is the only thing deciding whose browser can read an authenticated
# response. It is also the module where a mistake is invisible: a domain that
# fails to register produces no error anywhere, just a publisher whose ad tag
# quietly shows nothing.
#
# The specific bug these start from: an Origin header is always
# scheme://host[:port], so a domain stored without a scheme can never match one.
# The admin API rejects those at the door and add_origin raises on them, but
# seed_inventory.py writes publishers with direct SQL and bypassed both — so
# `demo.localhost` sat in the allowed set matching nothing.

from __future__ import annotations

import httpx
import pytest

from adplatform import cors
from adplatform.cors import (
    DEV_ORIGINS,
    REGISTERED_ORIGINS,
    add_origin,
    load_origins_from_db,
)

PUB = "https://publisher.example"


class FakePool:
    def __init__(self, domains):
        self.domains = domains

    async def fetch(self, _query):
        return [{"domain": d} for d in self.domains]


class BrokenPool:
    async def fetch(self, _query):
        raise ConnectionError("postgres went away")


@pytest.fixture(autouse=True)
def clean_origins():
    """REGISTERED_ORIGINS is module-level and shared; restore it around each test."""
    before = set(REGISTERED_ORIGINS)
    yield
    REGISTERED_ORIGINS.clear()
    REGISTERED_ORIGINS.update(before)


# --- registration ----------------------------------------------------------

async def test_a_domain_with_a_scheme_is_registered():
    n = await load_origins_from_db(FakePool([PUB]))

    assert n == 1
    assert PUB in REGISTERED_ORIGINS


async def test_a_domain_without_a_scheme_is_refused():
    """
    The regression. `demo.localhost` can never equal an Origin header, so
    registering it is worse than dropping it: the set reports a domain that is
    in fact unreachable.
    """
    n = await load_origins_from_db(FakePool(["demo.localhost"]))

    assert n == 0
    assert "demo.localhost" not in REGISTERED_ORIGINS


async def test_the_refusal_is_logged_loudly(caplog):
    """Nothing else surfaces this. The publisher just sees an empty ad slot."""
    import logging

    with caplog.at_level(logging.WARNING, logger="cors"):
        await load_origins_from_db(FakePool(["demo.localhost"]))

    messages = [r.getMessage() for r in caplog.records]
    assert any("demo.localhost" in m for m in messages), messages


async def test_one_bad_domain_does_not_cost_the_good_ones():
    n = await load_origins_from_db(FakePool([PUB, "no-scheme.example"]))

    assert n == 1
    assert PUB in REGISTERED_ORIGINS


async def test_a_trailing_slash_is_stripped():
    """An Origin header never carries a path, so `https://x/` would not match."""
    await load_origins_from_db(FakePool(["https://publisher.example/"]))

    assert PUB in REGISTERED_ORIGINS


async def test_dev_origins_survive_a_refresh():
    """load_origins_from_db clears the set; localhost has to come back or the
    demo page stops working after the first refresh tick."""
    await load_origins_from_db(FakePool([PUB]))

    assert DEV_ORIGINS <= REGISTERED_ORIGINS


async def test_a_failed_refresh_keeps_the_previous_set():
    """Stale origins are a rounding error. An empty set is an outage."""
    await load_origins_from_db(FakePool([PUB]))

    n = await load_origins_from_db(BrokenPool())

    assert n == 0
    assert PUB in REGISTERED_ORIGINS


async def test_add_origin_refuses_a_scheme_less_domain():
    with pytest.raises(ValueError):
        await add_origin(None, "no-scheme.example")


# --- the middleware --------------------------------------------------------

@pytest.fixture
async def client():
    """The real middleware over a trivial app, so the test exercises cors.py
    rather than a re-implementation of it."""
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    async def ok(_request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/x", ok, methods=["GET", "OPTIONS"])])
    app.add_middleware(cors.DynamicCORSMiddleware)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://testserver") as c:
        yield c


async def test_a_registered_origin_is_echoed_back(client):
    REGISTERED_ORIGINS.add(PUB)

    r = await client.get("/x", headers={"Origin": PUB})

    assert r.headers["access-control-allow-origin"] == PUB
    assert r.headers["vary"] == "Origin"


async def test_an_unregistered_origin_gets_no_cors_headers(client):
    """Not an error — the request succeeds server-side. The browser is what
    refuses to hand the response to the page."""
    r = await client.get("/x", headers={"Origin": "https://evil.example"})

    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


async def test_the_impression_id_header_is_exposed(client):
    """adtag.js reads X-Impression-Id off the response; without
    Access-Control-Expose-Headers the browser hides it from script."""
    REGISTERED_ORIGINS.add(PUB)

    r = await client.get("/x", headers={"Origin": PUB})

    assert "X-Impression-Id" in r.headers["access-control-expose-headers"]


async def test_preflight_from_a_registered_origin_allows_the_key_header(client):
    REGISTERED_ORIGINS.add(PUB)

    r = await client.request("OPTIONS", "/x", headers={"Origin": PUB})

    assert r.status_code == 204
    assert "X-API-Key" in r.headers["access-control-allow-headers"]


async def test_preflight_from_an_unregistered_origin_allows_nothing(client):
    r = await client.request("OPTIONS", "/x",
                             headers={"Origin": "https://evil.example"})

    assert r.status_code == 204
    assert "access-control-allow-origin" not in r.headers
