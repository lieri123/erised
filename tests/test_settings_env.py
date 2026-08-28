# tests/test_settings_env.py
#
# UNSET and EMPTY must mean different things.
#
#   unset  -> nobody configured this, use the default
#   ""     -> somebody configured it to nothing, honour that
#
# The obvious `if not raw: return default` conflates them, and _csv had exactly
# that. It mattered because cors.py documents DEV_ORIGINS="" as the way to drop
# localhost from the CORS allowlist in production: an empty string is falsy, so
# the default came back and every localhost origin stayed allowed. No error, no
# warning, and the operator has no reason to check. Following the documented
# instruction produced the opposite of what it promised.
#
# _int and _bool already got this right, which is what made _csv easy to miss.

import importlib

import pytest


def reload_settings(monkeypatch, **env):
    """Settings reads the environment at import, so reload after changing it."""
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import adplatform.settings as S
    return importlib.reload(S)


# --- the regression -------------------------------------------------------

def test_empty_dev_origins_drops_localhost(monkeypatch):
    """The documented production lever. This is the bug."""
    S = reload_settings(monkeypatch, DEV_ORIGINS="")
    assert S.settings.dev_origins == ()


def test_unset_dev_origins_keeps_the_default(monkeypatch):
    S = reload_settings(monkeypatch, DEV_ORIGINS=None)
    assert "http://localhost:8000" in S.settings.dev_origins


def test_whitespace_only_is_also_empty(monkeypatch):
    """A space used to be the only thing that worked. It must still work."""
    S = reload_settings(monkeypatch, DEV_ORIGINS="   ")
    assert S.settings.dev_origins == ()


def test_csv_splits_and_strips(monkeypatch):
    S = reload_settings(monkeypatch, DEV_ORIGINS="https://a.com, https://b.com ,")
    assert S.settings.dev_origins == ("https://a.com", "https://b.com")


# --- the same trap in the other parsers -----------------------------------

def test_empty_int_falls_back_rather_than_crashing(monkeypatch):
    """int("") raises. An empty value must not take the gateway down at import."""
    S = reload_settings(monkeypatch, INVENTORY_REFRESH_SECONDS="")
    assert S.settings.inventory_refresh_seconds == 60


def test_int_is_read_when_set(monkeypatch):
    S = reload_settings(monkeypatch, INVENTORY_REFRESH_SECONDS="15")
    assert S.settings.inventory_refresh_seconds == 15


def test_settings_are_frozen(monkeypatch):
    """
    Mutating settings at runtime would let one request change another's
    behaviour. Callers that need a variant use dataclasses.replace.
    """
    import dataclasses
    S = reload_settings(monkeypatch)
    with pytest.raises(dataclasses.FrozenInstanceError):
        S.settings.api_key_pepper = "nope"


# Leave the module in its default state for whatever runs next.
@pytest.fixture(autouse=True, scope="module")
def _restore():
    yield
    import adplatform.settings as S
    importlib.reload(S)


# --- the production boot gate ----------------------------------------------
#
# test_empty_dev_origins_drops_localhost above proves the escape hatch works.
# These prove something narrower and more useful: that nothing has to remember
# to use it. validate_for_production runs in the lifespan and aborts boot, so a
# check that is missing here is a misconfiguration that ships.

def _prod(monkeypatch, **env):
    env.setdefault("ENV", "production")
    env.setdefault("API_KEY_PEPPER", "a-real-pepper")
    env.setdefault("ADMIN_TOKEN", "a-real-token")
    env.setdefault("CLICKHOUSE_PASSWORD", "a-real-password")
    env.setdefault("MODEL_ARTIFACT_BACKEND", "s3")
    env.setdefault("MODEL_S3_BUCKET", "a-bucket")
    env.setdefault("BOOTSTRAP_API_KEY", None)
    return reload_settings(monkeypatch, **env)


def test_default_dev_origins_block_production_boot(monkeypatch):
    """
    settings.py documents the danger — "production can ship with an empty list
    rather than permanently trusting localhost" — but documenting it is not
    enforcing it. cors.py re-adds DEV_ORIGINS on every refresh, so shipping the
    defaults means localhost is a trusted CORS origin for the life of the
    deployment, and a page there can read authenticated responses.
    """
    S = _prod(monkeypatch, DEV_ORIGINS=None)

    problems = S.settings.validate_for_production()

    assert any("DEV_ORIGINS" in p for p in problems), problems


def test_empty_dev_origins_pass_production(monkeypatch):
    S = _prod(monkeypatch, DEV_ORIGINS="")
    assert S.settings.validate_for_production() == []


def test_a_named_origin_is_still_refused(monkeypatch):
    """
    The check is "empty", not "no localhost". A real domain belongs in the
    publishers table, where it can be revoked; hard-coding it in the
    environment puts it outside the revocation path entirely.
    """
    S = _prod(monkeypatch, DEV_ORIGINS="https://staging.example")
    assert any("DEV_ORIGINS" in p for p in S.settings.validate_for_production())


def test_development_is_unaffected(monkeypatch):
    """A dev box is allowed to run on defaults; the gate is production-only."""
    S = reload_settings(monkeypatch, ENV="development", DEV_ORIGINS=None)
    assert S.settings.validate_for_production() == []
