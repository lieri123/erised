# tests/test_click_signing.py
#
# /v1/click used to accept a bare impression_id. Every accepted click is a
# training label, and impression_ids are handed to every publisher page inside
# ad_markup — so anyone who scraped one could manufacture a click for whichever
# ad they liked, inflate its measured CTR, and teach the model to prefer it.
#
# These tests cover the forgery paths specifically: tampered ids, tampered
# expiries, truncated signatures, and the separator ambiguity that would let one
# signature validate a different (id, expiry) pair.

import time

import pytest

from adplatform import signing
from adplatform.signing import (
    ClickSignatureError,
    build_click_url,
    sign_click,
    verify_click,
)

IMP = "3bbf60f7-e22e-456b-9bdd-c6d26ba6c10d"


def parse(url: str) -> dict:
    return dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))


# --- round trip -----------------------------------------------------------

def test_freshly_signed_url_verifies():
    sig, exp = sign_click(IMP)
    verify_click(IMP, exp, sig)  # must not raise


def test_build_click_url_is_verifiable():
    q = parse(build_click_url(IMP, base_url="https://ads.example.com"))
    assert q["id"] == IMP
    verify_click(q["id"], int(q["exp"]), q["sig"])


def test_signing_is_deterministic_for_the_same_inputs():
    """Two gateway replicas must mint interchangeable signatures."""
    exp = int(time.time()) + 3600
    assert signing._compute(IMP, exp) == signing._compute(IMP, exp)


# --- forgery --------------------------------------------------------------

def test_signature_does_not_transfer_to_another_impression():
    """The headline attack: reuse one valid signature across scraped ids."""
    sig, exp = sign_click(IMP)
    with pytest.raises(ClickSignatureError):
        verify_click("some-other-impression-id", exp, sig)


def test_expiry_cannot_be_extended():
    """
    The signature covers expiry as well as id. Signing only the id would let an
    attacker keep a scraped URL alive forever by editing exp.
    """
    sig, exp = sign_click(IMP)
    with pytest.raises(ClickSignatureError):
        verify_click(IMP, exp + 86_400, sig)


def test_expired_url_is_rejected():
    sig, exp = sign_click(IMP, ttl_seconds=-1)
    with pytest.raises(ClickSignatureError, match="expired"):
        verify_click(IMP, exp, sig)


def test_absurd_future_expiry_is_rejected():
    """
    Bounds the blast radius if the pepper leaks: a forged URL cannot be minted
    to last for years.
    """
    far = int(time.time()) + 86_400 * 365
    with pytest.raises(ClickSignatureError, match="future"):
        verify_click(IMP, far, signing._compute(IMP, far))


@pytest.mark.parametrize("bad", ["", "x", "AAAAAAAAAAAAAAAAAAAAAA", "not-base64!!"])
def test_garbage_signatures_are_rejected(bad):
    _, exp = sign_click(IMP)
    with pytest.raises(ClickSignatureError):
        verify_click(IMP, exp, bad)


def test_truncated_signature_is_rejected():
    """compare_digest is length-sensitive; a prefix must not pass."""
    sig, exp = sign_click(IMP)
    with pytest.raises(ClickSignatureError):
        verify_click(IMP, exp, sig[:-1])


def test_separator_removes_concatenation_ambiguity():
    """
    Without a separator, ("ab", 1) and ("a", 21) sign the same bytes, so a
    signature minted for one impression validates a different one.
    """
    assert signing._compute("ab", 1) != signing._compute("a", 21)


def test_signature_depends_on_the_pepper(monkeypatch):
    """
    Rotating the pepper must invalidate outstanding signatures — that is the
    emergency lever if a signing key is ever suspected leaked. Signatures
    self-heal within click_url_ttl_seconds as old creatives age out.

    Settings is a frozen dataclass, so the whole object is swapped rather than
    the field mutated.
    """
    import dataclasses

    exp = int(time.time()) + 3600
    original = signing._compute(IMP, exp)

    rotated = dataclasses.replace(signing.settings, api_key_pepper="a-different-pepper")
    monkeypatch.setattr(signing, "settings", rotated)

    assert signing._compute(IMP, exp) != original


# --- shape ----------------------------------------------------------------

def test_signature_is_url_safe():
    """
    The signature is embedded in a URL inside an HTML attribute. Standard base64
    '+' and '/' would need escaping and break naive publisher templates.
    """
    sig, _ = sign_click(IMP)
    assert "+" not in sig and "/" not in sig and "=" not in sig


def test_signature_is_128_bits():
    sig, _ = sign_click(IMP)
    assert len(sig) == 22, "16 bytes base64url, unpadded"


def test_ttl_default_comes_from_settings():
    _, exp = sign_click(IMP)
    expected = int(time.time()) + signing.settings.click_url_ttl_seconds
    assert abs(exp - expected) <= 2
