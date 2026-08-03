# signing.py — HMAC signatures for click URLs.
#
# THE PROBLEM
# -----------
# /v1/click took only an impression_id. Anyone who saw one click URL could
# replay it, and — because impression_ids appear in ad_markup that we hand to
# every publisher page — that is not a hypothetical. Each accepted click is a
# training label. Forged clicks therefore:
#
#   1. inflate the measured CTR of whichever ad the attacker likes,
#   2. teach the CTR model that ad is good, so it wins more auctions,
#   3. and, once conversions are billed, cost the advertiser real money.
#
# mark_clicked's atomic check-and-set already stops the SAME id being counted
# twice, so the loss is bounded at one forged click per impression_id observed.
# That is still enough to poison a model: an attacker who scrapes ids across a
# publisher's inventory can click every one of them.
#
# WHAT THIS DOES AND DOES NOT SOLVE
# ---------------------------------
# Signing proves a click URL was MINTED BY US and has not been tampered with or
# replayed after expiry. It does not prove a human clicked it. A bot that loads
# the page, receives a legitimately signed URL and fetches it is indistinguishable
# from a user here — that is invalid-traffic detection, a much harder problem,
# and it is not solved in this file. Do not read a valid signature as "real
# click"; read it as "this URL came from us, recently".
#
# DESIGN NOTES
# ------------
# * The signature covers impression_id AND expiry together. Signing only the id
#   would let an attacker extend the window by editing the expiry.
# * Truncated to 16 bytes / 22 base64url chars. A full SHA-256 is 43 chars and
#   the URL is embedded in every creative; 128 bits is far beyond what a
#   forger can brute-force against a server that logs failures.
# * compare_digest, not ==. String comparison short-circuits on first mismatch,
#   which leaks how many leading bytes were right and turns forgery into a
#   byte-at-a-time search.
# * Reuses settings.api_key_pepper as the key. One secret to manage, and its
#   rotation semantics are already understood: rotating invalidates outstanding
#   signatures, which self-heals within click_url_ttl_seconds. That is a
#   deliberate trade — a separate CLICK_SIGNING_KEY would be cleaner but adds a
#   secret nobody remembers to set, and an unset secret is worse than a shared
#   one.

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
from typing import Optional

from .settings import settings

log = logging.getLogger("signing")

SIG_BYTES = 16


class ClickSignatureError(Exception):
    """Raised with a human-readable reason. Do NOT return the reason to the
    caller — 'expired' vs 'bad signature' tells a forger which half to fix."""


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _compute(impression_id: str, expires_at: int) -> str:
    # The separator matters. Without it, ("ab", 1) and ("a", 21) produce the
    # same signed message, so a signature for one validates the other.
    message = f"{impression_id}:{expires_at}".encode()
    digest = hmac.new(
        settings.api_key_pepper.encode(), message, hashlib.sha256
    ).digest()
    return _b64(digest[:SIG_BYTES])


def sign_click(impression_id: str, ttl_seconds: Optional[int] = None) -> tuple[str, int]:
    """Returns (signature, expires_at_unix). Called once per winning bid."""
    ttl = ttl_seconds if ttl_seconds is not None else settings.click_url_ttl_seconds
    expires_at = int(time.time()) + ttl
    return _compute(impression_id, expires_at), expires_at


def build_click_url(impression_id: str, base_url: Optional[str] = None) -> str:
    """The signed URL substituted into {{CLICK_URL}} in the creative."""
    base = (base_url or settings.public_base_url).rstrip("/")
    sig, exp = sign_click(impression_id)
    return f"{base}/v1/click?id={impression_id}&exp={exp}&sig={sig}"


def verify_click(impression_id: str, expires_at: int, signature: str) -> None:
    """Returns None if valid, raises ClickSignatureError otherwise."""
    now = int(time.time())

    # Expiry first: it is the cheap check, and a stale-but-authentic URL is the
    # common case (a user opens a tab and clicks an hour later), not an attack.
    if now > expires_at:
        raise ClickSignatureError(f"expired {now - expires_at}s ago")

    # An expiry far in the future is a tampering attempt, and rejecting it
    # bounds the damage if the pepper ever leaks. Allow a little slack for
    # clock skew between whichever gateway signed and whichever verifies.
    max_future = now + settings.click_url_ttl_seconds + 300
    if expires_at > max_future:
        raise ClickSignatureError("expiry too far in the future")

    if not hmac.compare_digest(_compute(impression_id, expires_at), signature):
        raise ClickSignatureError("signature mismatch")
