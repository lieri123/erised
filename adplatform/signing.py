# signing.py — HMAC signatures for click URLs.

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

    if now > expires_at:
        raise ClickSignatureError(f"expired {now - expires_at}s ago")

    
    max_future = now + settings.click_url_ttl_seconds + 300
    if expires_at > max_future:
        raise ClickSignatureError("expiry too far in the future")

    if not hmac.compare_digest(_compute(impression_id, expires_at), signature):
        raise ClickSignatureError("signature mismatch")
