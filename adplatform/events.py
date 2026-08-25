# events.py — Kafka/Redpanda event producer.

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .settings import settings

log = logging.getLogger("events")

_producer: Optional[Any] = None
_enabled = False

_dropped = 0
_DROP_LOG_EVERY = 100


def _default(obj: Any) -> Any:
    """Serialize the types that end up in event payloads but are not JSON-native."""
    import datetime
    import uuid

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if hasattr(obj, "item"):        # numpy scalars from the feature vector
        return obj.item()
    if hasattr(obj, "tolist"):      # numpy arrays
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _serialize(payload: dict) -> bytes:
    # JSONEachRow — one compact JSON object per message, no trailing newline.
    return json.dumps(payload, default=_default, separators=(",", ":")).encode()


def _security_options() -> dict:
    """
    Translate the KAFKA_SECURITY_* settings into aiokafka producer kwargs.

    Returns an empty dict for PLAINTEXT so the compose path is byte-for-byte the
    producer it always was — no ssl module import, no behaviour change on a
    stack that does not need transport security.

    The four protocols map cleanly: PLAINTEXT (nothing), SSL (TLS, no auth),
    SASL_PLAINTEXT (auth in the clear — only defensible inside a VPC), and
    SASL_SSL (both, which is what a managed broker will insist on).
    """
    protocol = settings.kafka_security_protocol
    if protocol == "PLAINTEXT":
        return {}

    if protocol not in {"SSL", "SASL_PLAINTEXT", "SASL_SSL"}:
        raise ValueError(
            f"KAFKA_SECURITY_PROTOCOL={protocol!r} is not one of "
            "PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL"
        )

    options: dict = {"security_protocol": protocol}

    if protocol in {"SSL", "SASL_SSL"}:
        import ssl

        # cafile=None falls back to the system trust store, which is what a
        # managed broker with a public CA needs. A self-signed Redpanda cert
        # needs KAFKA_SSL_CAFILE pointing at the CA that signed it.
        options["ssl_context"] = ssl.create_default_context(
            cafile=settings.kafka_ssl_cafile or None
        )

    if protocol.startswith("SASL"):
        options["sasl_mechanism"] = settings.kafka_sasl_mechanism
        options["sasl_plain_username"] = settings.kafka_sasl_username
        options["sasl_plain_password"] = settings.kafka_sasl_password

    return options


async def init_kafka() -> None:
    """
    Start the producer. If Redpanda is unreachable the gateway
    still serves ads, it just stops feeding the training pipeline.
    """
    global _producer, _enabled
    try:
        from aiokafka import AIOKafkaProducer

        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap,
            value_serializer=_serialize,
            linger_ms=settings.kafka_linger_ms,
            compression_type="gzip",
            acks=1,
            request_timeout_ms=settings.kafka_request_timeout_ms,
            **_security_options(),
        )
        await _producer.start()
        _enabled = True
        log.info("Kafka producer connected to %s (%s)",
                 settings.kafka_bootstrap, settings.kafka_security_protocol)
    except Exception as e:
        if _producer is not None:
            try:
                await _producer.stop()
            except Exception:
                pass
        _producer = None
        _enabled = False
        log.error("Kafka unavailable (%s) — events will be dropped, "
                  "CTR training data will NOT accumulate", e)


async def close_kafka() -> None:
    global _producer, _enabled
    if _producer is not None:
        try:
            await _producer.stop()
            log.info("Kafka producer stopped cleanly")
        except Exception:
            log.exception("error stopping Kafka producer")
    _producer = None
    _enabled = False


async def publish_event(topic: str, payload: dict) -> None:
    """
    Fire one event. Called via spawn(), so exceptions are logged by the task
    registry rather than surfacing to the request.
    """
    global _dropped

    if not _enabled or _producer is None:
        _dropped += 1
        if _dropped % _DROP_LOG_EVERY == 1:
            log.warning("Kafka down — %d events dropped so far", _dropped)
        return

    try:
        await _producer.send(topic, payload)
    except Exception:
        log.exception("failed to publish to topic=%s", topic)


def is_connected() -> bool:
    """For /health."""
    return _enabled


def dropped_count() -> int:
    return _dropped