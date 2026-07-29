# events.py — Kafka/Redpanda event producer.
#
# Everything published here is consumed by ClickHouse via the Kafka-engine
# tables in ml/kafka_sink.sql. Topics:
#
#   impressions  -> ad_impressions   (carries the feature vector)
#   clicks       -> ad_clicks        (the training label, via the join)
#   conversions  -> not yet sunk to ClickHouse
#
# Why publish to Kafka at all rather than writing ClickHouse directly: the bid
# path has a ~20ms budget. A direct write stalls the request whenever ClickHouse
# is compacting, and loses the row entirely if it is down. Redpanda accepts in
# microseconds and holds the message on disk until a consumer catches up.

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .settings import settings

log = logging.getLogger("events")

_producer: Optional[Any] = None
_enabled = False

# Broker outages are usually sustained, and one log line per dropped event
# would bury everything else. Count them and log periodically instead.
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
    # ClickHouse's Kafka engine parses exactly this.
    return json.dumps(payload, default=_default, separators=(",", ":")).encode()


async def init_kafka() -> None:
    """
    Start the producer. Never raises — if Redpanda is unreachable the gateway
    still serves ads, it just stops feeding the training pipeline. Losing
    analytics is survivable; refusing to bid is not.
    """
    global _producer, _enabled
    try:
        from aiokafka import AIOKafkaProducer

        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap,
            value_serializer=_serialize,
            # Batch briefly. At bid volume this cuts round trips substantially
            # for 5ms of added delivery latency, which nothing downstream cares
            # about — these are analytics events, not the bid response.
            linger_ms=settings.kafka_linger_ms,
            compression_type="gzip",
            acks=1,
            request_timeout_ms=settings.kafka_request_timeout_ms,
        )
        await _producer.start()
        _enabled = True
        log.info("Kafka producer connected to %s", settings.kafka_bootstrap)
    except Exception as e:
        # The producer object exists even though start() failed, and aiokafka
        # logs "Unclosed AIOKafkaProducer" at GC if we just drop the reference.
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
            # Flushes buffered messages. Without this, up to linger_ms of
            # events are lost on every deploy.
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

    Uses send() not send_and_wait(): we do not need the broker ack before
    returning, and awaiting it would serialize event publishing behind network
    round trips.
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