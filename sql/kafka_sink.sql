-- kafka_sink.sql — connects events.py's Kafka topics to the training tables.
-- Run AFTER schema.sql.
--
-- ClickHouse's Kafka engine acts as a consumer group. A Kafka-engine table is
-- a queue, not a table: each row is delivered to attached materialized views
-- exactly once and then gone. SELECTing from it directly consumes messages and
-- makes them unavailable to the real consumer, so never query kafka_* tables
-- to "check if data is arriving" — check the MergeTree tables instead.
 
-- ---------------------------------------------------------------------------
-- Impressions
--
-- kafka_skip_broken_messages is deliberately non-zero. One malformed JSON
-- payload with it set to 0 stalls the consumer permanently and you stop
-- ingesting everything. Better to drop a handful of rows and keep the pipeline
-- moving; system.kafka_consumers records what was skipped.
-- ---------------------------------------------------------------------------
 
CREATE TABLE IF NOT EXISTS kafka_impressions
(
    impression_id     String,
    ts                String,
    publisher_id      String,
    placement_id      String,
    ad_id             String,
    advertiser_id     String,
    device_type       String,
    feature_version   UInt16,
    features          Array(Float32),
    predicted_ctr     Float32,
    bid_value         Float32,
    win_price         Float32,
    is_exploration    UInt8,
    serve_propensity  Float32,
    model_version     String
)
ENGINE = Kafka
SETTINGS
    kafka_broker_list         = 'redpanda:9092',
    kafka_topic_list          = 'impressions',
    kafka_group_name          = 'clickhouse_impressions',
    kafka_format              = 'JSONEachRow',
    kafka_num_consumers       = 1,
    kafka_max_block_size      = 65536,
    kafka_skip_broken_messages = 100,
    input_format_skip_unknown_fields = 1;
 
 
CREATE MATERIALIZED VIEW IF NOT EXISTS kafka_impressions_mv
TO ad_impressions
AS SELECT
    toUUIDOrZero(impression_id)              AS impression_id,
    parseDateTime64BestEffortOrZero(ts, 3)   AS ts,
    publisher_id,
    placement_id,
    ad_id,
    advertiser_id,
    device_type,
    feature_version,
    features,
    predicted_ctr,
    bid_value,
    win_price,
    is_exploration,
    serve_propensity,
    model_version
FROM kafka_impressions
-- no_fill events share the topic but carry no ad and no features. Dropping
-- them here keeps ad_impressions meaning "an ad was served", which is what the
-- training query assumes. If you later want to model fill rate, give no_fills
-- their own topic rather than relaxing this filter.
WHERE feature_version > 0
  AND length(features) > 0
  AND ad_id != '';
 
 
-- ---------------------------------------------------------------------------
-- Clicks
-- ---------------------------------------------------------------------------
 
CREATE TABLE IF NOT EXISTS kafka_clicks
(
    impression_id  String,
    ad_id          String,
    placement_id   String,
    clicked_at_ms  Int64
)
ENGINE = Kafka
SETTINGS
    kafka_broker_list         = 'redpanda:9092',
    kafka_topic_list          = 'clicks',
    kafka_group_name          = 'clickhouse_clicks',
    kafka_format              = 'JSONEachRow',
    kafka_skip_broken_messages = 100,
    input_format_skip_unknown_fields = 1;
 
 
CREATE MATERIALIZED VIEW IF NOT EXISTS kafka_clicks_mv
TO ad_clicks
AS SELECT
    toUUIDOrZero(impression_id)        AS impression_id,
    fromUnixTimestamp64Milli(clicked_at_ms) AS ts,
    ad_id,
    placement_id
FROM kafka_clicks
WHERE kafka_clicks.impression_id != '';
 
 
-- ---------------------------------------------------------------------------
-- Verification queries. Run these after your first few bid requests.
-- ---------------------------------------------------------------------------
 
-- Are impressions landing, with features attached?
--   SELECT count(), avg(length(features)), avg(predicted_ctr),
--          countIf(is_exploration = 1) AS explored
--   FROM ad_impressions WHERE ts > now() - INTERVAL 10 MINUTE;
 
-- THE important one: does the join actually match?
-- If clicks > 0 but matched = 0, impression_id is being minted twice again.
--   SELECT
--       (SELECT count() FROM ad_clicks WHERE ts > now() - INTERVAL 1 HOUR) AS clicks,
--       (SELECT count() FROM ad_clicks c
--          WHERE c.ts > now() - INTERVAL 1 HOUR
--            AND c.impression_id IN (SELECT impression_id FROM ad_impressions)
--       ) AS matched;
 
-- Consumer health — nonzero num_rebalance_revocations or a stuck last_poll_time
-- means the consumer is struggling, usually a broker hostname problem.
--   SELECT table, last_poll_time, num_messages_read, last_exception
--   FROM system.kafka_consumers;
 
-- Is the model actually calibrated in production? This is the query to put on a
-- dashboard — offline metrics cannot tell you this.
--   SELECT
--       toStartOfHour(i.ts) AS hour,
--       count() AS impressions,
--       avg(i.predicted_ctr) AS predicted,
--       countIf(c.impression_id != toUUID('00000000-0000-0000-0000-000000000000')) / count() AS actual
--   FROM ad_impressions i
--   LEFT JOIN ad_clicks c ON i.impression_id = c.impression_id
--   WHERE i.ts > now() - INTERVAL 2 DAY AND i.ts < now() - INTERVAL 2 HOUR
--   GROUP BY hour ORDER BY hour;