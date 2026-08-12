-- kafka_sink.sql — connects events.py's Kafka topics to the training tables.

 
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

WHERE feature_version > 0
  AND length(features) > 0
  AND ad_id != '';
 
 
-- Clicks
 
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
 