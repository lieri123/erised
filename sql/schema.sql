-- schema.sql — ClickHouse tables for the CTR data flywheel.
 
CREATE TABLE IF NOT EXISTS ad_impressions
(
    impression_id     UUID,
    ts                DateTime64(3, 'UTC'),
 
    publisher_id      LowCardinality(String),
    placement_id      String,
    ad_id             String,
    advertiser_id     LowCardinality(String),
    device_type       LowCardinality(String),
 
    feature_version   UInt16,
    features          Array(Float32),
 
    predicted_ctr     Float32,
    bid_value         Float32,
    win_price         Float32,
 
    is_exploration    UInt8,
    serve_propensity  Float32,   -- P(this ad was served | eligible set)
 
    model_version     LowCardinality(String) DEFAULT 'baseline'
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (ts, ad_id)
TTL toDateTime(ts) + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;
 
 

CREATE TABLE IF NOT EXISTS ad_clicks
(
    impression_id  UUID,
    ts             DateTime64(3, 'UTC'),
    ad_id          String,
    placement_id   String
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (ts, impression_id);
 
 
CREATE TABLE IF NOT EXISTS ctr_agg_pair
(
    day            Date,
    ad_id          String,
    placement_id   String,
    impressions    AggregateFunction(sum, UInt64),
    clicks         AggregateFunction(sum, UInt64)
)
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(day)
ORDER BY (day, ad_id, placement_id);
 
CREATE MATERIALIZED VIEW IF NOT EXISTS ctr_agg_pair_mv
TO ctr_agg_pair
AS SELECT
    toDate(ts)                AS day,
    ad_id,
    placement_id,
    sumState(toUInt64(1))     AS impressions,
    sumState(toUInt64(0))     AS clicks
FROM ad_impressions
GROUP BY day, ad_id, placement_id;
 
CREATE MATERIALIZED VIEW IF NOT EXISTS ctr_agg_pair_clicks_mv
TO ctr_agg_pair
AS SELECT
    toDate(ts)                AS day,
    ad_id,
    placement_id,
    sumState(toUInt64(0))     AS impressions,
    sumState(toUInt64(1))     AS clicks
FROM ad_clicks
GROUP BY day, ad_id, placement_id;
 
 