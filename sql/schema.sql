-- schema.sql — ClickHouse tables for the CTR data flywheel.
-- Run once against your analytics cluster.
 
-- ---------------------------------------------------------------------------
-- 1. Impression log — written at serve time, one row per served ad.
--
-- The `features` array is the exact vector that was fed to the model. This is
-- the anti-skew guarantee: training never recomputes features from raw request
-- data, it replays what actually happened. If you ever remove this column and
-- reconstruct features later, you have reintroduced the bug this whole design
-- exists to prevent.
-- ---------------------------------------------------------------------------
 
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
 
    -- Exploration bookkeeping. Without these two columns you cannot correct
    -- for the fact that your own past ranking decided what data you collected.
    is_exploration    UInt8,
    serve_propensity  Float32,   -- P(this ad was served | eligible set)
 
    model_version     LowCardinality(String) DEFAULT 'baseline'
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (ts, ad_id)
TTL toDateTime(ts) + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;
 
 
-- ---------------------------------------------------------------------------
-- 2. Click log. impression_id is the join key, so your click endpoint must
--    carry it through the redirect (signed, so advertisers cannot forge clicks).
-- ---------------------------------------------------------------------------
 
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
 
 
-- ---------------------------------------------------------------------------
-- 3. Rolling CTR aggregates, read by the serving process into CtrStats.
--
-- AggregatingMergeTree so this stays cheap as volume grows. Refreshed into
-- process memory every few minutes by a background task, never queried on the
-- hot path.
-- ---------------------------------------------------------------------------
 
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
 
 
-- Query the serving process runs on its refresh loop (last 30 days):
--
--   SELECT ad_id, placement_id,
--          sumMerge(impressions) AS imps,
--          sumMerge(clicks)      AS clicks
--   FROM ctr_agg_pair
--   WHERE day >= today() - 30
--   GROUP BY ad_id, placement_id;
 
 
-- ---------------------------------------------------------------------------
-- 4. Training-set query. This is the one the training job runs.
--
-- Two things here are easy to get wrong:
--
--   (a) ATTRIBUTION WINDOW in the JOIN condition, not the WHERE clause.
--       Putting `c.ts <= i.ts + INTERVAL 1 HOUR` in WHERE turns the LEFT JOIN
--       into an INNER JOIN and silently deletes every negative example. Your
--       training set becomes 100% clicks and the model learns to predict 1.0.
--
--   (b) The `i.ts < now() - INTERVAL 2 HOUR` cutoff. An impression served 30
--       seconds ago has not had time to be clicked. Labelling it 0 teaches the
--       model that recent traffic never converts. The cutoff must be at least
--       as long as your attribution window.
-- ---------------------------------------------------------------------------
 
-- :start_ts, :end_ts, :feature_version are bound by train_ctr.py
--
-- SELECT
--     i.ts                    AS ts,
--     i.features              AS features,
--     i.serve_propensity      AS serve_propensity,
--     i.is_exploration        AS is_exploration,
--     if(c.impression_id IS NULL, 0, 1) AS clicked
-- FROM ad_impressions AS i
-- LEFT JOIN ad_clicks AS c
--     ON  i.impression_id = c.impression_id
--     AND c.ts >= i.ts
--     AND c.ts <= i.ts + INTERVAL 1 HOUR
-- WHERE i.ts >= :start_ts
--   AND i.ts <  :end_ts
--   AND i.ts <  now() - INTERVAL 2 HOUR
--   AND i.feature_version = :feature_version
--   AND length(i.features) = :n_features
-- ORDER BY i.ts ASC;