-- 003_campaign_budgets.sql — put campaign_id into the ClickHouse impression
-- pipeline so budgets can be enforced and reconciled per CAMPAIGN.
--
-- WHY
-- ---
-- daily_budget_usd lives on the campaign row, but budget.py keyed Redis on
-- ad_id. A campaign with three creatives therefore got three independent
-- counters, each measured against the campaign's full budget: 3x overspend,
-- with every counter reporting itself healthy. Five creatives, 5x.
--
-- The enforcement fix is small. The reason this file exists is reconcile(),
-- which rebuilds Redis from ad_impressions every few minutes and is what makes
-- the fail-open read path and fire-and-forget write path safe. Reconciling by
-- campaign requires campaign_id in ad_impressions, and it was never there --
-- campaign_id existed only in Postgres.
--
-- ORDER MATTERS. Apply top to bottom. The Kafka table is dropped and recreated
-- rather than altered because a Kafka engine table's column list is its parse
-- contract; ALTER leaves the running consumer with a stale schema.
--
-- SAFE TO RE-RUN. Every statement is IF EXISTS / IF NOT EXISTS.
--
-- BRIEF CONSUMER PAUSE. Between the DROP and CREATE of kafka_impressions,
-- nothing consumes the topic. Messages are not lost -- Redpanda retains them and
-- the new consumer resumes from the committed offset under the same
-- kafka_group_name. Expect a few seconds of lag, not a gap.

-- 1. Add the column to the destination table.
--
-- Nullable would be more honest for historical rows, but Nullable(String) in an
-- ORDER BY / GROUP BY column costs both storage and query speed for what is a
-- one-time backfill concern. Default '' instead: rows written before this
-- migration carry an empty campaign_id, which reconcile() explicitly skips.
ALTER TABLE ad_impressions
    ADD COLUMN IF NOT EXISTS campaign_id LowCardinality(String) DEFAULT '';

-- 2. The MV must be dropped before the Kafka table it reads from.
DROP VIEW IF EXISTS kafka_impressions_mv;

-- 3. Recreate the Kafka table with campaign_id in its parse schema.
--
-- input_format_skip_unknown_fields = 1 is why the gateway can start emitting
-- campaign_id before this migration runs -- the field is ignored rather than
-- rejected. That is deliberate: it makes the deploy order flexible in one
-- direction only. Emitting the field is safe early; consuming it is not.
DROP TABLE IF EXISTS kafka_impressions;

CREATE TABLE IF NOT EXISTS kafka_impressions
(
    impression_id     String,
    ts                String,
    publisher_id      String,
    placement_id      String,
    ad_id             String,
    campaign_id       String,
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

-- 4. Recreate the MV, carrying campaign_id through.
--
-- The WHERE clause is unchanged and deliberately does NOT require a non-empty
-- campaign_id. An ad served during the window between deploying the gateway and
-- applying this migration is still a real impression with a real cost, and
-- dropping it would corrupt both the training set and the spend total. It lands
-- with campaign_id = '' and reconcile() skips it -- see the note there.
CREATE MATERIALIZED VIEW IF NOT EXISTS kafka_impressions_mv
TO ad_impressions
AS SELECT
    toUUIDOrZero(impression_id)              AS impression_id,
    parseDateTime64BestEffortOrZero(ts, 3)   AS ts,
    publisher_id,
    placement_id,
    ad_id,
    campaign_id,
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
