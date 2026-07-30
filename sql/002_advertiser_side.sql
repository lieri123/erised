-- 002_advertiser_side.sql — advertiser-side tables + the servable_ads view.
--
-- Applied by scripts/bootstrap.py AFTER adplatform.db.SCHEMA (publishers,
-- impressions, conversions), because campaigns references advertisers and
-- api_keys references publishers.
--
-- Idempotent: every statement is CREATE ... IF NOT EXISTS / CREATE OR REPLACE,
-- so `make bootstrap` can be re-run safely.

CREATE TABLE IF NOT EXISTS advertisers (
    advertiser_id   TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    contact_email   TEXT,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id       TEXT PRIMARY KEY,
    advertiser_id     TEXT NOT NULL REFERENCES advertisers(advertiser_id),
    name              TEXT NOT NULL,
    daily_budget_usd  DOUBLE PRECISION NOT NULL,
    target_cpm        DOUBLE PRECISION NOT NULL,
    floor_price       DOUBLE PRECISION NOT NULL DEFAULT 0,
    target_device     TEXT NOT NULL DEFAULT 'all',
    -- Stored pre-normalised (lowercased, deduped, sorted) — see
    -- CampaignRequest.normalise_keywords in gateway.py. Scoring reads this
    -- straight off the servable_ads view with no further cleanup.
    target_keywords   JSONB NOT NULL DEFAULT '[]'::jsonb,
    start_date        DATE,
    end_date          DATE,
    status            TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'paused', 'archived')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS campaigns_advertiser_idx ON campaigns (advertiser_id);

CREATE TABLE IF NOT EXISTS ads (
    ad_id             TEXT PRIMARY KEY,
    campaign_id       TEXT NOT NULL REFERENCES campaigns(campaign_id),
    name              TEXT,
    creative_html     TEXT NOT NULL,
    destination_url   TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'paused', 'archived')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ads_campaign_idx ON ads (campaign_id);

-- One row per issued key. owner_type + owner_id together resolve to either a
-- publishers row or an advertisers row — see auth.load_keys_from_db's LEFT
-- JOIN, which is why there is no FK here (a single FK column can't point at
-- two different tables).
CREATE TABLE IF NOT EXISTS api_keys (
    key_id          TEXT PRIMARY KEY,
    owner_type      TEXT NOT NULL CHECK (owner_type IN ('publisher', 'advertiser')),
    owner_id        TEXT NOT NULL,
    key_hash        TEXT NOT NULL UNIQUE,
    key_prefix      TEXT NOT NULL,
    name            TEXT,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    revoked_at      TIMESTAMPTZ,
    last_used_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS api_keys_owner_idx ON api_keys (owner_type, owner_id);

-- The view inventory.load_inventory() queries wholesale on every refresh.
-- Campaign-level fields (cpm, floor, device, keywords, budget) are joined onto
-- each ad row because bidding is per-creative but targeting and budget are set
-- per-campaign — see inventory._row_to_ad's note on budget_id being the
-- campaign id, not the ad id.
CREATE OR REPLACE VIEW servable_ads AS
SELECT
    a.ad_id,
    c.advertiser_id,
    a.creative_html,
    a.destination_url,
    c.target_cpm,
    c.floor_price,
    c.target_device,
    c.target_keywords,
    c.daily_budget_usd,
    c.campaign_id,
    a.created_at
FROM ads a
JOIN campaigns    c ON c.campaign_id    = a.campaign_id
JOIN advertisers  d ON d.advertiser_id  = c.advertiser_id
WHERE a.status = 'active'
  AND c.status = 'active'
  AND d.active = TRUE
  AND (c.start_date IS NULL OR c.start_date <= CURRENT_DATE)
  AND (c.end_date   IS NULL OR c.end_date   >= CURRENT_DATE);
