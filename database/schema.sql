-- business_cards table for PostgreSQL / Supabase.
-- The app creates this automatically on startup (SQLAlchemy create_all); run it
-- manually in the Supabase SQL editor only if the DB user lacks CREATE rights.
CREATE TABLE IF NOT EXISTS business_cards (
    id               BIGSERIAL PRIMARY KEY,
    name             TEXT,
    company_name     TEXT,
    address          TEXT,
    telephone        TEXT,
    mobile_phone     TEXT,
    email            TEXT,
    job_title        TEXT,
    business_type    TEXT,                             -- web-enriched company classification/summary
    business_confidence INTEGER,                      -- 0-100 confidence in web classification
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_business_cards_email     ON business_cards (email);
CREATE INDEX IF NOT EXISTS ix_business_cards_telephone ON business_cards (telephone);
CREATE INDEX IF NOT EXISTS ix_business_cards_mobile_phone ON business_cards (mobile_phone);
CREATE INDEX IF NOT EXISTS ix_business_cards_name      ON business_cards (name);
