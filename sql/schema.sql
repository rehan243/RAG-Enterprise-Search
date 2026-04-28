-- Document store + observability. Partition search_logs monthly in prod (cron job).

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_key    TEXT NOT NULL UNIQUE,
    title           TEXT,
    source_uri      TEXT,
    mime_type       TEXT,
    language        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    owner_tenant_id UUID NOT NULL,
    deleted_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents (owner_tenant_id) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index     INT  NOT NULL,
    content         TEXT NOT NULL,
    token_count     INT,
    embedding_model TEXT,
    checksum        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);

CREATE TABLE IF NOT EXISTS search_logs (
    id           BIGSERIAL,
    tenant_id    UUID NOT NULL,
    query_text   TEXT NOT NULL,
    latency_ms   INT,
    hits_returned INT,
    trace_id     TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

CREATE TABLE IF NOT EXISTS search_logs_default PARTITION OF search_logs
    FOR VALUES FROM ('2020-01-01') TO ('2030-01-01');

CREATE INDEX IF NOT EXISTS idx_search_logs_tenant_time ON search_logs (tenant_id, created_at DESC);

CREATE TABLE IF NOT EXISTS search_analytics_daily (
    day          DATE NOT NULL,
    tenant_id    UUID NOT NULL,
    queries      BIGINT NOT NULL DEFAULT 0,
    zero_result  BIGINT NOT NULL DEFAULT 0,
    p95_latency_ms INT,
    PRIMARY KEY (day, tenant_id)
);
