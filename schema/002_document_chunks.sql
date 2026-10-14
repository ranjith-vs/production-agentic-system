-- No ivfflat/hnsw index here on purpose: ivfflat trains its cluster lists
-- from whatever rows exist *at CREATE INDEX time* - building it against an
-- empty table (as a fresh DB would have) silently breaks ORDER BY ... LIMIT
-- queries against rows inserted afterwards (they stop being returned at all,
-- with no error). Plain seq-scan cosine distance is exact and plenty fast at
-- this corpus size; add an ANN index later with REINDEX once there's real
-- data loaded, not before.
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_name TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
