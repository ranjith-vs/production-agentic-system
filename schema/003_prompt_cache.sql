CREATE TABLE IF NOT EXISTS prompt_cache (
    id SERIAL PRIMARY KEY,
    prompt TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    response TEXT NOT NULL,
    agent_id INT,
    created_at TIMESTAMPTZ DEFAULT now()
);
-- agent_id added after this table already shipped - ADD COLUMN IF NOT EXISTS
-- so this stays idempotent against a DB from before that.
ALTER TABLE prompt_cache ADD COLUMN IF NOT EXISTS agent_id INT;
