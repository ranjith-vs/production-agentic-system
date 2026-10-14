CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    use_general_tools BOOLEAN NOT NULL DEFAULT true,
    use_research_tools BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);
