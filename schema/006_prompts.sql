-- Planner/judge prompts as data (see app/agent/prompts.py) - read fresh on
-- every turn and editable via the /prompts API with no redeploy.
CREATE TABLE IF NOT EXISTS prompts (
    key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now()
);
