-- LangGraph's checkpointer already has every checkpoint for a thread, but
-- not "list me every thread" - its tables aren't meant to be queried that
-- way directly. This is just an index for the UI's history view: one row
-- per thread, upserted on every /chat call, so "which threads exist and
-- which agent did they last use" is a cheap query instead of a checkpoint
-- table scan.
CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id TEXT PRIMARY KEY,
    agent_id INT,
    last_message TEXT,
    last_response TEXT,
    updated_at TIMESTAMPTZ DEFAULT now()
);
