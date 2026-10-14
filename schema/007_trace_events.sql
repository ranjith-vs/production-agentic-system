-- One row per LLM call / tool call / turn boundary (see app/agent/tracing.py)
-- - the Observability UI's raw material, keyed by (thread_id, turn_id) so a
-- single turn's full flow can be reconstructed in order.
CREATE TABLE IF NOT EXISTS trace_events (
    id SERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    node TEXT,
    agent_id INT,
    model TEXT,
    tool_name TEXT,
    input_summary TEXT,
    output_summary TEXT,
    input_tokens INT,
    output_tokens INT,
    total_tokens INT,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trace_events_thread_idx ON trace_events (thread_id, id);
