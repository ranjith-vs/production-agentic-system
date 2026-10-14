"""Structured trace events for the Observability UI: one row per LLM call,
tool call, cache check, or turn boundary, keyed by (thread_id, turn_id) so a
single conversation turn's full flow - planner decision, specialist LLM/tool
calls, judge verdict - can be reconstructed and rendered in order, with token
usage attached to each LLM call. Table structure lives in
schema/007_trace_events.sql, not here."""


async def log_event(
    pool,
    *,
    thread_id: str,
    turn_id: str,
    event_type: str,
    node: str | None = None,
    agent_id: int | None = None,
    model: str | None = None,
    tool_name: str | None = None,
    input_summary: str | None = None,
    output_summary: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    latency_ms: int | None = None,
) -> None:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO trace_events
                    (thread_id, turn_id, event_type, node, agent_id, model, tool_name,
                     input_summary, output_summary, input_tokens, output_tokens, total_tokens, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    thread_id,
                    turn_id,
                    event_type,
                    node,
                    agent_id,
                    model,
                    tool_name,
                    (input_summary or "")[:4000],
                    (output_summary or "")[:4000],
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    latency_ms,
                ),
            )
