"""Prompt store: planner/judge prompts live in Postgres (`prompts` table, see
schema/006_prompts.sql), not hardcoded in graph.py/judge.py, so editing one
doesn't need a redeploy. Each default below is only a seed - the DB row is
the source of truth afterwards."""

DEFAULT_PROMPTS = {
    "planner": """You are a planner deciding which specialist agent(s) should handle the user's
latest message:
- 'general': everyday questions, arithmetic, the current time (has a calculator and a clock tool).
- 'research': questions about previously uploaded documents / the knowledge base (has a document
  search tool).

Return the agents needed, in the order they should run. Most messages need only one. Use both
only when the message genuinely needs both (e.g. "look up X in the docs, then add 5 to it").""",
    "judge": """You are a governance judge reviewing an AI assistant's draft reply before it is
sent to the user. Check:
- Relevance: does the draft actually address what the user asked?
- Safety: no instructions for violence, illegal acts, or other clearly disallowed content.

Be lenient on relevance (a reasonable attempt passes) and strict on safety. Give a one-sentence
reason either way.""",
}


async def seed_prompts(pool) -> None:
    """Insert each default the first time it's missing; never overwrites an
    existing row, so an edit made through the API survives app restarts."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            for key, content in DEFAULT_PROMPTS.items():
                await cur.execute(
                    "INSERT INTO prompts (key, content) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
                    (key, content),
                )


async def get_prompt(pool, key: str) -> str:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT content FROM prompts WHERE key = %s", (key,))
            row = await cur.fetchone()
    return row[0] if row else DEFAULT_PROMPTS.get(key, "")


async def list_prompts(pool) -> dict[str, str]:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT key, content FROM prompts ORDER BY key")
            rows = await cur.fetchall()
    return {key: content for key, content in rows}


async def set_prompt(pool, key: str, content: str) -> None:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO prompts (key, content, updated_at) VALUES (%s, %s, now())
                ON CONFLICT (key) DO UPDATE SET content = EXCLUDED.content, updated_at = now()
                """,
                (key, content),
            )
