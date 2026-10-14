import os

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector

GATEWAY_URL = os.environ.get("APISIX_GATEWAY_URL", "http://apisix:9080")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://postgres:postgres@postgres:5432/agentic"
)
# Cosine distance: 0 = identical, ~1 = unrelated. This is a blunt, un-tuned
# threshold picked by hand, not validated against a labeled eval set - that's
# exactly the kind of thing Step 9's judge/eval mechanism should check.
SIMILARITY_THRESHOLD = float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", "0.05"))

embed_client = OpenAI(
    base_url=f"{GATEWAY_URL}/openai/v1",
    api_key="unused-key-apisix-holds-the-real-one",
)


def _embed(text: str) -> list[float]:
    return embed_client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


def lookup_cached_response(prompt: str, agent_id: int | None) -> str | None:
    embedding = _embed(prompt)
    with psycopg.connect(POSTGRES_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # agent_id scopes the cache per agent (NULL = the default,
            # all-specialists agent): two agents with different system
            # prompts / tool access can give legitimately different correct
            # answers to the same-looking question, and an unscoped cache
            # serves one agent's answer to a completely different agent.
            cur.execute(
                """
                SELECT response, embedding <=> %s::vector AS distance
                FROM prompt_cache
                WHERE agent_id IS NOT DISTINCT FROM %s
                ORDER BY distance
                LIMIT 1
                """,
                (embedding, agent_id),
            )
            row = cur.fetchone()
    if row and row[1] <= SIMILARITY_THRESHOLD:
        return row[0]
    return None


def store_cache_entry(prompt: str, response: str, agent_id: int | None) -> None:
    embedding = _embed(prompt)
    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prompt_cache (prompt, embedding, response, agent_id) VALUES (%s, %s, %s, %s)",
                (prompt, embedding, response, agent_id),
            )
