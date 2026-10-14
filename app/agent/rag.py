import os

import psycopg
from langchain_core.tools import tool
from openai import OpenAI
from pgvector.psycopg import register_vector

GATEWAY_URL = os.environ.get("APISIX_GATEWAY_URL", "http://apisix:9080")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://postgres:postgres@postgres:5432/agentic"
)

embed_client = OpenAI(
    base_url=f"{GATEWAY_URL}/openai/v1",
    api_key="unused-key-apisix-holds-the-real-one",
)


@tool
def search_documents(query: str) -> str:
    """Search previously uploaded documents for passages relevant to the query."""
    embedding = embed_client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    with psycopg.connect(POSTGRES_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_name, content
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT 4
                """,
                (embedding,),
            )
            rows = cur.fetchall()
    if not rows:
        return "No relevant documents found."
    return "\n\n".join(f"[{name}] {content}" for name, content in rows)
