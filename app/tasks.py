import os

import psycopg
from celery import Celery
from openai import OpenAI
from pgvector.psycopg import register_vector

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://postgres:postgres@postgres:5432/agentic"
)
GATEWAY_URL = os.environ.get("APISIX_GATEWAY_URL", "http://apisix:9080")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

# Embeddings don't fit the ai-proxy route's chat-completions schema either, so
# this goes through the same "openai-raw" passthrough route as tool calling.
embed_client = OpenAI(
    base_url=f"{GATEWAY_URL}/openai/v1",
    api_key="unused-key-apisix-holds-the-real-one",
)


def _chunk_text(text: str, size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


@celery_app.task(name="ingest_document")
def ingest_document(document_name: str, text: str) -> dict:
    chunks = _chunk_text(text)
    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                resp = embed_client.embeddings.create(model=EMBED_MODEL, input=[chunk])
                embedding = resp.data[0].embedding
                cur.execute(
                    """
                    INSERT INTO document_chunks (document_name, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (document_name, i, chunk, embedding),
                )
    return {"document_name": document_name, "chunks": len(chunks)}
