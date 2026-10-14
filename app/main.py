import json
import os
import uuid
from contextlib import asynccontextmanager

from observability import init_observability

# Must run before anything imports langchain/langgraph: instrumentation
# patches the base Runnable classes, so any chat model or tool invoked
# afterwards - including inside nested subgraphs - gets traced automatically.
init_observability()

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from agent.graph import build_graph_builder
from agent.prompts import PROMPTS_SCHEMA, list_prompts, seed_prompts, set_prompt
from tasks import celery_app, ingest_document

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://postgres:postgres@postgres:5432/agentic"
)

#  No ivfflat/hnsw index here on purpose: ivfflat trains its cluster lists
#  from whatever rows exist *at CREATE INDEX time* - building it against an
#  empty table (as this startup hook would, on a fresh DB) silently breaks
#  ORDER BY ... LIMIT queries against rows inserted afterwards (they stop
#  being returned at all, with no error). Plain seq-scan cosine distance is
#  exact and plenty fast at this corpus size; add an ANN index later with
#  REINDEX once there's real data loaded, not before.
DOCUMENT_CHUNKS_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_name TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS prompt_cache (
    id SERIAL PRIMARY KEY,
    prompt TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    response TEXT NOT NULL,
    agent_id INT,
    created_at TIMESTAMPTZ DEFAULT now()
);
-- agent_id added after prompt_cache already shipped (Step 10) - ADD COLUMN
-- IF NOT EXISTS so this stays idempotent against a DB from before that.
ALTER TABLE prompt_cache ADD COLUMN IF NOT EXISTS agent_id INT;
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    use_general_tools BOOLEAN NOT NULL DEFAULT true,
    use_research_tools BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);
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
""" + PROMPTS_SCHEMA


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncConnectionPool(
        conninfo=POSTGRES_URL,
        max_size=10,
        kwargs={"autocommit": True},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        async with pool.connection() as conn:
            await conn.execute(DOCUMENT_CHUNKS_SCHEMA)
        await seed_prompts(pool)
        # Built after seeding: plan_and_run/judge_reply read the planner/judge
        # prompts from this same pool on every turn (see agent/prompts.py), so
        # editing a row in `prompts` changes behavior without a redeploy.
        graph_builder = await build_graph_builder(pool)
        app.state.graph = graph_builder.compile(checkpointer=checkpointer)
        app.state.pg_pool = pool
        yield


app = FastAPI(title="production-agentic-system", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    agent_id: int | None = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str


class AgentCreate(BaseModel):
    name: str
    system_prompt: str = ""
    use_general_tools: bool = True
    use_research_tools: bool = True


class Agent(AgentCreate):
    id: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agents", response_model=Agent)
async def create_agent(payload: AgentCreate, request: Request):
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO agents (name, system_prompt, use_general_tools, use_research_tools)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    payload.name,
                    payload.system_prompt,
                    payload.use_general_tools,
                    payload.use_research_tools,
                ),
            )
            (agent_id,) = await cur.fetchone()
    return Agent(id=agent_id, **payload.model_dump())


@app.get("/agents", response_model=list[Agent])
async def list_agents(request: Request):
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, name, system_prompt, use_general_tools, use_research_tools
                FROM agents ORDER BY id
                """
            )
            rows = await cur.fetchall()
    return [
        Agent(id=r[0], name=r[1], system_prompt=r[2], use_general_tools=r[3], use_research_tools=r[4])
        for r in rows
    ]


class PromptUpdate(BaseModel):
    content: str


@app.get("/prompts")
async def get_prompts(request: Request):
    return await list_prompts(request.app.state.pg_pool)


@app.put("/prompts/{key}")
async def update_prompt(key: str, payload: PromptUpdate, request: Request):
    await set_prompt(request.app.state.pg_pool, key, payload.content)
    return {"key": key, "content": payload.content}


async def _resolve_agent(agent_id: int | None, request: Request) -> dict:
    if agent_id is None:
        return {}
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT system_prompt, use_general_tools, use_research_tools
                FROM agents WHERE id = %s
                """,
                (agent_id,),
            )
            row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"agent {agent_id} not found")
    system_prompt, use_general, use_research = row
    allowed = [n for n, enabled in [("general", use_general), ("research", use_research)] if enabled]
    return {
        "system_prompt": system_prompt,
        "allowed_specialists": allowed or ["general"],
        "agent_id": agent_id,
    }


async def _touch_thread(
    thread_id: str, agent_id: int | None, last_message: str, last_response: str, request: Request
) -> None:
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO chat_threads (thread_id, agent_id, last_message, last_response, updated_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (thread_id) DO UPDATE SET
                    agent_id = EXCLUDED.agent_id,
                    last_message = EXCLUDED.last_message,
                    last_response = EXCLUDED.last_response,
                    updated_at = now()
                """,
                (thread_id, agent_id, last_message, last_response),
            )


class ThreadSummary(BaseModel):
    thread_id: str
    agent_id: int | None
    agent_name: str | None
    last_message: str | None
    last_response: str | None
    updated_at: str


@app.get("/threads", response_model=list[ThreadSummary])
async def list_threads(request: Request):
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT t.thread_id, t.agent_id, a.name, t.last_message, t.last_response, t.updated_at
                FROM chat_threads t
                LEFT JOIN agents a ON a.id = t.agent_id
                ORDER BY t.updated_at DESC
                """
            )
            rows = await cur.fetchall()
    return [
        ThreadSummary(
            thread_id=r[0],
            agent_id=r[1],
            agent_name=r[2],
            last_message=r[3],
            last_response=r[4],
            updated_at=r[5].isoformat(),
        )
        for r in rows
    ]


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")
    task = ingest_document.delay(file.filename, text)
    return {"task_id": task.id, "document_name": file.filename}


@app.get("/documents/status/{task_id}")
def document_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return {"state": result.state, "result": result.result if result.ready() else None}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    agent_overrides = await _resolve_agent(req.agent_id, request)
    result = await request.app.state.graph.ainvoke(
        {"messages": [("user", req.message)], **agent_overrides}, config
    )
    response = result["messages"][-1].content
    await _touch_thread(thread_id, req.agent_id, req.message, response, request)
    return ChatResponse(response=response, thread_id=thread_id)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = request.app.state.graph
    agent_overrides = await _resolve_agent(req.agent_id, request)

    async def event_generator():
        yield _sse("thread", {"thread_id": thread_id})
        saw_token = False
        async for message_chunk, _metadata in graph.astream(
            {"messages": [("user", req.message)], **agent_overrides},
            config,
            stream_mode="messages",
        ):
            if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                saw_token = True
                yield _sse("token", {"content": message_chunk.content})
        if not saw_token:
            # A node (e.g. a semantic cache hit) returned a complete message
            # directly instead of streaming it token-by-token from a model
            # call - stream_mode="messages" only surfaces the latter, so fall
            # back to the final state rather than silently emitting nothing.
            state = await graph.aget_state(config)
            final_content = state.values["messages"][-1].content
            yield _sse("token", {"content": final_content})
        else:
            state = await graph.aget_state(config)
            final_content = state.values["messages"][-1].content
        await _touch_thread(thread_id, req.agent_id, req.message, final_content, request)
        yield _sse("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _serialize_message(m) -> dict:
    return {"type": type(m).__name__, "content": m.content, "name": getattr(m, "name", None)}


class CheckpointSummary(BaseModel):
    checkpoint_id: str
    step: int | None
    num_messages: int
    last_message: dict | None
    agent_id: int | None


@app.get("/threads/{thread_id}/checkpoints", response_model=list[CheckpointSummary])
async def list_checkpoints(thread_id: str, request: Request):
    """Every node transition for this thread, oldest first - one row per
    checkpoint LangGraph wrote, not one row per chat turn. Debugging a single
    turn usually means looking at several consecutive rows here."""
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}
    out = []
    async for snap in graph.checkpointer.alist(config):
        values = snap.checkpoint.get("channel_values", {})
        messages = values.get("messages", [])
        out.append(
            CheckpointSummary(
                checkpoint_id=snap.config["configurable"]["checkpoint_id"],
                step=snap.metadata.get("step"),
                num_messages=len(messages),
                last_message=_serialize_message(messages[-1]) if messages else None,
                agent_id=values.get("agent_id"),
            )
        )
    out.reverse()  # alist() yields newest-first; oldest-first reads like a transcript
    return out


@app.get("/threads/{thread_id}/checkpoints/{checkpoint_id}")
async def get_checkpoint(thread_id: str, checkpoint_id: str, request: Request):
    """Full state frozen at one checkpoint: every message so far, plus
    whatever agent config (system_prompt/allowed_specialists) was active."""
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    snap = await graph.aget_state(config)
    if snap is None or not snap.values:
        raise HTTPException(status_code=404, detail="checkpoint not found")
    return {
        "checkpoint_id": checkpoint_id,
        "next": list(snap.next),
        "messages": [_serialize_message(m) for m in snap.values.get("messages", [])],
        "system_prompt": snap.values.get("system_prompt"),
        "allowed_specialists": snap.values.get("allowed_specialists"),
        "agent_id": snap.values.get("agent_id"),
    }


class ReplayRequest(BaseModel):
    checkpoint_id: str
    message: str | None = None


@app.post("/threads/{thread_id}/replay")
async def replay(thread_id: str, payload: ReplayRequest, request: Request):
    """Rewind to `checkpoint_id` and either continue forward with no new
    input (re-runs whatever nodes come after it, fresh LLM calls - useful for
    "would the judge call it differently now") or fork with `message` (a new
    branch from that historical point, same thread_id). Either way this
    becomes the new "current" state for the thread on plain /chat calls with
    no checkpoint_id - the branch you forked from is not deleted, just no
    longer the default; fetch it later by its own checkpoint_id."""
    graph = request.app.state.graph
    fork_config = {"configurable": {"thread_id": thread_id, "checkpoint_id": payload.checkpoint_id}}
    existing = await graph.aget_state(fork_config)
    if existing is None or not existing.values:
        raise HTTPException(status_code=404, detail="checkpoint not found")

    input_ = {"messages": [("user", payload.message)]} if payload.message else None
    result = await graph.ainvoke(input_, fork_config)

    new_head = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return {
        "response": result["messages"][-1].content,
        "replayed_from_checkpoint_id": payload.checkpoint_id,
        "new_checkpoint_id": new_head.config["configurable"]["checkpoint_id"],
    }


# Mounted last and at "/" on purpose: Starlette checks explicit routes (all
# of the above) before falling through to a mount, so this only serves
# whatever path none of the API routes already claimed.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
