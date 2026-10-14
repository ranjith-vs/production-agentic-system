import json
import os
import uuid
from contextlib import asynccontextmanager

from observability import init_observability

# Must run before anything imports langchain/langgraph: instrumentation
# patches the base Runnable classes, so any chat model or tool invoked
# afterwards - including inside nested subgraphs - gets traced automatically.
init_observability()

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.errors import UniqueViolation
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from agent.graph import build_graph_builder
from agent.prompts import list_prompts, seed_prompts, set_prompt
from auth import create_access_token, get_current_user, hash_password, verify_password
from db_schema import apply_schema
from tasks import celery_app, ingest_document

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://postgres:postgres@postgres:5432/agentic"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncConnectionPool(
        conninfo=POSTGRES_URL,
        max_size=10,
        kwargs={"autocommit": True},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        await apply_schema(pool)
        await seed_prompts(pool)
        # Built after seeding: plan_and_run/judge_reply read the planner/judge
        # prompts from this same pool on every turn (see agent/prompts.py), so
        # editing a row in `prompts` changes behavior without a redeploy.
        graph_builder = await build_graph_builder(pool)
        app.state.graph = graph_builder.compile(checkpointer=checkpointer)
        app.state.pg_pool = pool
        yield


app = FastAPI(title="production-agentic-system", lifespan=lifespan)

# Everything except /health and /auth/* requires a valid bearer token - see
# auth.py. The static UI shell itself is unauthenticated (it's just HTML/JS;
# it can't do anything without a token) and shows a login screen until one
# is obtained via /auth/login or /auth/register.
api = APIRouter(dependencies=[Depends(get_current_user)])


@app.get("/health")
def health():
    return {"status": "ok"}


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


@app.post("/auth/register", response_model=TokenResponse)
async def register(payload: RegisterRequest, request: Request):
    if not payload.username.strip() or len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="username required, password must be >= 8 chars")
    pw_hash = hash_password(payload.password)
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            try:
                await cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
                    (payload.username, pw_hash),
                )
            except UniqueViolation:
                raise HTTPException(status_code=409, detail="username already taken")
            (user_id,) = await cur.fetchone()
    token = create_access_token(user_id, payload.username)
    return TokenResponse(access_token=token, username=payload.username)


@app.post("/auth/login", response_model=TokenResponse)
async def login(payload: LoginRequest, request: Request):
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id, password_hash FROM users WHERE username = %s", (payload.username,)
            )
            row = await cur.fetchone()
    if row is None or not verify_password(payload.password, row[1]):
        raise HTTPException(status_code=401, detail="invalid username or password")
    token = create_access_token(row[0], payload.username)
    return TokenResponse(access_token=token, username=payload.username)


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


@api.post("/agents", response_model=Agent)
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


@api.get("/agents", response_model=list[Agent])
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


@api.get("/prompts")
async def get_prompts(request: Request):
    return await list_prompts(request.app.state.pg_pool)


@api.put("/prompts/{key}")
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


async def _log_turn_boundary(
    request: Request, thread_id: str, turn_id: str, event_type: str, content: str
) -> None:
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO trace_events (thread_id, turn_id, event_type, output_summary)
                VALUES (%s, %s, %s, %s)
                """,
                (thread_id, turn_id, event_type, content[:4000]),
            )


class ThreadSummary(BaseModel):
    thread_id: str
    agent_id: int | None
    agent_name: str | None
    last_message: str | None
    last_response: str | None
    updated_at: str


@api.get("/threads", response_model=list[ThreadSummary])
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


@api.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")
    task = ingest_document.delay(file.filename, text)
    return {"task_id": task.id, "document_name": file.filename}


@api.get("/documents/status/{task_id}")
def document_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return {"state": result.state, "result": result.result if result.ready() else None}


@api.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    thread_id = req.thread_id or str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    agent_overrides = await _resolve_agent(req.agent_id, request)
    await _log_turn_boundary(request, thread_id, turn_id, "user_message", req.message)
    result = await request.app.state.graph.ainvoke(
        {"messages": [("user", req.message)], "thread_id": thread_id, "turn_id": turn_id, **agent_overrides},
        config,
    )
    response = result["messages"][-1].content
    await _log_turn_boundary(request, thread_id, turn_id, "assistant_message", response)
    await _touch_thread(thread_id, req.agent_id, req.message, response, request)
    return ChatResponse(response=response, thread_id=thread_id)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@api.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    thread_id = req.thread_id or str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = request.app.state.graph
    agent_overrides = await _resolve_agent(req.agent_id, request)

    async def event_generator():
        yield _sse("thread", {"thread_id": thread_id})
        await _log_turn_boundary(request, thread_id, turn_id, "user_message", req.message)
        saw_token = False
        async for message_chunk, _metadata in graph.astream(
            {
                "messages": [("user", req.message)],
                "thread_id": thread_id,
                "turn_id": turn_id,
                **agent_overrides,
            },
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
        await _log_turn_boundary(request, thread_id, turn_id, "assistant_message", final_content)
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


@api.get("/threads/{thread_id}/checkpoints", response_model=list[CheckpointSummary])
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


@api.get("/threads/{thread_id}/checkpoints/{checkpoint_id}")
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


@api.post("/threads/{thread_id}/replay")
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


@api.get("/threads/{thread_id}/trace")
async def thread_trace(thread_id: str, request: Request):
    """Every trace_events row for this thread, grouped into turns (ordered
    oldest-first within each turn) - the raw material for the Observability
    tab's per-turn flow view: planner decision, each specialist's LLM/tool
    calls, judge verdict, cache hit/miss, with token usage on every LLM call."""
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, turn_id, event_type, node, agent_id, model, tool_name,
                       input_summary, output_summary, input_tokens, output_tokens,
                       total_tokens, latency_ms, created_at
                FROM trace_events
                WHERE thread_id = %s
                ORDER BY id
                """,
                (thread_id,),
            )
            rows = await cur.fetchall()

    turns: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        turn_id = r[1]
        if turn_id not in turns:
            turns[turn_id] = {"turn_id": turn_id, "events": []}
            order.append(turn_id)
        turns[turn_id]["events"].append(
            {
                "id": r[0],
                "event_type": r[2],
                "node": r[3],
                "agent_id": r[4],
                "model": r[5],
                "tool_name": r[6],
                "input_summary": r[7],
                "output_summary": r[8],
                "input_tokens": r[9],
                "output_tokens": r[10],
                "total_tokens": r[11],
                "latency_ms": r[12],
                "created_at": r[13].isoformat(),
            }
        )
    return [turns[t] for t in order]


@api.get("/observability/summary")
async def observability_summary(request: Request):
    """Aggregate token/latency/outcome stats across every thread - the cards
    at the top of the Observability tab. Everything here is derived straight
    from `trace_events`, no separate rollup table to keep in sync."""
    async with request.app.state.pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT node, model, COUNT(*) AS calls,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
                FROM trace_events
                WHERE event_type = 'llm_call'
                GROUP BY node, model
                ORDER BY total_tokens DESC
                """
            )
            by_node = await cur.fetchall()

            await cur.execute(
                """
                SELECT thread_id, COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COUNT(DISTINCT turn_id) AS turns
                FROM trace_events
                WHERE event_type = 'llm_call'
                GROUP BY thread_id
                ORDER BY total_tokens DESC
                LIMIT 20
                """
            )
            by_thread = await cur.fetchall()

            await cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE event_type = 'cache_result' AND output_summary = 'hit') AS cache_hits,
                    COUNT(*) FILTER (WHERE event_type = 'cache_result') AS cache_checks,
                    COUNT(*) FILTER (WHERE event_type = 'llm_call' AND node = 'judge'
                                      AND output_summary LIKE 'pass:%') AS judge_passes,
                    COUNT(*) FILTER (WHERE event_type = 'llm_call' AND node = 'judge') AS judge_total,
                    COUNT(DISTINCT thread_id) AS total_threads,
                    COUNT(DISTINCT turn_id) AS total_turns,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM trace_events
                """
            )
            totals = await cur.fetchone()

    return {
        "by_node": [
            {
                "node": r[0],
                "model": r[1],
                "calls": r[2],
                "input_tokens": r[3],
                "output_tokens": r[4],
                "total_tokens": r[5],
                "avg_latency_ms": round(r[6], 1),
            }
            for r in by_node
        ],
        "by_thread": [
            {"thread_id": r[0], "total_tokens": r[1], "turns": r[2]} for r in by_thread
        ],
        "totals": {
            "cache_hits": totals[0],
            "cache_checks": totals[1],
            "judge_passes": totals[2],
            "judge_total": totals[3],
            "total_threads": totals[4],
            "total_turns": totals[5],
            "total_tokens": totals[6],
        },
    }


app.include_router(api)

# Mounted last and at "/" on purpose: Starlette checks explicit routes (all
# of the above) before falling through to a mount, so this only serves
# whatever path none of the API routes already claimed.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
