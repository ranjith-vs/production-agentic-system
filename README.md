### Overview

This repo can be used for a learning and debugging experience on production AI systems.

A backend reference implementation for production-ready multi-agent systems built on **LangGraph**, fronted by an **Apache APISIX** AI gateway.

Docker containers signify the distributed nature of the system.

### Key Features

* **Gateway-routed LLM calls** — all OpenAI traffic goes through APISIX (`ai-proxy` for plain chat, a raw passthrough route for tool-calling/embeddings); the real API key lives only in the gateway config, never in app code.
* **Streaming** — `/chat/stream` streams token-by-token via Server-Sent Events.
* **State persistence** — conversation state checkpointed to Postgres (`AsyncPostgresSaver`); pass a `thread_id` to resume a conversation.
* **MCP tool support** — tools served by a standalone MCP server and loaded via `langchain-mcp-adapters`.
* **RAG** — document upload → async chunk/embed (Celery + Redis) → pgvector similarity search with HNSW GIN indexes, reranking top k retrievals, exposed as a tool to agents
* **Multi-agent orchestration** — a planner picks one or more specialist subgraphs (`general`, `research`) per turn and synthesizes their replies when more than one runs. Agents with skills, tool access and confined boundaries for efficient monitoring
* **Semantic caching** — prompt/response pairs cached by embedding similarity in Postgres, scoped per agent.
* **Governance** — a zero-temperature judge model reviews each draft reply for relevance/safety before it's returned.
* **Prompts as data** — planner and judge prompts live in a Postgres `prompts` table (seeded with defaults on startup) and are read fresh on every turn, so they can be edited via the `/prompts` API with no redeploy.
* **Agent Builder UI** — create agents (name, system prompt, allowed tool-sets) through a UI; chat against a specific agent by id.
* **Auth** — username/password login (bcrypt + JWT); every API route except `/health` and `/auth/*` requires a valid token.
* **Observability** — full OpenTelemetry trace tree per request via `arize-phoenix-otel` (`http://localhost:6006`), plus an in-app Observability tab showing every turn's planner decision, specialist LLM/tool calls, and judge verdict with per-call token usage, backed by `GET /threads/{id}/trace` and `GET /observability/summary`.
* **Replay/debugging** — inspect and replay any thread's checkpoint history (`/threads/{id}/checkpoints`, `/threads/{id}/replay`).

### Running it

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
docker compose up --build
```

Then open **http://localhost:8000** for the UI (Chat / Documents / Agents / Threads), or **http://localhost:6006** for Phoenix traces.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'

# streaming (SSE)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'
```

Request path: `app` (FastAPI + LangGraph) → `apisix:9080` → `api.openai.com`. Routes are provisioned at startup by the `apisix-init` container (`apisix/init-routes.sh`).

First call needs a token - register once, then pass it on every request:

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "you", "password": "a-real-password"}'
# -> {"access_token": "...", ...}

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{"message": "hello"}'
```

### Database schema

All table definitions live under [`schema/`](schema/) as plain, numbered `.sql` files - nothing about table structure is embedded in the application code. On startup the app applies every file in that folder, in order, against whatever Postgres `POSTGRES_URL` points to (every statement is `IF NOT EXISTS`, so re-running them on each boot is a no-op). This project targets **local Postgres only** - the `postgres` service in `docker-compose.yml` (image `pgvector/pgvector:pg16`) - by design, so the whole stack runs with one `docker compose up` and nothing depends on an external hosted database.

### Editing prompts at runtime

```bash
curl http://localhost:8000/prompts
curl -X PUT http://localhost:8000/prompts/planner \
  -H "Content-Type: application/json" \
  -d '{"content": "your new planner prompt"}'
```

### Tech Stack

| Component | Used for | Alternatives |
|---|---|---|
| **LangGraph** | Agent/graph orchestration (planner, specialist subgraphs, checkpointing) | LlamaIndex Workflows, CrewAI, AutoGen/Semantic Kernel, a hand-rolled state machine |
| **FastAPI** | HTTP API, SSE streaming, dependency-injected auth | Flask, Django REST Framework, Litestar, raw Starlette, aiohttp |
| **Apache APISIX** | AI gateway — centralizes the OpenAI key, rate limiting, routing | Kong AI Gateway, Envoy + ext_proc, LiteLLM Proxy, Portkey, Cloudflare AI Gateway |
| **PostgreSQL + pgvector** (local, via Docker) | Conversation state, semantic cache, RAG chunks, agents/prompts/users/trace_events | Pinecone/Weaviate/Qdrant/Milvus (dedicated vector DBs), a managed Postgres (RDS, Cloud SQL, Neon), SQLite (dev-only, no vector support) |
| **Celery + Redis** | Async document ingestion (chunk/embed) off the request path | RQ, Dramatiq, arq (asyncio-native), Temporal, SQS + Lambda |
| **OpenAI (gpt-4o-mini / text-embedding-3-small)** | Chat, planning, judging, embeddings | Anthropic Claude, Google Gemini, Azure OpenAI, local models via Ollama/vLLM |
| **MCP (FastMCP + langchain-mcp-adapters)** | Serving tools (`get_current_time`, `add_numbers`) from a separate process | Plain in-process LangChain `@tool` functions, OpenAI function calling directly, LangServe |
| **Arize Phoenix + OpenTelemetry** | Full trace tree per request (auto-instrumented via `openinference`) | LangSmith, Langfuse, Helicone, Datadog LLM Observability |
| **bcrypt + PyJWT** | Password hashing + stateless login tokens | Argon2 (hashing), session cookies instead of JWT, managed auth (Auth0, Clerk) |
| **etcd** | APISIX's config store | APISIX standalone mode (file-based config, no etcd needed), Consul |
| **Docker Compose** | Local multi-service orchestration | Kubernetes (prod), Nomad, Tilt/Skaffold |
| **Vanilla HTML/CSS/JS UI** | Single static page, no build step | React/Next.js, Vue, Svelte, htmx |
