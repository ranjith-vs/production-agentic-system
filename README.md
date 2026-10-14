### Overview

This repo can be used for a learning and debugging experience on production AI systems.

A backend reference implementation for production-ready multi-agent systems built on **LangGraph**, fronted by an **Apache APISIX** AI gateway.

Docker containers signify the distributed nature of the system.

### Key Features

* **Gateway-routed LLM calls** — all OpenAI traffic goes through APISIX (`ai-proxy` for plain chat, a raw passthrough route for tool-calling/embeddings); the real API key lives only in the gateway config, never in app code.
* **Streaming** — `/chat/stream` streams token-by-token via Server-Sent Events.
* **State persistence** — conversation state checkpointed to Postgres (`AsyncPostgresSaver`); pass a `thread_id` to resume a conversation.
* **MCP tool support** — tools served by a standalone MCP server and loaded via `langchain-mcp-adapters`.
* **RAG** — document upload → async chunk/embed (Celery + Redis) → pgvector similarity search, exposed as a tool.
* **Multi-agent orchestration** — a planner picks one or more specialist subgraphs (`general`, `research`) per turn and synthesizes their replies when more than one runs.
* **Semantic caching** — prompt/response pairs cached by embedding similarity in Postgres, scoped per agent.
* **Governance** — a zero-temperature judge model reviews each draft reply for relevance/safety before it's returned.
* **Prompts as data** — planner and judge prompts live in a Postgres `prompts` table (seeded with defaults on startup) and are read fresh on every turn, so they can be edited via the `/prompts` API with no redeploy.
* **Agent Builder UI** — create agents (name, system prompt, allowed tool-sets) through a UI; chat against a specific agent by id.
* **Observability** — full OpenTelemetry trace tree per request via `arize-phoenix-otel`, browsable at `http://localhost:6006`.
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

### Editing prompts at runtime

```bash
curl http://localhost:8000/prompts
curl -X PUT http://localhost:8000/prompts/planner \
  -H "Content-Type: application/json" \
  -d '{"content": "your new planner prompt"}'
```
