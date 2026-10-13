### Overview

This repository serves as a backend reference implementation for production-ready multi-agent systems built using **LangGraph**.

### Key Architectural Features

* **State Persistence:** Managed using checkpointers.
* **Communication & Protocols:** Server-Sent Events (SSE) and tools/Model Context Protocol (MCP) support.
* **Governance & Evaluation:** Includes validation checks, governance rules, and judge mechanisms.
* **Retrieval-Augmented Generation:** Integrated RAG support using async processing
* **Gateway Routing:** Features semantic caching, multi-provider failover, and rate limiting.
* **Observability & Debugging:** Trace collection via OpenTelemetry and replay features for agent runs.

### UI & Agent Builder Capabilities

* **Natural Language Agent Creation:** Build and configure agents using natural language.
* **Customization:** Add custom skills, tools, LLM choices, and agent capabilities directly through the UI.

 ### Product Features & Execution Architecture

* **Flexible Agent Creation:** Build and configure agents dynamically either through the natural language UI [Agent Builder] or via manual system prompt and skill assignments.
* **Dual-Mode Execution & Streaming:** Encapsulate and run agents under both **non-streaming** modes and real-time **Server-Sent Events (SSE)** streaming modes for immediate visibility, these agents then can be used as an API. Once creation is confirmed these agents can be tested individually using their own API's which act as a single agent.
 
* **Orchestration, Flows & Agent Building:** Support both **manual definitions** and **Agent Builder** capabilities for workflows and multi-agent systems. Users can define custom chat flows, default behaviors (such as a default greeting like "Hello! How can I help you?"), and individual agent structures as `BotStructure` to be executed at runtime by LangGraph. In multi-agent setups, a central **Planner Agent** interprets incoming user prompts, dynamically selects and deploys the necessary specialized agents, and aggregates their outputs to fulfill complex tasks. Alternatively, these defined workflows and standalone agents can be independently deployed and used elsewhere.

* **Robust Core Infrastructure:** Backed by LangGraph [State Persistence](https://github.com/ranjith-vs/production-agentic-system/edit/main/README.md), Model Context Protocol (MCP) tool support, multi-provider gateway routing, and OpenTelemetry observability.

