import asyncio
import os
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from .cache import lookup_cached_response, store_cache_entry
from .judge import JudgeVerdict
from .prompts import get_prompt
from .rag import search_documents
from .tools import load_mcp_tools
from .tracing import log_event

GATEWAY_URL = os.environ.get("APISIX_GATEWAY_URL", "http://apisix:9080")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Goes through the "openai-raw" passthrough route (apisix/init-routes.sh), not
# the "openai-chat" ai-proxy route from Step 1/2 - ai-proxy's schema rejects
# tool_calls/tool-role messages, so it can't carry a tool-calling agent loop.
# The real OpenAI key lives in the apisix route config either way, never here.
llm = ChatOpenAI(
    base_url=f"{GATEWAY_URL}/openai/v1",
    api_key="unused-key-apisix-holds-the-real-one",
    model=MODEL,
)

# A judge should be as repeatable as a non-zero-temperature model can be: the
# same draft reply judged twice gave different pass/fail verdicts at the
# default temperature (see README). temperature=0 doesn't make it fully
# deterministic, but cuts the variance a lot - a real judge should also vote
# across multiple samples rather than trust a single call, which this doesn't.
judge_llm = ChatOpenAI(
    base_url=f"{GATEWAY_URL}/openai/v1",
    api_key="unused-key-apisix-holds-the-real-one",
    model=MODEL,
    temperature=0,
)


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    system_prompt: str
    allowed_specialists: list[str]
    agent_id: int | None
    thread_id: str
    turn_id: str


class Plan(BaseModel):
    agents: list[Literal["general", "research"]]


def _usage(response) -> dict:
    u = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "total_tokens": u.get("total_tokens"),
    }


def _build_react_subgraph(tools: list, pool, specialist_name: str) -> StateGraph:
    """A plain ReAct tool-calling loop, parameterized by its tool set - one
    per specialist, built as a standalone compiled graph the planner calls
    directly (not wired into the parent graph's own edges). The model and
    tool nodes are hand-rolled rather than langgraph.prebuilt's ToolNode /
    tools_condition so each call can be timed, token-counted, and written to
    `trace_events` for the Observability UI's per-turn flow view."""
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def call_model(state: State) -> State:
        t0 = time.monotonic()
        response = await llm_with_tools.ainvoke(state["messages"])
        await log_event(
            pool,
            thread_id=state["thread_id"],
            turn_id=state["turn_id"],
            event_type="llm_call",
            node=specialist_name,
            agent_id=state.get("agent_id"),
            model=MODEL,
            output_summary=str(response.content) or "(tool call)",
            latency_ms=int((time.monotonic() - t0) * 1000),
            **_usage(response),
        )
        return {"messages": [response]}

    async def call_tools(state: State) -> State:
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            tool_obj = tools_by_name.get(call["name"])
            t0 = time.monotonic()
            try:
                output = await tool_obj.ainvoke(call["args"]) if tool_obj else f"Unknown tool: {call['name']}"
            except Exception as exc:  # noqa: BLE001 - surfaced to the model as the tool result
                output = f"Error: {exc}"
            latency_ms = int((time.monotonic() - t0) * 1000)
            await log_event(
                pool,
                thread_id=state["thread_id"],
                turn_id=state["turn_id"],
                event_type="tool_call",
                node=specialist_name,
                agent_id=state.get("agent_id"),
                tool_name=call["name"],
                input_summary=str(call["args"]),
                output_summary=str(output),
                latency_ms=latency_ms,
            )
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    def route_tools(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    sub = StateGraph(State)
    sub.add_node("call_model", call_model)
    sub.add_node("tools", call_tools)
    sub.set_entry_point("call_model")
    sub.add_conditional_edges("call_model", route_tools, {"tools": "tools", END: END})
    sub.add_edge("tools", "call_model")
    # Explicitly opt out of inheriting the parent graph's checkpointer: these
    # specialists are stateless one-shot tool loops within a single planner
    # turn, and >1 of them sharing the parent's checkpoint namespace in the
    # same turn trips LangGraph's MultipleSubgraphsError guard.
    return sub.compile(checkpointer=False)


async def build_graph_builder(pool) -> StateGraph:
    """`pool` is the app's shared Postgres connection pool - passed through so
    planner/judge prompts can be read fresh from the `prompts` table on every
    turn (see app/agent/prompts.py), and so every LLM/tool call in this graph
    can log a row to `trace_events` (see app/agent/tracing.py) for the
    Observability UI."""
    mcp_tools = await load_mcp_tools()
    specialists = {
        "general": _build_react_subgraph(mcp_tools, pool, "general"),
        "research": _build_react_subgraph([search_documents], pool, "research"),
    }
    planner = llm.with_structured_output(Plan, include_raw=True)
    judge = judge_llm.with_structured_output(JudgeVerdict, include_raw=True)

    async def plan_and_run(state: State) -> State:
        allowed = state.get("allowed_specialists") or ["general", "research"]
        system_prompt = state.get("system_prompt") or ""

        planner_prompt = await get_prompt(pool, "planner")
        planner_input = [("system", planner_prompt), *state["messages"]]
        t0 = time.monotonic()
        planner_out = await planner.ainvoke(planner_input)
        plan: Plan = planner_out["parsed"]
        await log_event(
            pool,
            thread_id=state["thread_id"],
            turn_id=state["turn_id"],
            event_type="llm_call",
            node="planner",
            agent_id=state.get("agent_id"),
            model=MODEL,
            output_summary=f"selected: {', '.join(plan.agents) or '(none)'}",
            latency_ms=int((time.monotonic() - t0) * 1000),
            **_usage(planner_out["raw"]),
        )
        agent_names = [n for n in dict.fromkeys(plan.agents) if n in allowed] or [allowed[0]]

        specialist_messages = state["messages"]
        if system_prompt:
            # Ephemeral: prepended for this turn's specialist calls only, not
            # persisted into the checkpointed thread history, so switching
            # agents mid-thread doesn't leave every prior persona's prompt
            # stacked into the context.
            specialist_messages = [("system", system_prompt), *specialist_messages]

        replies = []
        for name in agent_names:
            result = await specialists[name].ainvoke(
                {
                    "messages": specialist_messages,
                    "thread_id": state["thread_id"],
                    "turn_id": state["turn_id"],
                    "agent_id": state.get("agent_id"),
                }
            )
            replies.append((name, result["messages"][-1].content))

        if len(replies) == 1:
            final_content = replies[0][1]
        else:
            findings = "\n\n".join(f"[{name}]: {content}" for name, content in replies)
            synthesis_prompt = (
                "Combine these specialist findings into one clear answer for the user. "
                f"Findings:\n\n{findings}"
            )
            t0 = time.monotonic()
            synthesis = await llm.ainvoke([("system", synthesis_prompt), state["messages"][-1]])
            await log_event(
                pool,
                thread_id=state["thread_id"],
                turn_id=state["turn_id"],
                event_type="llm_call",
                node="synthesis",
                agent_id=state.get("agent_id"),
                model=MODEL,
                output_summary=str(synthesis.content),
                latency_ms=int((time.monotonic() - t0) * 1000),
                **_usage(synthesis),
            )
            final_content = synthesis.content

        return {"messages": [AIMessage(content=final_content)]}

    async def judge_reply(state: State) -> State:
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_message = human_messages[-1].content if human_messages else ""
        draft = state["messages"][-1].content
        judge_prompt = await get_prompt(pool, "judge")
        t0 = time.monotonic()
        judge_out = await judge.ainvoke(
            [
                ("system", judge_prompt),
                ("user", f"User asked: {user_message}\n\nDraft reply: {draft}"),
            ]
        )
        verdict: JudgeVerdict = judge_out["parsed"]
        await log_event(
            pool,
            thread_id=state["thread_id"],
            turn_id=state["turn_id"],
            event_type="llm_call",
            node="judge",
            agent_id=state.get("agent_id"),
            model=MODEL,
            output_summary=f"{verdict.verdict}: {verdict.reason}",
            latency_ms=int((time.monotonic() - t0) * 1000),
            **_usage(judge_out["raw"]),
        )
        if verdict.verdict == "fail":
            return {
                "messages": [
                    AIMessage(
                        content="I can't help with that request.",
                        name="governance_blocked",
                    )
                ]
            }
        return {}

    def route_after_judge(state: State) -> str:
        last_message = state["messages"][-1]
        was_blocked = (
            isinstance(last_message, AIMessage)
            and getattr(last_message, "name", None) == "governance_blocked"
        )
        # A blocked reply isn't cached - caching it would mean a safe rephrase
        # of the same question later gets the refusal served straight out of
        # the semantic cache, bypassing the judge entirely.
        return END if was_blocked else "store_cache"

    async def check_cache(state: State) -> State:
        last_user_message = state["messages"][-1].content
        agent_id = state.get("agent_id")
        cached = await asyncio.to_thread(lookup_cached_response, last_user_message, agent_id)
        await log_event(
            pool,
            thread_id=state["thread_id"],
            turn_id=state["turn_id"],
            event_type="cache_result",
            node="check_cache",
            agent_id=agent_id,
            output_summary="hit" if cached else "miss",
        )
        if cached is None:
            return {}
        return {"messages": [AIMessage(content=cached, name="semantic_cache_hit")]}

    def route_after_cache(state: State) -> str:
        last_message = state["messages"][-1]
        was_cache_hit = (
            isinstance(last_message, AIMessage)
            and getattr(last_message, "name", None) == "semantic_cache_hit"
        )
        return END if was_cache_hit else "plan_and_run"

    async def store_cache(state: State) -> State:
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if human_messages:
            await asyncio.to_thread(
                store_cache_entry,
                human_messages[-1].content,
                state["messages"][-1].content,
                state.get("agent_id"),
            )
        return {}

    builder = StateGraph(State)
    builder.add_node("check_cache", check_cache)
    builder.add_node("plan_and_run", plan_and_run)
    builder.add_node("judge_reply", judge_reply)
    builder.add_node("store_cache", store_cache)
    builder.set_entry_point("check_cache")
    builder.add_conditional_edges(
        "check_cache", route_after_cache, {"plan_and_run": "plan_and_run", END: END}
    )
    builder.add_edge("plan_and_run", "judge_reply")
    builder.add_conditional_edges(
        "judge_reply", route_after_judge, {"store_cache": "store_cache", END: END}
    )
    builder.add_edge("store_cache", END)
    return builder
