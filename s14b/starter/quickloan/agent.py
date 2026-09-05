"""
quickloan/agent.py
------------------
Builds and runs the QuickLoan multi-agent LangGraph supervisor.

Session 14b: same graph structure as S14, upgraded guard (LlamaGuard 3 8B).
"""
import os
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB, MCP_SERVER_PATH
from .nodes import (
    blocked,
    call_compliance_agent,
    call_policy_agent,
    call_rates_agent,
    classify,
    decline,
    escalate,
    guard,
    route_guard,
    route_supervisor,
)
from .state import QuickLoanState


def build_graph(checkpointer=None):
    builder = StateGraph(QuickLoanState)

    builder.add_node("guard",                guard)
    builder.add_node("blocked",              blocked)
    builder.add_node("classify",             classify)
    builder.add_node("call_policy_agent",    call_policy_agent)
    builder.add_node("call_rates_agent",     call_rates_agent)
    builder.add_node("call_compliance_agent", call_compliance_agent)
    builder.add_node("escalate",             escalate)
    builder.add_node("decline",              decline)

    builder.set_entry_point("guard")
    builder.add_conditional_edges("guard", route_guard, {
        "classify": "classify",
        "blocked":  "blocked",
    })
    builder.add_edge("blocked", END)

    builder.add_conditional_edges("classify", route_supervisor, {
        "call_policy_agent": "call_policy_agent",
        "call_rates_agent":  "call_rates_agent",
        "escalate":          "escalate",
        "decline":           "decline",
    })

    builder.add_edge("call_policy_agent",     "call_compliance_agent")
    builder.add_edge("call_rates_agent",      "call_compliance_agent")
    builder.add_edge("call_compliance_agent", END)
    builder.add_edge("escalate",              END)
    builder.add_edge("decline",               END)

    return builder.compile(checkpointer=checkpointer)  # None = no checkpointer, Studio-safe


graph = build_graph()


def run() -> None:
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    g         = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    if not MCP_SERVER_PATH.exists():
        print(f"[QuickLoan] WARNING: MCP server not found at {MCP_SERVER_PATH}")

    print("=" * 60)
    print("  QuickLoan | FastFinance India")
    print("  Architecture: Guard (LlamaGuard 3) -> Supervisor -> [Policy|Rates] -> Compliance")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nQuickLoan: Session ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nQuickLoan: Thank you for choosing FastFinance India. Goodbye!")
            break

        result = g.invoke(
            {
                "customer_message":  user_input,
                "response":          "",
                "specialist":        "",
                "retrieved_docs":    [],
                "compliance_status": "",
                "blocked_reason":    "",
            },
            config=config,
        )

        blocked_r = result.get("blocked_reason", "")
        if blocked_r:
            print(f"\n[Guard: BLOCKED ({blocked_r})]")
        else:
            sp = result.get("specialist", "?")
            cs = result.get("compliance_status", "")
            print(f"\n[Route: {result.get('query_type','?')} -> {sp}]", end="")
            if cs:
                print(f"  [Compliance: {cs}]", end="")
            print()
        print(f"\nQuickLoan: {result['response']}")


if __name__ == "__main__":
    run()
