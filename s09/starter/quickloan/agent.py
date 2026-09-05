"""
quickloan/agent.py
------------------
STARTER FILE -- your task is to wire check_compliance into the graph (TODO 3).

Goal
  Route the graph through the new check_compliance node so every LLM response
  is filtered for RBI compliance before being returned to the customer.

What is already done for you
  - All node imports (including check_compliance) are present
  - The graph structure from Session 8 is intact
  - The run() function prints compliance_status alongside the route info

Your task
  TODO 3: Add the check_compliance node and re-wire respond → check_compliance → END

Run when done
  python -m quickloan.agent   (from inside s09/starter/)
"""
import os
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB, MCP_SERVER_PATH
from .nodes import (
    check_compliance,
    classify,
    decline,
    escalate,
    respond,
    retrieve_docs,
    route_query,
)
from .state import QuickLoanState


def build_graph(checkpointer=None):
    builder = StateGraph(QuickLoanState)

    builder.add_node("classify",      classify)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("respond",       respond)
    builder.add_node("escalate",      escalate)
    builder.add_node("decline",       decline)

    # ---------------------------------------------------------------------------
    # TODO 3 of 3 -- Wire check_compliance into the graph
    # ---------------------------------------------------------------------------
    # 1. Add node:  builder.add_node("check_compliance", check_compliance)
    # 2. Change the edge from respond to check_compliance (not directly to END):
    #      builder.add_edge("respond", "check_compliance")
    #      builder.add_edge("check_compliance", END)
    # ---------------------------------------------------------------------------

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_query, {
        "retrieve_docs": "retrieve_docs",
        "escalate":      "escalate",
        "decline":       "decline",
    })

    builder.add_edge("retrieve_docs", "respond")
    builder.add_edge("respond",       END)  # TODO 3: route through check_compliance first
    builder.add_edge("escalate",      END)
    builder.add_edge("decline",       END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def run() -> None:
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    if not MCP_SERVER_PATH.exists():
        print(f"[QuickLoan] WARNING: MCP server not found at {MCP_SERVER_PATH}")
        print("  Complete Session 7 first.")

    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project    = os.environ.get("LANGCHAIN_PROJECT", "batch1-quickloan")

    print("=" * 60)
    print("  QuickLoan | FastFinance India")
    print("  Compliance: RBI phrase filter + rate verification")
    print(f"  Tracing   : {'LangSmith (' + project + ')' if tracing_on else 'off (set LANGSMITH_API_KEY to enable)'}")
    print("  Tools: MCP server (query_rates, query_eligibility)")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print(f"  Session: {thread_id[:8]}...")
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

        result = _graph.invoke(
            {"customer_message": user_input, "response": "", "compliance_status": ""},
            config=config,
        )
        route      = result.get("query_type", "?")
        compliance = result.get("compliance_status", "")
        docs       = result.get("retrieved_docs", [])

        print(f"\n[Routed: {route}]", end="")
        if docs:
            sources = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
            print(f"  [RAG: {len(docs)} chunk(s) from {', '.join(sorted(sources))}]", end="")
        if compliance:
            print(f"  [Compliance: {compliance}]", end="")
        print()
        print(f"\nQuickLoan: {result['response']}")


if __name__ == "__main__":
    run()
