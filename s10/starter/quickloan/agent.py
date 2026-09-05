"""
quickloan/agent.py
------------------
STARTER FILE -- complete the TODOs in nodes.py first, then wire the graph here.

Session 10: Supervisor + Specialist Agent architecture.
  - Supervisor classifies into RATES / POLICY / COMPLEX / OUT_OF_SCOPE
  - Rates Agent   uses MCP tools (query_rates, query_eligibility)
  - Policy Agent  uses RAG (vectorstore) -- no live DB access

What to implement (after completing nodes.py TODOs 1-3):
  The build_graph() function below needs you to:
  1. Add all five nodes to builder (classify, call_policy_agent, call_rates_agent, escalate, decline)
  2. Set the entry point to "classify"
  3. Add conditional edges from "classify" using route_supervisor with the explicit path map
  4. Add terminal edges from each specialist node to END

Run when done
  python -m quickloan.agent   (from inside s10/starter/)
"""
import os
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB, MCP_SERVER_PATH
from .nodes import (
    call_policy_agent,
    call_rates_agent,
    classify,
    decline,
    escalate,
    route_supervisor,
)
from .state import QuickLoanState


def build_graph(checkpointer=None):
    builder = StateGraph(QuickLoanState)

    # ---------------------------------------------------------------------------
    # TODO: Wire the supervisor graph
    # ---------------------------------------------------------------------------
    # builder.add_node("classify",          classify)
    # builder.add_node("call_policy_agent", call_policy_agent)
    # builder.add_node("call_rates_agent",  call_rates_agent)
    # builder.add_node("escalate",          escalate)
    # builder.add_node("decline",           decline)
    #
    # builder.set_entry_point("classify")
    # builder.add_conditional_edges("classify", route_supervisor, {
    #     "call_policy_agent": "call_policy_agent",
    #     "call_rates_agent":  "call_rates_agent",
    #     "escalate":          "escalate",
    #     "decline":           "decline",
    # })
    #
    # builder.add_edge("call_policy_agent", END)
    # builder.add_edge("call_rates_agent",  END)
    # builder.add_edge("escalate",          END)
    # builder.add_edge("decline",           END)
    # ---------------------------------------------------------------------------

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def run() -> None:
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    g         = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    if not MCP_SERVER_PATH.exists():
        print(f"[QuickLoan] WARNING: MCP server not found at {MCP_SERVER_PATH}")
        print("  Complete Session 7 first.")

    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project    = os.environ.get("LANGCHAIN_PROJECT", "batch1-quickloan")

    print("=" * 60)
    print("  QuickLoan | FastFinance India")
    print("  Architecture: Supervisor + Rates Agent + Policy Agent")
    print(f"  Tracing: {'LangSmith (' + project + ')' if tracing_on else 'off'}")
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

        result = g.invoke(
            {"customer_message": user_input, "response": "",
             "specialist": "", "retrieved_docs": []},
            config=config,
        )
        specialist = result.get("specialist", "?")
        docs       = result.get("retrieved_docs", [])

        print(f"\n[Route: {result.get('query_type','?')} → {specialist}]", end="")
        if docs:
            sources = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
            print(f"  [RAG: {len(docs)} chunk(s) from {', '.join(sorted(sources))}]", end="")
        print()
        print(f"\nQuickLoan: {result['response']}")


if __name__ == "__main__":
    run()
