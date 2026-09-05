"""
quickloan/agent.py
------------------
Builds and runs the QuickLoan LangGraph agent.

Session 4: the graph adds a retrieve_docs node between classify and respond
so relevant FastFinance India policy chunks are fetched before generating the answer.

Run with:
    python -m quickloan.agent
"""
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB
from .nodes import classify, decline, escalate, respond, retrieve_docs, route_query
from .state import QuickLoanState


def build_graph(checkpointer=None):
    builder = StateGraph(QuickLoanState)

    builder.add_node("classify",      classify)
    # ---------------------------------------------------------------------------
    # TODO 4 of 4 -- Add the retrieve_docs node and wire it to respond
    # ---------------------------------------------------------------------------
    # 1. Add the node:
    #      builder.add_node("retrieve_docs", retrieve_docs)
    #
    # 2. After adding conditional edges from "classify", also add:
    #      builder.add_edge("retrieve_docs", "respond")
    #
    # (Also update route_query in nodes.py to return "retrieve_docs" for SIMPLE.)
    # ---------------------------------------------------------------------------
    builder.add_node("respond",       respond)
    builder.add_node("escalate",      escalate)
    builder.add_node("decline",       decline)

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_query, {
        "retrieve_docs": "retrieve_docs",
        "escalate":      "escalate",
        "decline":       "decline",
    })

    # TODO: add builder.add_edge("retrieve_docs", "respond") here
    builder.add_edge("respond",       END)
    builder.add_edge("escalate",      END)
    builder.add_edge("decline",       END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def run() -> None:
    import os
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    print("=" * 55)
    print("  QuickLoan | FastFinance India")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print(f"  Session : {thread_id[:8]}...")
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        project = os.getenv("LANGSMITH_PROJECT", "batch1-quickloan")
        print(f"  Tracing : LangSmith ({project})")
    print("=" * 55)

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
            {"customer_message": user_input, "response": ""},
            config=config,
        )
        route = result.get("query_type", "?")
        docs  = result.get("retrieved_docs", [])
        print(f"\n[Routed: {route}]", end="")
        if docs:
            sources = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
            print(f"  [Retrieved {len(docs)} chunk(s) from: {', '.join(sorted(sources))}]")
        else:
            print()
        print(f"\nQuickLoan: {result['response']}")


if __name__ == "__main__":
    run()
