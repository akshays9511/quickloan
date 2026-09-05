"""
quickloan/agent.py
------------------
Builds and runs the QuickLoan LangGraph agent.

Session 3: the graph routes each query through classify() first,
then branches to respond(), escalate(), or decline().

Run with:
    cd s03/starter/
    python -m quickloan.agent
"""
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB
from .nodes import classify, decline, escalate, respond, route_query
from .state import QuickLoanState


# ---------------------------------------------------------------------------
# TODO 4 of 4 -- Build the graph with query routing
# ---------------------------------------------------------------------------
# Session 3 graph:
#   START --> classify --> route_query --> {respond | escalate | decline} --> END
#
# Steps:
# 1. Create builder:
#      builder = StateGraph(QuickLoanState)
#
# 2. Add all four nodes:
#      builder.add_node("classify", classify)
#      builder.add_node("respond",  respond)
#      builder.add_node("escalate", escalate)
#      builder.add_node("decline",  decline)
#
# 3. Set entry point to "classify":
#      builder.set_entry_point("classify")
#
# 4. Add conditional edges from "classify" using route_query:
#      builder.add_conditional_edges("classify", route_query, {
#          "respond":  "respond",
#          "escalate": "escalate",
#          "decline":  "decline",
#      })
#    The dict is required so LangGraph Studio can draw the routing arrows.
#
# 5. Add edges from each terminal node to END:
#      builder.add_edge("respond",  END)
#      builder.add_edge("escalate", END)
#      builder.add_edge("decline",  END)
#
# 6. Compile and return (pass checkpointer through — do not default to MemorySaver):
#      return builder.compile(checkpointer=checkpointer)
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None):
    raise NotImplementedError("TODO 4: implement build_graph()")


graph = build_graph()


def run() -> None:
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    print("=" * 55)
    print("  QuickLoan | FastFinance India")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print(f"  Session: {thread_id[:8]}...")
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
        print(f"\n[Routed: {route}]")
        print(f"\nQuickLoan: {result['response']}")


if __name__ == "__main__":
    run()
