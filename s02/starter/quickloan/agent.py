"""
quickloan/agent.py
------------------
Builds and runs the QuickLoan LangGraph agent.

Session 2: the graph is compiled with a checkpointer so LangGraph
persists conversation history across turns automatically.

Run with:
    cd s02/
    python -m quickloan.agent
"""
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB
from .nodes import respond
from .state import QuickLoanState


# ---------------------------------------------------------------------------
# TODO 4 of 4 -- Build the graph with checkpointer support
# ---------------------------------------------------------------------------
# def build_graph(checkpointer=None):
#     builder = StateGraph(QuickLoanState)
#     builder.add_node("respond", respond)
#     builder.set_entry_point("respond")
#     builder.add_edge("respond", END)
#     return builder.compile(checkpointer=checkpointer)
#
# ---------------------------------------------------------------------------
# TODO: uncomment the function above, then delete these two placeholder lines
def build_graph(checkpointer=None):
    raise NotImplementedError("TODO 4: implement build_graph() in quickloan/agent.py")


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
        print(f"\nQuickLoan: {result['response']}")


if __name__ == "__main__":
    run()
