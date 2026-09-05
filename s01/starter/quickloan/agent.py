"""
quickloan/agent.py
------------------
Graph construction and the terminal loop.

Run the agent from the session folder:
    cd s01/
    python -m quickloan.agent

Session 1 graph:
    START --> respond --> END
"""
from langgraph.graph import END, StateGraph

from .nodes import respond
from .state import QuickLoanState


# ---------------------------------------------------------------------------
# TODO 5 of 5 -- build_graph
# ---------------------------------------------------------------------------
# Implement build_graph() so it:
#
#   1. Creates a StateGraph:
#        builder = StateGraph(QuickLoanState)
#
#   2. Registers the respond node:
#        builder.add_node("respond", respond)
#
#   3. Sets the entry point (first node to run):
#        builder.set_entry_point("respond")
#
#   4. Connects respond → END (the graph exits after one response):
#        builder.add_edge("respond", END)
#
#   5. Compiles and returns the graph:
#        return builder.compile()
#
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the QuickLoan LangGraph graph."""
    raise NotImplementedError("TODO 5: implement build_graph() in quickloan/agent.py")


# Module-level graph instance required by langgraph.json for LangGraph Studio.
# run() uses this directly rather than building a second copy.
graph = build_graph()


# ---------------------------------------------------------------------------
# Terminal loop (provided -- no changes needed)
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 55)
    print("  QuickLoan | FastFinance India")
    print("  Type 'quit' to exit")
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

        result = graph.invoke({"customer_message": user_input, "response": ""})
        print(f"\nQuickLoan: {result['response']}")


if __name__ == "__main__":
    run()
