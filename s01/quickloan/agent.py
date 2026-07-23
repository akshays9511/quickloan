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
from uuid import uuid4
import sqlite3 , os
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from .config import CHECKPOINT_DB,ESCALATE_RESPONSE
from .nodes import respond , classify,decline,escalate,route_query, retrieve_docs
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

def build_graph(checkpointer=None):
    builder = StateGraph(QuickLoanState)
    builder.add_node("classify", classify) #Naming of node
    builder.add_node("decline", decline)
#   builder.add_node("escalate", escalate)
    builder.add_node("respond", respond)
    builder.add_node("retrieve_docs", retrieve_docs)
 
    builder.set_entry_point("classify") # START
    builder.add_conditional_edges("classify", route_query, {
        "respond": "respond",
        # "esclate": "escalate",
        "decline": "decline",
        "retrieve_docs": "retrieve_docs"
    })
    builder.add_edge("respond", END)
    # builder.add_edge("escalate", END)
    builder.add_edge("retrieve_docs", "respond")
    builder.add_edge("decline", END)  
    builder.add_edge("respond", END)
    # if checkpointer is None:
        # checkpointer=MemorySaver()
    return builder.compile(checkpointer=checkpointer)

# Module-level graph instance required by langgraph.json for LangGraph Studio.
# run() uses this directly rather than building a second copy.
graph = build_graph()


# ---------------------------------------------------------------------------
# Terminal loop (provided -- no changes needed)
# ---------------------------------------------------------------------------

def run() -> None:
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn) )
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    print("=" * 55)
    print("  QuickLoan | FastFinance India")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print(f"  Session: {thread_id[:8]}...")  # sanity check -- confirms config actually reached graph.invoke()
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        project = os.getenv("LANGSMITH_PROJECT", "batch1-wealthdesk")
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
        result = _graph.invoke({"customer_message": user_input, "response": ""}, config=config,)
        route = result.get("query_type", "?")
        docs = result.get("retrieved_docs", [])
        response = result["response"]
        print(f"\n[Routed: {route}]")
        if docs and response != ESCALATE_RESPONSE:
            sources = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
            print(f"  [Retrieved {len(docs)} chunk(s) from: {', '.join(sorted(sources))}]")
        else:
            print()
        print(f"\nQuickLoan: {result['response']}")


if __name__ == "__main__":
    run()
