"""
quickloan/agent.py
------------------
STARTER FILE for Session 12: Multi-Agent Architecture Part 2.

The supervisor graph from Session 10 is already wired below.
Your task (TODO 3) is to route policy/rates responses through the
Compliance Agent before returning to the user.

Current graph (S10):
  classify → route_supervisor →
    call_policy_agent → END
    call_rates_agent  → END
    escalate          → END
    decline           → END

Target graph (S12 -- after TODO 3):
  classify → route_supervisor →
    call_policy_agent → call_compliance_agent → END
    call_rates_agent  → call_compliance_agent → END
    escalate          → END
    decline           → END
"""
import sqlite3

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import CHECKPOINT_DB
from .nodes import (
    classify,
    call_policy_agent,
    call_rates_agent,
    escalate,
    decline,
    route_supervisor,
    # TODO 3: also import call_compliance_agent
)
from .state import QuickLoanState


def build_graph(checkpointer=None):
    builder = StateGraph(QuickLoanState)

    builder.add_node("classify",          classify)
    builder.add_node("call_policy_agent", call_policy_agent)
    builder.add_node("call_rates_agent",  call_rates_agent)
    builder.add_node("escalate",          escalate)
    builder.add_node("decline",           decline)

    # TODO 3: add the compliance agent node
    # builder.add_node("call_compliance_agent", call_compliance_agent)

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify",
        route_supervisor,
        {
            "call_policy_agent": "call_policy_agent",
            "call_rates_agent":  "call_rates_agent",
            "escalate":          "escalate",
            "decline":           "decline",
        },
    )

    # S10 graph: policy/rates go directly to END
    # TODO 3: change these to route through call_compliance_agent
    builder.add_edge("call_policy_agent", END)
    builder.add_edge("call_rates_agent",  END)

    # TODO 3: wire compliance agent to END
    # builder.add_edge("call_compliance_agent", END)

    builder.add_edge("escalate", END)
    builder.add_edge("decline",  END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def run(customer_message: str, thread_id: str = "demo") -> str:
    conn        = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    g           = build_graph(checkpointer=checkpointer)
    config      = {"configurable": {"thread_id": thread_id}}
    result      = g.invoke(
        {
            "customer_message":  customer_message,
            "response":          "",
            "history":           [],
            "query_type":        "",
            "retrieved_docs":    [],
            "specialist":        "",
            "compliance_status": "",
        },
        config=config,
    )
    return result["response"]


if __name__ == "__main__":
    print(run("What is the home loan interest rate?"))
