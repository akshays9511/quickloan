"""
quickloan/nodes.py
------------------
Graph nodes and routing function for QuickLoan.

Session 3 adds classify() and route_query() so queries are directed
to respond(), escalate(), or decline() based on their type.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE, SYSTEM_PROMPT
from .state import QuickLoanState
from .tools import classifier_llm, llm


def classify(state: QuickLoanState) -> dict:
    """Classify the customer question into SIMPLE, COMPLEX, or OUT_OF_SCOPE."""
    # -----------------------------------------------------------------------
    # TODO 2 of 4 -- Implement classify()
    # -----------------------------------------------------------------------
    # 1. Build the message list for the classifier LLM:
    #      messages = [
    #          SystemMessage(content=CLASSIFY_SYSTEM),
    #          HumanMessage(content=state["customer_message"]),
    #      ]
    #
    # 2. Call classifier_llm.invoke(messages) inside a try/except:
    #      On success:
    #        query_type = result.content.strip().upper()
    #        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
    #            query_type = "SIMPLE"
    #      On exception:
    #        print(f"[QuickLoan] Classification error: {e}")
    #        query_type = "SIMPLE"
    #
    # 3. Return {"query_type": query_type}
    # -----------------------------------------------------------------------
    # TODO: implement this node
    pass


def respond(state: QuickLoanState) -> dict:
    """Handle SIMPLE queries. Provided -- no changes needed."""
    history  = state.get("history", [])
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result        = llm.invoke(messages)
        response_text = result.content
    except Exception as e:
        print(f"[QuickLoan] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


def escalate(state: QuickLoanState) -> dict:
    """Handle COMPLEX queries with a loan officer referral. Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: QuickLoanState) -> dict:
    """Handle OUT_OF_SCOPE queries with a canned decline. Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: QuickLoanState) -> str:
    """Read query_type and return the name of the next node."""
    # -----------------------------------------------------------------------
    # TODO 3 of 4 -- Implement route_query()
    # -----------------------------------------------------------------------
    # Read the query type and route accordingly:
    #   if query_type == "COMPLEX":      return "escalate"
    #   if query_type == "OUT_OF_SCOPE": return "decline"
    #   otherwise:                        return "respond"
    #
    # Use state.get("query_type", "SIMPLE") to read safely.
    # -----------------------------------------------------------------------
    # TODO: implement this function
    pass
