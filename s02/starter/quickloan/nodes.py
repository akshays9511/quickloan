"""
quickloan/nodes.py
------------------
Graph nodes for QuickLoan.

Session 2: the respond() node now carries conversation history across
turns so the LLM can refer back to earlier messages.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import SYSTEM_PROMPT
from .state import QuickLoanState
from .tools import llm


def respond(state: QuickLoanState) -> dict:
    history = state.get("history", [])

    # -----------------------------------------------------------------------
    # TODO 3 of 4 -- Build the message list and update history
    # -----------------------------------------------------------------------
    # Step A: Build the message list for the LLM call.
    #
    #   messages = [SystemMessage(content=SYSTEM_PROMPT)]
    #
    #   Then loop over `history` and append each turn:
    #     - {"role": "user", ...}      → HumanMessage(content=turn["content"])
    #     - {"role": "assistant", ...} → AIMessage(content=turn["content"])
    #
    #   Finally append the new customer turn:
    #     messages.append(HumanMessage(content=state["customer_message"]))
    #
    # Step B: Call the LLM.
    #
    #   try:
    #       result = llm.invoke(messages)
    #       response_text = result.content
    #   except Exception as e:
    #       print(f"[QuickLoan] LLM error: {e}")
    #       response_text = "I am temporarily unavailable. Please try again in a moment."
    #
    # Step C: Append this turn to history and return both fields.
    #
    #   new_history = history + [
    #       {"role": "user",      "content": state["customer_message"]},
    #       {"role": "assistant", "content": response_text},
    #   ]
    #   return {"response": response_text, "history": new_history}
    #
    # -----------------------------------------------------------------------
    # TODO: replace this placeholder with the implementation above
    response_text = "TODO: implement respond()"
    return {"response": response_text, "history": history}
