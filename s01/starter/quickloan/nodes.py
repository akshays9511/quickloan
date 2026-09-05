"""
quickloan/nodes.py
------------------
Node functions for the QuickLoan graph.

Each node is a plain Python function:
  - Input : the full QuickLoanState (read-only)
  - Output: a dict containing ONLY the keys this node changed
             (LangGraph merges it into the state automatically)
"""
from langchain_core.messages import HumanMessage, SystemMessage

from .config import SYSTEM_PROMPT
from .state import QuickLoanState
from .tools import llm


# ---------------------------------------------------------------------------
# TODO 4 of 5 -- respond node
# ---------------------------------------------------------------------------
# Implement the respond() function so it:
#
#   1. Builds a messages list:
#        messages = [
#            SystemMessage(content=SYSTEM_PROMPT),
#            HumanMessage(content=state["customer_message"]),
#        ]
#
#   2. Calls the LLM inside a try / except block:
#        result = llm.invoke(messages)
#
#   3. On success  → return {"response": result.content}
#      On exception → print the error with a [QuickLoan] prefix
#                      and return a safe fallback string so the
#                      agent never crashes mid-conversation.
#
# ---------------------------------------------------------------------------

def respond(state: QuickLoanState) -> dict:
    """Call the LLM and return the agent's reply."""
    raise NotImplementedError("TODO 4: implement respond() in quickloan/nodes.py")
