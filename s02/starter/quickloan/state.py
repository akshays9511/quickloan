"""
quickloan/state.py
------------------
The shared state that flows through the LangGraph graph.

Session 2 adds conversation history so the agent can remember
previous turns within the same session.
"""
from typing import TypedDict


class QuickLoanState(TypedDict):
    customer_message: str    # the question the customer typed
    response:         str    # the answer QuickLoan will return

    # -----------------------------------------------------------------------
    # TODO 2 of 4 -- Add the history field
    # -----------------------------------------------------------------------
    # Add one more field to track the conversation so far:
    #
    #   history : list[dict]
    #       Each dict has two keys:
    #           {"role": "user",      "content": "..."}
    #           {"role": "assistant", "content": "..."}
    #
    # The respond() node will read this to build a full message list,
    # then append the new turn before returning it.
    #
    # -----------------------------------------------------------------------
    # TODO: add  history: list[dict]
