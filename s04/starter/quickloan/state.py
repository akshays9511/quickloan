"""
quickloan/state.py
------------------
The shared state that flows through the LangGraph graph.

Session 4 adds retrieved_docs so RAG chunks can flow from
retrieve_docs() into respond().
"""
from typing import TypedDict


class QuickLoanState(TypedDict):
    customer_message: str
    response:         str
    history:          list[dict]
    query_type:       str

    # -----------------------------------------------------------------------
    # TODO 1 of 4 -- Add the retrieved_docs field
    # -----------------------------------------------------------------------
    # retrieved_docs holds the text chunks returned by retrieve_docs().
    # respond() reads it to add policy context to the system message.
    # Type hint: list[str]
    #
    # TODO: add  retrieved_docs: list[str]
