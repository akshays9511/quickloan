"""
quickloan/nodes.py
------------------
Graph nodes and routing for QuickLoan.

Session 4 adds ChromaDB retrieval so SIMPLE queries get relevant
policy context before the LLM generates a response.
"""
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, SYSTEM_PROMPT, VECTORSTORE_DIR,
)
from .state import QuickLoanState
from .tools import classifier_llm, llm

vectorstore = None


def _init_vectorstore() -> None:
    """Load ChromaDB + embeddings. No-op if already initialised or mocked."""
    global vectorstore
    if vectorstore is not None:
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        if count == 0:
            print(f"[QuickLoan] Vectorstore opened but is EMPTY (0 chunks).")
            print("  Run 'python data/ingest.py' from the project root to populate it.")
        else:
            print(f"[QuickLoan] Vectorstore ready — {count} chunks loaded.")
    except Exception as e:
        print(f"[QuickLoan] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


def classify(state: QuickLoanState) -> dict:
    """Classify the customer question. Provided -- no changes needed."""
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state["customer_message"]),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "SIMPLE"
    except Exception as e:
        print(f"[QuickLoan] Classification error: {e}")
        query_type = "SIMPLE"
    return {"query_type": query_type}


def retrieve_docs(state: QuickLoanState) -> dict:
    """Query ChromaDB for policy chunks relevant to the customer's question.

    vectorstore.similarity_search() returns LangChain Document objects.
    Each Document has .page_content (the text) and .metadata (dict with 'source').
    """
    # -----------------------------------------------------------------------
    # TODO 2 of 4 -- Implement retrieve_docs()
    # -----------------------------------------------------------------------
    # 1. Call _init_vectorstore() to ensure the vectorstore is loaded.
    #
    # 2. If vectorstore is None (ingest.py not yet run), return {"retrieved_docs": []}.
    #
    # 3. Inside a try/except block:
    #      docs = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
    #      retrieved = [
    #          f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
    #          for doc in docs
    #      ]
    #    On exception: print the error, set retrieved = []
    #
    # 4. Return {"retrieved_docs": retrieved}
    # -----------------------------------------------------------------------
    # TODO: implement this node
    return {"retrieved_docs": []}


def respond(state: QuickLoanState) -> dict:
    """Handle SIMPLE queries, enriched with retrieved document context."""
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    # -----------------------------------------------------------------------
    # TODO 3 of 4 -- Build system_content from retrieved docs
    # -----------------------------------------------------------------------
    # If retrieved is non-empty, prepend the chunks to the system prompt:
    #
    #   context_block  = "\n\n---\n\n".join(retrieved)
    #   system_content = (
    #       SYSTEM_PROMPT
    #       + "\n\nThe following sections from FastFinance India's policy documents are relevant "
    #       "to the customer's question. Use this information in your answer:\n\n"
    #       + context_block
    #   )
    #
    # Otherwise:
    #   system_content = SYSTEM_PROMPT
    #
    # Then replace SYSTEM_PROMPT with system_content in the line below.
    # -----------------------------------------------------------------------
    # TODO: replace SYSTEM_PROMPT with system_content (built from retrieved)
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
    """Handle COMPLEX queries. Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: QuickLoanState) -> dict:
    """Handle OUT_OF_SCOPE queries. Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: QuickLoanState) -> str:
    """Route after classify(). Session 4: SIMPLE now routes to retrieve_docs."""
    qt = state.get("query_type", "SIMPLE")
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "respond"  # TODO 4: change "respond" to "retrieve_docs"
