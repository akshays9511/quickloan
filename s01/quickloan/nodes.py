"""
quickloan/nodes.py
------------------
Node functions for the QuickLoan graph.

Each node is a plain Python function:
  - Input : the full QuickLoanState (read-only)
  - Output: a dict containing ONLY the keys this node changed
             (LangGraph merges it into the state automatically)
"""
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .config import (
    SYSTEM_PROMPT,CLASSIFY_SYSTEM_PROMPT,ESCALATE_RESPONSE,DECLINE_RESPONSE,
    EMBED_MODEL,VECTORSTORE_DIR, RETRIEVAL_K,RETRIEVAL_SCORE_THRESHOLD
)
from .state import QuickLoanState
from .tools import llm,classifier_llm

vectorstore = None  # shared across calls; initialised once by _init_vectorstore()
 
 
def _init_vectorstore() -> None:
    global vectorstore
    if vectorstore is not None:  # already loaded — skip the 90 MB model reload
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)  # loads ~90 MB model from ~/.cache/huggingface/
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),  # opens chroma.sqlite3 on disk — does NOT load all chunks into memory
            embedding_function=embeddings,            # same model used at ingest time — must match or retrieval breaks
        )
    except Exception as e:
        print(f"[Quickloan] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")
 
def classify(state: QuickLoanState) -> dict:
    valid_types = {"IN_SCOPE", "OUT_OF_SCOPE"}
    messages= [
            SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=state["customer_message"]),
        ]
    try:
        result = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in valid_types:
            query_type = "IN_SCOPE"
    except Exception as e:
        print(f"[WealthDesk] Classification error: {e}")
        query_type = "IN_SCOPE"
 
    return {"query_type": query_type, "retrieved_docs": []}
 
 
def retrieve_docs(state: QuickLoanState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        # similarity_search_with_relevance_scores returns a list of (doc, score) tuples.
        #   doc.page_content : the raw chunk text (e.g. "PAN card and Aadhaar are required...")
        #   doc.metadata     : dict — e.g. {"source": "home_loan_guide.md"}
        #   score            : cosine similarity 0–1; higher = more relevant to the query
        results   = vectorstore.similarity_search_with_relevance_scores(
            state["customer_message"], k=RETRIEVAL_K
        )
        retrieved = []
        for doc, score in results:  # doc = LangChain Document object; score = float 0–1
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                retrieved.append(
                    f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                )
            else:
                print(
                    f"[WealthDesk] Chunk skipped (score {score:.2f} < {RETRIEVAL_SCORE_THRESHOLD}): "
                    f"{doc.metadata.get('source', 'unknown')}"
                )
    except Exception as e:
        print(f"[WealthDesk] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}
    
   
def respond(state: QuickLoanState) -> dict:

    history= state.get('history',[])
    retrieved = state.get("retrieved_docs", [])
    if not retrieved:
        new_history = history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": ESCALATE_RESPONSE},
        ]
        return {"response": ESCALATE_RESPONSE, "history": new_history}
    context_block  = "\n\n---\n\n".join(retrieved)
    system_content = (
        SYSTEM_PROMPT
        + "\n\nThe following sections from BNB's policy documents are relevant "
        "to the customer's question. Use this information in your answer:\n\n"
        + context_block
    )
    messages = [
        SystemMessage(content=system_content)
    ]
 
    for turn in history:
        if(turn["role"]=="user"):
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result=llm.invoke(messages)
        response_text=result.content
    except Exception as e:
        print("LLM error:" ,e)
        return {"response":"Agent is unavailable"}
    new_history=history+[{"role":"user","content":state["customer_message"]},
    {"role":"assistant","content":response_text}]
    return {"response":response_text,"history":new_history}

def escalate(state: QuickLoanState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}

def decline(state: QuickLoanState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}
 
def route_query(state: QuickLoanState)->str:
   query_type = state.get("query_type","IN_SCOPE")
   if query_type == "OUT_OF_SCOPE":
      return "decline"
   return "retrieve_docs" 
