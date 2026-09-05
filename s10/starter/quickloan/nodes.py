"""
quickloan/nodes.py
------------------
STARTER FILE -- your task is to implement the three TODO sections below.

Goal
  Refactor the QuickLoan agent into a Supervisor + Specialist Agent architecture.

What is already provided
  - _init_vectorstore(), _policy_retrieve(), _policy_respond(), _rates_respond()
    (the internal node functions for each specialist agent)
  - classify(), escalate(), decline() supervisor utility nodes
  - All imports and state type

Your task
  TODO 1: Implement the agent factory functions create_policy_agent() and
          create_rates_agent(), then instantiate them.
  TODO 2: Implement the supervisor caller nodes call_policy_agent() and
          call_rates_agent() that invoke the sub-agents and merge results.
  TODO 3: Implement route_supervisor() to map query_type to the correct node name.

Run when done
  python -m quickloan.agent   (from inside s10/starter/)
"""
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph

from .config import (
    CLASSIFY_SYSTEM,
    DECLINE_RESPONSE,
    EMBED_MODEL,
    ESCALATE_RESPONSE,
    POLICY_SYSTEM_PROMPT,
    RETRIEVAL_K,
    SYSTEM_PROMPT,
    VECTORSTORE_DIR,
)
from .state import QuickLoanState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools

vectorstore = None


def _init_vectorstore() -> None:
    global vectorstore
    if vectorstore is not None:
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
    except Exception as e:
        print(f"[QuickLoan] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


# ---------------------------------------------------------------------------
# Specialist agent node functions (provided -- no changes needed)
# ---------------------------------------------------------------------------

def _policy_retrieve(state: QuickLoanState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        docs = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
        return {
            "retrieved_docs": [
                f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ]
        }
    except Exception as e:
        print(f"[QuickLoan] Policy Agent retrieval error: {e}")
        return {"retrieved_docs": []}


def _policy_respond(state: QuickLoanState) -> dict:
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    context_block  = "\n\n---\n\n".join(retrieved) if retrieved else ""
    system_content = (
        POLICY_SYSTEM_PROMPT
        + (
            "\n\nThe following sections from FastFinance's policy documents are relevant "
            "to the customer's question. Use this information in your answer:\n\n"
            + context_block
            if context_block else ""
        )
    )

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"])
        )
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result        = llm.invoke(messages)
        response_text = result.content
    except Exception as e:
        print(f"[QuickLoan] Policy Agent LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    return {
        "response": response_text,
        "history":  history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response_text},
        ],
    }


def _rates_respond(state: QuickLoanState) -> dict:
    history  = state.get("history", [])
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"])
        )
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm_with_tools.invoke(messages)

        if result.tool_calls:
            messages.append(result)
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(
                    f"[QuickLoan] Rates Agent MCP: {tc['name']}({tc['args']}) "
                    f"-> {str(tool_output)[:80]}"
                )
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
            result = llm.invoke(messages)

        response_text = result.content

    except Exception as e:
        print(f"[QuickLoan] Rates Agent LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    return {
        "response": response_text,
        "history":  history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response_text},
        ],
    }


# ---------------------------------------------------------------------------
# TODO 1 of 3 -- Implement the agent factory functions
# ---------------------------------------------------------------------------
# create_policy_agent(): StateGraph with retrieve_docs → respond → END
# create_rates_agent():  StateGraph with respond → END
#
# Template:
#   def create_policy_agent():
#       builder = StateGraph(QuickLoanState)
#       builder.add_node("retrieve_docs", _policy_retrieve)
#       builder.add_node("respond",       _policy_respond)
#       builder.set_entry_point("retrieve_docs")
#       builder.add_edge("retrieve_docs", "respond")
#       builder.add_edge("respond",       END)
#       return builder.compile()
#
#   def create_rates_agent():
#       builder = StateGraph(QuickLoanState)
#       builder.add_node("respond", _rates_respond)
#       builder.set_entry_point("respond")
#       builder.add_edge("respond", END)
#       return builder.compile()
# ---------------------------------------------------------------------------
def create_policy_agent():
    raise NotImplementedError("TODO 1: implement create_policy_agent()")


def create_rates_agent():
    raise NotImplementedError("TODO 1: implement create_rates_agent()")


_policy_agent = None   # TODO 1: set to create_policy_agent()
_rates_agent  = None   # TODO 1: set to create_rates_agent()


# ---------------------------------------------------------------------------
# Supervisor nodes (provided -- no changes needed)
# ---------------------------------------------------------------------------

def classify(state: QuickLoanState) -> dict:
    messages = [SystemMessage(content=CLASSIFY_SYSTEM)]
    for turn in state.get("history", [])[-2:]:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"])
        )
    messages.append(HumanMessage(content=state["customer_message"]))
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"RATES", "POLICY", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "RATES"
    except Exception as e:
        print(f"[QuickLoan] Supervisor classification error: {e}")
        query_type = "RATES"
    return {"query_type": query_type}


# ---------------------------------------------------------------------------
# TODO 2 of 3 -- Implement call_policy_agent() and call_rates_agent()
# ---------------------------------------------------------------------------
# These are the supervisor's nodes that invoke each specialist sub-agent.
# Each must:
#   1. Print a routing message like: print("[QuickLoan] Supervisor → Policy Agent")
#   2. Call _policy_agent.invoke({...}) with the current state fields
#   3. Return the specialist's response, history, retrieved_docs, and set "specialist" field
#
# Template for call_policy_agent:
#   def call_policy_agent(state: QuickLoanState) -> dict:
#       print("[QuickLoan] Supervisor → Policy Agent")
#       result = _policy_agent.invoke({
#           "customer_message": state["customer_message"],
#           "history":          state.get("history", []),
#           "response":         "",
#           "query_type":       state.get("query_type", "POLICY"),
#           "retrieved_docs":   [],
#           "specialist":       "",
#       })
#       return {
#           "response":       result["response"],
#           "retrieved_docs": result.get("retrieved_docs", []),
#           "history":        result.get("history", state.get("history", [])),
#           "specialist":     "policy_agent",
#       }
# ---------------------------------------------------------------------------
def call_policy_agent(state: QuickLoanState) -> dict:
    raise NotImplementedError("TODO 2: implement call_policy_agent()")


def call_rates_agent(state: QuickLoanState) -> dict:
    raise NotImplementedError("TODO 2: implement call_rates_agent()")


def escalate(state: QuickLoanState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history, "specialist": "escalated"}


def decline(state: QuickLoanState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history, "specialist": "declined"}


# ---------------------------------------------------------------------------
# TODO 3 of 3 -- Implement route_supervisor()
# ---------------------------------------------------------------------------
# Map query_type to the correct node name:
#   POLICY       → "call_policy_agent"
#   COMPLEX      → "escalate"
#   OUT_OF_SCOPE → "decline"
#   default      → "call_rates_agent"
# ---------------------------------------------------------------------------
def route_supervisor(state: QuickLoanState) -> str:
    raise NotImplementedError("TODO 3: implement route_supervisor()")
