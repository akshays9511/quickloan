"""
quickloan/nodes.py
------------------
STARTER FILE for Session 12: Multi-Agent Architecture Part 2.

The Session 10 multi-agent foundation (Policy Agent, Rates Agent, supervisor nodes)
is already implemented below. Your task is to add the Compliance Agent sub-graph.

TODOs:
  TODO 1 -- Implement create_compliance_agent()
  TODO 2 -- Implement call_compliance_agent() supervisor node
  TODO 3 -- Wire the compliance agent into build_graph() in agent.py

Architecture you are building:
  POLICY → call_policy_agent → call_compliance_agent → END
  RATES  → call_rates_agent  → call_compliance_agent → END
  COMPLEX → escalate → END
  OOS    → decline   → END

Compliance sub-graph (create_compliance_agent):
  check_rbi → [FAIL: revise | PASS: END]
"""
import re
import sqlite3

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable
from langgraph.graph import END, StateGraph

from .config import (
    CLASSIFY_SYSTEM,
    DB_PATH,
    DECLINE_RESPONSE,
    EMBED_MODEL,
    ESCALATE_RESPONSE,
    POLICY_SYSTEM_PROMPT,
    QUICKLOAN_BANNED_PHRASES,
    RETRIEVAL_K,
    SAFE_COMPLIANCE_RESPONSE,
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
# Specialist agent node functions  (provided -- no changes needed)
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
                print(f"[QuickLoan] Rates Agent MCP: {tc['name']}({tc['args']}) -> {str(tool_output)[:80]}")
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
# Specialist agent factory functions  (provided -- no changes needed)
# ---------------------------------------------------------------------------

def create_policy_agent():
    builder = StateGraph(QuickLoanState)
    builder.add_node("retrieve_docs", _policy_retrieve)
    builder.add_node("respond",       _policy_respond)
    builder.set_entry_point("retrieve_docs")
    builder.add_edge("retrieve_docs", "respond")
    builder.add_edge("respond",       END)
    return builder.compile()


def create_rates_agent():
    builder = StateGraph(QuickLoanState)
    builder.add_node("respond", _rates_respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)
    return builder.compile()


_policy_agent = create_policy_agent()
_rates_agent  = create_rates_agent()


# ---------------------------------------------------------------------------
# Compliance helpers  (provided -- no changes needed)
# ---------------------------------------------------------------------------

def _load_valid_rates() -> set:
    try:
        conn  = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        rows  = conn.execute("SELECT annual_rate_pct FROM rate_slabs").fetchall()
        conn.close()
        return {row[0] for row in rows}
    except Exception:
        return set()


def _extract_rates(text: str) -> list:
    matches = re.findall(r"(\d+\.?\d*)\s*%\s*p\.a\.", text, re.IGNORECASE)
    return [float(m) for m in matches]


@traceable(name="rbi_compliance_check")
def _check_compliance_logic(draft: str) -> tuple:
    lower = draft.lower()
    for phrase in QUICKLOAN_BANNED_PHRASES:
        if phrase in lower:
            return False, f"banned phrase: '{phrase}'"
    mentioned_rates = _extract_rates(draft)
    if mentioned_rates:
        valid_rates = _load_valid_rates()
        if valid_rates:
            for rate in mentioned_rates:
                if rate not in valid_rates:
                    return False, f"hallucinated rate: {rate}% p.a. not in database"
    return True, "PASS"


# ---------------------------------------------------------------------------
# Compliance Agent node functions  (provided -- no changes needed)
# ---------------------------------------------------------------------------

def check_rbi(state: QuickLoanState) -> dict:
    draft          = state["response"]
    passed, reason = _check_compliance_logic(draft)
    if not passed:
        print(f"[QuickLoan] Compliance FAIL: {reason}")
        return {"compliance_status": f"FAIL: {reason}"}
    print("[QuickLoan] Compliance PASS")
    return {"compliance_status": "PASS"}


def revise_response(state: QuickLoanState) -> dict:
    draft  = state["response"]
    reason = state.get("compliance_status", "violation").replace("FAIL: ", "")
    prompt = (
        "You are a FastFinance India compliance officer reviewing an AI loan assistant response.\n\n"
        f"The response was flagged for: {reason}\n\n"
        "Rewrite it to fix the violation while keeping the response helpful.\n\n"
        "Rules:\n"
        "  1. Never guarantee loan approval or imply a decision has been made.\n"
        "  2. Only state interest rates that appeared in the original -- do not add new ones.\n"
        "  3. Keep the rewritten response under 150 words.\n"
        "  4. End with 'QuickLoan | FastFinance India'\n\n"
        f"Original response:\n{draft}\n\n"
        "Compliant rewrite:"
    )
    try:
        result       = llm.invoke([HumanMessage(content=prompt)])
        revised_text = result.content.strip() or SAFE_COMPLIANCE_RESPONSE
    except Exception as e:
        print(f"[QuickLoan] Compliance Agent revision error: {e}")
        revised_text = SAFE_COMPLIANCE_RESPONSE
    print("[QuickLoan] Compliance Agent: response revised")
    return {"response": revised_text, "compliance_status": "REVISED"}


def route_compliance(state: QuickLoanState) -> str:
    return "revise" if state.get("compliance_status", "").startswith("FAIL") else END


# ---------------------------------------------------------------------------
# TODO 1 of 2 -- Implement create_compliance_agent()
# ---------------------------------------------------------------------------
# Build a sub-graph with two nodes: "check_rbi" and "revise".
#
# Graph structure:
#   entry → check_rbi
#   check_rbi → conditional(route_compliance):
#     if "FAIL": → "revise"
#     else:      → END
#   revise → END
#
# Template:
#   def create_compliance_agent():
#       builder = StateGraph(QuickLoanState)
#       builder.add_node("check_rbi", check_rbi)
#       builder.add_node("revise",    revise_response)
#       builder.set_entry_point("check_rbi")
#       builder.add_conditional_edges(
#           "check_rbi",
#           route_compliance,
#           {"revise": "revise", END: END},
#       )
#       builder.add_edge("revise", END)
#       return builder.compile()
# ---------------------------------------------------------------------------
def create_compliance_agent():
    raise NotImplementedError("TODO 1: implement create_compliance_agent()")

_compliance_agent = None  # TODO 1: set to create_compliance_agent()


# ---------------------------------------------------------------------------
# TODO 2 of 2 -- Implement call_compliance_agent() supervisor node
# ---------------------------------------------------------------------------
# This is the supervisor's node that calls the compliance sub-graph.
# It must:
#   1. Call _compliance_agent.invoke({...}) passing state["response"] and all
#      other required state fields (compliance_status starts as "").
#   2. Return {"response": ..., "compliance_status": ...} from the result.
# ---------------------------------------------------------------------------
def call_compliance_agent(state: QuickLoanState) -> dict:
    raise NotImplementedError("TODO 2: implement call_compliance_agent()")


# ---------------------------------------------------------------------------
# Supervisor nodes  (provided -- no changes needed)
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


def call_policy_agent(state: QuickLoanState) -> dict:
    print("[QuickLoan] Supervisor → Policy Agent")
    result = _policy_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "POLICY"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
    })
    return {
        "response":       result["response"],
        "retrieved_docs": result.get("retrieved_docs", []),
        "history":        result.get("history", state.get("history", [])),
        "specialist":     "policy_agent",
    }


def call_rates_agent(state: QuickLoanState) -> dict:
    print("[QuickLoan] Supervisor → Rates Agent")
    result = _rates_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "RATES"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "rates_agent",
    }


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


def route_supervisor(state: QuickLoanState) -> str:
    qt = state.get("query_type", "RATES")
    if qt == "POLICY":
        return "call_policy_agent"
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "call_rates_agent"
