"""
app.py
------
STARTER FILE -- Session 13: Streamlit Frontend.

What is already provided (no changes needed):
  - compliance_badge()   returns a display badge string for compliance_status
  - format_route_label() formats the routing info into one line
  - is_escalated()       returns True when specialist == "escalated"
  - _init_session()      initialises graph + session state on first load
  - _sidebar()           renders the sidebar with agent descriptions
  - _render_history()    renders previous chat turns
  - main()               wires everything together

Your task (3 TODOs):
  TODO 1: Implement build_input_state(message)
          Return the dict that graph.invoke() expects as its first argument.
          Fields: customer_message, response, specialist, retrieved_docs, compliance_status

  TODO 2: Implement get_thread_config(thread_id)
          Return {"configurable": {"thread_id": thread_id}}

  TODO 3: In main(), complete the response display block
          Show the response with st.chat_message("assistant")
          Use st.warning() for escalated responses, st.markdown() for others.
          Show the route caption below the message.
          Append response + route_label to session_state.

Run when done:
    streamlit run app.py   (from inside s13/starter/)
"""
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from quickloan.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# TODO 1 of 3 -- Implement build_input_state()
# ---------------------------------------------------------------------------
# Return a dict with these keys (the graph's initial state):
#   "customer_message":  message   (the customer's question)
#   "response":          ""        (empty -- agent fills this in)
#   "specialist":        ""        (empty -- supervisor fills this)
#   "retrieved_docs":    []        (empty list)
#   "compliance_status": ""        (empty -- compliance agent fills this)
#
# Template:
#   def build_input_state(message: str) -> dict:
#       return {
#           "customer_message":  message,
#           "response":          "",
#           "specialist":        "",
#           "retrieved_docs":    [],
#           "compliance_status": "",
#       }
# ---------------------------------------------------------------------------
def build_input_state(message: str) -> dict:
    raise NotImplementedError("TODO 1: implement build_input_state()")


# ---------------------------------------------------------------------------
# TODO 2 of 3 -- Implement get_thread_config()
# ---------------------------------------------------------------------------
# LangGraph needs a thread ID to keep memory across turns in the same session.
# Return {"configurable": {"thread_id": thread_id}}
#
# Template:
#   def get_thread_config(thread_id: str) -> dict:
#       return {"configurable": {"thread_id": thread_id}}
# ---------------------------------------------------------------------------
def get_thread_config(thread_id: str) -> dict:
    raise NotImplementedError("TODO 2: implement get_thread_config()")


# ---------------------------------------------------------------------------
# Already implemented -- no changes needed for these
# ---------------------------------------------------------------------------

def compliance_badge(status: str) -> str:
    if status == "PASS":
        return "✅ RBI Compliant"
    if status == "REVISED":
        return "⚠️ Revised"
    if status.startswith("FAIL"):
        return "❌ Violation"
    return ""


def format_route_label(result: dict) -> str:
    qt    = result.get("query_type", "—")
    sp    = result.get("specialist", "—")
    cs    = result.get("compliance_status", "")
    badge = compliance_badge(cs)
    label = f"Route: {qt} → {sp}"
    if badge:
        label += f" | {badge}"
    return label


def is_escalated(result: dict) -> bool:
    return result.get("specialist", "") == "escalated"


def _init_session() -> None:
    if "graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        st.session_state.graph     = build_graph(checkpointer=MemorySaver())
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages  = []
        st.session_state.routes    = []


def _sidebar() -> None:
    with st.sidebar:
        st.header("💰 QuickLoan")
        st.caption("FastFinance AI Loan Assistant")
        st.divider()
        if st.button("🔄 New Conversation", use_container_width=True):
            for key in ["graph", "thread_id", "messages", "routes"]:
                st.session_state.pop(key, None)
            st.rerun()
        if "thread_id" in st.session_state:
            st.caption(f"Session: {st.session_state.thread_id[:8]}…")
        st.divider()
        st.subheader("Agents")
        st.markdown(
            "- **Supervisor** — classifies query\n"
            "- **Rates Agent** — live interest rates via MCP\n"
            "- **Policy Agent** — loan policy via RAG\n"
            "- **Compliance Agent** — RBI rules check\n"
            "- **Escalate** → officer / **Decline** → out-of-scope"
        )


def _render_history() -> None:
    messages = st.session_state.get("messages", [])
    routes   = st.session_state.get("routes",   [])
    assistant_idx = 0
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if assistant_idx < len(routes):
                st.caption(routes[assistant_idx])
            assistant_idx += 1


def main() -> None:
    st.set_page_config(page_title="QuickLoan | FastFinance", page_icon="💰", layout="wide")
    st.title("💰 QuickLoan | FastFinance")
    st.caption("AI-powered loan assistant — Session 13: Streamlit UI")

    _init_session()
    _sidebar()
    _render_history()

    prompt = st.chat_input("Ask about loan rates, eligibility, or our policies…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("QuickLoan is processing your query…"):
            result = st.session_state.graph.invoke(
                build_input_state(prompt),
                config=get_thread_config(st.session_state.thread_id),
            )

        response    = result["response"]
        route_label = format_route_label(result)

        # ---------------------------------------------------------------------------
        # TODO 3 of 3 -- Display the response
        # ---------------------------------------------------------------------------
        # 1. Open a chat_message block with role "assistant"
        # 2. Inside it: if is_escalated(result), use st.warning(response)
        #               otherwise, use st.markdown(response)
        # 3. After the block: show st.caption(route_label)
        # 4. Append {"role": "assistant", "content": response} to st.session_state.messages
        # 5. Append route_label to st.session_state.routes
        #
        # Template:
        #   with st.chat_message("assistant"):
        #       if is_escalated(result):
        #           st.warning(response)
        #       else:
        #           st.markdown(response)
        #   st.caption(route_label)
        #   st.session_state.messages.append({"role": "assistant", "content": response})
        #   st.session_state.routes.append(route_label)
        # ---------------------------------------------------------------------------
        raise NotImplementedError("TODO 3: display the response")


if __name__ == "__main__":
    main()
