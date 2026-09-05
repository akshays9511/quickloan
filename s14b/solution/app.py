"""
app.py
------
Streamlit chat UI for QuickLoan — FastFinance's AI loan assistant.

Session 14b: same UI as S14 but uses LlamaGuard 3 8B instead of Prompt Guard 2.
guard_badge no longer shows a score (LlamaGuard 3 returns category codes, not floats).

Run:
    streamlit run app.py   (from inside s14b/solution/)
"""
import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from quickloan.agent import build_graph  # noqa: E402
import quickloan.nodes as _nodes        # noqa: E402


class _StreamingState:
    def __init__(self, placeholder, token_delay: float = 0.0) -> None:
        self._placeholder = placeholder
        self._text = ""
        self._delay = token_delay

    def __call__(self, token: str) -> None:
        self._text += token
        self._placeholder.markdown(self._text + "▌")
        if self._delay > 0:
            time.sleep(self._delay)

    @property
    def text(self) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Helper functions (pure, testable -- no Streamlit calls)
# ---------------------------------------------------------------------------

def build_input_state(message: str) -> dict:
    """Return the initial state dict for graph.invoke()."""
    return {
        "customer_message":  message,
        "response":          "",
        "specialist":        "",
        "retrieved_docs":    [],
        "compliance_status": "",
        "blocked_reason":    "",
    }


def get_thread_config(thread_id: str) -> dict:
    """Return the LangGraph thread config dict."""
    return {"configurable": {"thread_id": thread_id}}


def compliance_badge(status: str) -> str:
    """Return a short human-readable badge for the RBI compliance status."""
    if status == "PASS":
        return "✅ RBI Compliant"
    if status == "REVISED":
        return "⚠️ Revised"
    if status.startswith("FAIL"):
        return "❌ Violation"
    return ""


def guard_badge(blocked_reason: str) -> str:
    """Return a guard status badge."""
    if blocked_reason == "pii":
        return "🔒 Blocked (PII)"
    if blocked_reason == "llamaguard":
        return "🤖 Blocked (LlamaGuard)"
    if blocked_reason:
        return "🛡️ Blocked (injection)"
    return ""


def needs_human_review(result: dict) -> bool:
    """Return True when the Compliance Agent revised the response."""
    return result.get("compliance_status", "") == "REVISED"


def format_route_label(result: dict) -> str:
    """Return a one-line route summary for display as a caption."""
    blocked_r = result.get("blocked_reason", "")
    if blocked_r:
        return f"Guard: {guard_badge(blocked_r)}"

    qt    = result.get("query_type", "—")
    sp    = result.get("specialist", "—")
    cs    = result.get("compliance_status", "")
    badge = compliance_badge(cs)
    label = f"Route: {qt} → {sp}"
    if badge:
        label += f" | {badge}"
    return label


def is_escalated(result: dict) -> bool:
    """Return True when the query was escalated to a loan officer."""
    return result.get("specialist", "") == "escalated"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

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
            for key in ["graph", "thread_id", "messages", "routes", "pending_hitl"]:
                st.session_state.pop(key, None)
            st.rerun()

        if "thread_id" in st.session_state:
            st.caption(f"Session: {st.session_state.thread_id[:8]}…")

        st.divider()
        st.subheader("Agents")
        st.markdown(
            "- **Guard** — LlamaGuard 3 8B (13 safety categories) *(upgraded in S14b)*\n"
            "- **Supervisor** — classifies clean queries\n"
            "- **Rates Agent** — live interest rates via MCP\n"
            "- **Policy Agent** — loan policy via RAG\n"
            "- **Compliance Agent** — RBI rules check\n"
            "- **Human-in-the-Loop** — reviews revisions"
        )

        st.divider()
        st.subheader("Demo settings")
        st.session_state["token_delay"] = st.slider(
            "Token delay (ms)",
            min_value=0, max_value=100, value=st.session_state.get("token_delay", 0),
            step=5,
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


def _handle_hitl() -> bool:
    if "pending_hitl" not in st.session_state:
        return False

    pending = st.session_state.pending_hitl
    st.warning(
        "⚠️ **Compliance Review Required** — The Compliance Agent revised this response. "
        "Please review and approve before sending to the customer."
    )

    with st.form("hitl_approval"):
        edited = st.text_area(
            "Review and edit the response if needed:",
            value=pending["response"],
            height=220,
        )
        col1, col2 = st.columns(2)
        approved  = col1.form_submit_button("✅ Approve & Send", use_container_width=True)
        discarded = col2.form_submit_button("❌ Discard",         use_container_width=True)

    if approved:
        st.session_state.messages.append({"role": "assistant", "content": edited})
        st.session_state.routes.append(pending["route_label"])
        del st.session_state.pending_hitl
        st.rerun()
    elif discarded:
        del st.session_state.pending_hitl
        st.rerun()

    return True


def main() -> None:
    st.set_page_config(
        page_title="QuickLoan | FastFinance",
        page_icon="💰",
        layout="wide",
    )
    st.title("💰 QuickLoan | FastFinance")
    st.caption("AI-powered loan assistant — Session 14b: LlamaGuard 3 8B")

    _init_session()
    _sidebar()
    _render_history()

    hitl_active = _handle_hitl()

    if not hitl_active:
        prompt = st.chat_input("Ask about loan rates, eligibility, or our policies…")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()

            delay_ms = st.session_state.get("token_delay", 0)
            streamer = _StreamingState(placeholder, token_delay=delay_ms / 1000)
            _nodes._stream_callback = streamer
            try:
                result = st.session_state.graph.invoke(
                    build_input_state(prompt),
                    config=get_thread_config(st.session_state.thread_id),
                )
            finally:
                _nodes._stream_callback = None

            route_label = format_route_label(result)
            blocked_r   = result.get("blocked_reason", "")

            if blocked_r:
                placeholder.warning(result["response"])
                st.caption(route_label)
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                st.session_state.routes.append(route_label)
            elif needs_human_review(result):
                placeholder.empty()
                st.session_state.pending_hitl = {
                    "response":    result["response"],
                    "route_label": route_label,
                }
                st.rerun()
            else:
                response = result["response"]
                if is_escalated(result):
                    placeholder.warning(response)
                else:
                    placeholder.markdown(response)
                st.caption(route_label)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.routes.append(route_label)


if __name__ == "__main__":
    main()
