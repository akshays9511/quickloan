"""
app.py
------
Streamlit chat UI for QuickLoan — FastFinance India's AI loan assistant.

Session 15: Cloud Deployment.

What changed from S14b:
  1. Graceful API key check — shows a Streamlit error instead of a Python
     traceback when GROQ_API_KEY is missing (required for cloud deployment).
  2. Guard reverted to regex-only (PII + injection) — the LlamaGuard 3 8B /
     Ollama layer from S14b was dropped so the app has no external service
     dependency and can run in a container with only GROQ_API_KEY set.

Run locally:
    streamlit run app.py   (from inside s15/solution/)

Deploy to Streamlit Community Cloud:
    Push this directory to GitHub. In the Streamlit Cloud dashboard, set
    GROQ_API_KEY under Settings → Secrets.
"""
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# ---------------------------------------------------------------------------
# S15: Graceful API key check
#
# Validate GROQ_API_KEY BEFORE importing quickloan. The package reads the key
# at module load time (config.py raises ValueError if it is missing). Without
# this guard, a missing key on a cloud host shows a raw Python traceback to
# the user instead of an actionable error message.
# ---------------------------------------------------------------------------
if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "⚠️ **GROQ_API_KEY is not configured.**\n\n"
        "- **Streamlit Community Cloud:** go to *Settings → Secrets* and add:\n"
        "  ```\n  GROQ_API_KEY = \"gsk_...\"\n  ```\n"
        "- **Local Docker:** `docker run -p 8501:8501 -e GROQ_API_KEY=gsk_... quickloan`  \n"
        "  or `docker run -p 8501:8501 --env-file .env quickloan`\n"
        "- **Local dev:** copy `.env.example` to `.env` and fill in your key"
    )
    st.stop()

from quickloan.agent import build_graph  # noqa: E402
from quickloan.config import CHECKPOINT_DB, DB_PATH  # noqa: E402
from quickloan.emi import calculate_emi_breakdown  # noqa: E402
import quickloan.nodes as _nodes        # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402


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
# S15: EMI Calculator tab
#
# A dedicated form alongside the chat, for customers who want to explore
# numbers directly instead of asking in natural language. It reads real loan
# products/rate slabs straight from fastfinance_data.db (the same database
# the chat agent's query_rates tool uses) so the dropdowns only ever offer
# rates FastFinance actually provides -- and it calls the exact same
# calculate_emi_breakdown() function the chat agent's calculate_emi MCP tool
# calls, so the two surfaces can never disagree on the math.
# ---------------------------------------------------------------------------

def _get_loan_products() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT product_id, product_name, min_tenure_months, max_tenure_months, max_loan_amount "
        "FROM loan_products ORDER BY product_name"
    ).fetchall()
    conn.close()
    return [
        {
            "product_id":  r[0],
            "product_name": r[1],
            "min_tenure":  r[2],
            "max_tenure":  r[3],
            "max_amount":  r[4],
        }
        for r in rows
    ]


def _get_rate_slabs(product_id: str) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT min_cibil, max_cibil, annual_rate_pct FROM rate_slabs "
        "WHERE product_id = ? ORDER BY min_cibil DESC",
        (product_id,),
    ).fetchall()
    conn.close()
    return [{"min_cibil": r[0], "max_cibil": r[1], "rate": r[2]} for r in rows]


def _slab_label(slab: dict) -> str:
    if slab["min_cibil"] <= 0 and slab["max_cibil"] >= 900:
        return f"Flat rate — {slab['rate']:.2f}% p.a."
    return f"CIBIL {slab['min_cibil']}-{slab['max_cibil']} — {slab['rate']:.2f}% p.a."


def _emi_calculator_tab() -> None:
    st.subheader("🧮 EMI Calculator")
    st.caption("Rates and loan limits below are read live from FastFinance's database.")

    products = _get_loan_products()
    product_by_name = {p["product_name"]: p for p in products}
    product_name = st.selectbox("Loan type", list(product_by_name.keys()))
    product = product_by_name[product_name]

    slabs = _get_rate_slabs(product["product_id"])
    slab_by_label = {_slab_label(s): s for s in slabs}
    slab_label = st.selectbox("Interest rate (by CIBIL score)", list(slab_by_label.keys()))
    slab = slab_by_label[slab_label]

    col1, col2 = st.columns(2)
    principal = col1.number_input(
        "Loan amount (Rs.)",
        min_value=10_000,
        max_value=int(product["max_amount"]),
        value=min(500_000, int(product["max_amount"])),
        step=10_000,
        help=f"Maximum for {product_name}: Rs. {product['max_amount']:,}",
    )
    tenure_months = col2.number_input(
        "Tenure (months)",
        min_value=int(product["min_tenure"]),
        max_value=int(product["max_tenure"]),
        value=int((product["min_tenure"] + product["max_tenure"]) // 2),
        step=1,
        help=f"{product_name} tenure range: {product['min_tenure']}-{product['max_tenure']} months",
    )

    if st.button("Calculate EMI", type="primary", use_container_width=True):
        try:
            breakdown = calculate_emi_breakdown(principal, slab["rate"], int(tenure_months))
        except ValueError as e:
            st.error(str(e))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Monthly EMI", f"Rs. {breakdown['emi']:,.2f}")
            m2.metric("Total Payment", f"Rs. {breakdown['total_payment']:,.2f}")
            m3.metric("Total Interest", f"Rs. {breakdown['total_interest']:,.2f}")
            st.caption(
                f"{product_name} · {slab_label} · {int(tenure_months)} months · "
                "Pre-qualification estimate only, subject to final approval."
            )


# ---------------------------------------------------------------------------
# S15: Persistent conversations
#
# MemorySaver (S13/S14) discarded every thread the moment the graph object
# was rebuilt -- which "New Conversation" did on every click. Switching to
# SqliteSaver (same checkpoint DB the CLI already uses) means a thread's
# state survives "New Conversation", page reloads, and app restarts.
#
# LangGraph's checkpoint tables aren't meant to be queried for a
# human-readable thread list, so a small sidecar table (conversations)
# tracks thread_id -> title/timestamp purely for the sidebar picker.
# ---------------------------------------------------------------------------

def _ensure_conversations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversations ("
        "  thread_id  TEXT PRIMARY KEY,"
        "  title      TEXT NOT NULL,"
        "  created_at TEXT NOT NULL"
        ")"
    )
    # Route captions ("Route: RATES → rates_agent | ✅ RBI Compliant") are
    # Streamlit-only display metadata -- the graph's checkpointed state
    # overwrites specialist/query_type/compliance_status each turn rather
    # than keeping history of them, so they can't be recovered from
    # graph.get_state(). This table is the only place they're kept.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversation_messages ("
        "  thread_id   TEXT NOT NULL,"
        "  seq         INTEGER NOT NULL,"
        "  role        TEXT NOT NULL,"
        "  content     TEXT NOT NULL,"
        "  route_label TEXT,"
        "  PRIMARY KEY (thread_id, seq)"
        ")"
    )
    conn.commit()


def _persist_message(
    conn: sqlite3.Connection, thread_id: str, seq: int, role: str, content: str, route_label: str = ""
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO conversation_messages (thread_id, seq, role, content, route_label) "
        "VALUES (?, ?, ?, ?, ?)",
        (thread_id, seq, role, content, route_label),
    )
    conn.commit()


def _record_conversation_start(conn: sqlite3.Connection, thread_id: str, first_message: str) -> None:
    title = first_message.strip().replace("\n", " ")[:60]
    conn.execute(
        "INSERT OR IGNORE INTO conversations (thread_id, title, created_at) VALUES (?, ?, ?)",
        (thread_id, title, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _list_past_conversations(conn: sqlite3.Connection, exclude_thread_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT thread_id, title, created_at FROM conversations "
        "WHERE thread_id != ? ORDER BY created_at DESC LIMIT 20",
        (exclude_thread_id,),
    ).fetchall()
    return [{"thread_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


def _load_conversation(thread_id: str) -> None:
    """Rebuild the displayed transcript (messages + route captions) for an
    existing thread from conversation_messages, and make it the active thread."""
    rows = st.session_state.db_conn.execute(
        "SELECT role, content, route_label FROM conversation_messages "
        "WHERE thread_id = ? ORDER BY seq",
        (thread_id,),
    ).fetchall()

    st.session_state.thread_id = thread_id
    st.session_state.messages  = [{"role": role, "content": content} for role, content, _ in rows]
    st.session_state.routes    = [route or "" for role, _, route in rows if role == "assistant"]
    st.session_state.pop("pending_hitl", None)


def _init_session() -> None:
    if "graph" not in st.session_state:
        conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
        _ensure_conversations_table(conn)
        st.session_state.db_conn   = conn
        st.session_state.graph     = build_graph(checkpointer=SqliteSaver(conn))
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages  = []
        st.session_state.routes    = []


def _sidebar() -> None:
    with st.sidebar:
        st.header("💰 QuickLoan")
        st.caption("FastFinance AI Loan Assistant")
        st.divider()

        if st.button("🆕 New Conversation", use_container_width=True):
            # Keep "graph"/"db_conn" alive -- they hold the persistent
            # SqliteSaver connection. Only reset which thread is active.
            for key in ["thread_id", "messages", "routes", "pending_hitl"]:
                st.session_state.pop(key, None)
            st.session_state.thread_id = str(uuid4())
            st.session_state.messages  = []
            st.session_state.routes    = []
            st.rerun()

        if "thread_id" in st.session_state:
            st.caption(f"Session: {st.session_state.thread_id[:8]}…")

        st.divider()
        st.subheader("Past Conversations")
        past = _list_past_conversations(st.session_state.db_conn, st.session_state.thread_id)
        if not past:
            st.caption("No past conversations yet.")
        else:
            for conv in past:
                label = conv["title"] or "(untitled)"
                if st.button(f"💬 {label}", key=f"conv_{conv['thread_id']}", use_container_width=True):
                    _load_conversation(conv["thread_id"])
                    st.rerun()

        st.divider()
        st.subheader("Agents")
        st.markdown(
            "- **Guard** — blocks injections & PII\n"
            "- **Supervisor** — classifies clean queries\n"
            "- **Rates Agent** — live interest rates & EMI calculator via MCP\n"
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
        _persist_message(
            st.session_state.db_conn, st.session_state.thread_id,
            len(st.session_state.messages) - 1, "assistant", edited, pending["route_label"],
        )
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
    st.caption("AI-powered loan assistant — Session 15: Cloud Deployment")

    _init_session()
    _sidebar()

    tab_chat, tab_emi = st.tabs(["💬 Chat", "🧮 EMI Calculator"])

    with tab_chat:
        _render_history()

        hitl_active = _handle_hitl()

        if not hitl_active:
            prompt = st.chat_input("Ask about loan rates, eligibility, or our policies…")
            if prompt:
                if not st.session_state.messages:
                    _record_conversation_start(st.session_state.db_conn, st.session_state.thread_id, prompt)

                st.session_state.messages.append({"role": "user", "content": prompt})
                _persist_message(
                    st.session_state.db_conn, st.session_state.thread_id,
                    len(st.session_state.messages) - 1, "user", prompt,
                )
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
                    _persist_message(
                        st.session_state.db_conn, st.session_state.thread_id,
                        len(st.session_state.messages) - 1, "assistant", result["response"], route_label,
                    )
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
                    _persist_message(
                        st.session_state.db_conn, st.session_state.thread_id,
                        len(st.session_state.messages) - 1, "assistant", response, route_label,
                    )

    with tab_emi:
        _emi_calculator_tab()


if __name__ == "__main__":
    main()
