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

import bcrypt
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
from quickloan.config import (  # noqa: E402
    CHECKPOINT_DB,
    DB_PATH,
    GRIEVANCE_OFFICER_CONTACT,
    RBI_OMBUDSMAN_NOTE,
)
from quickloan.emi import calculate_apr, calculate_emi_breakdown  # noqa: E402
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
        "compliance_reason": "",
        "original_response": "",
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
# Authentication
#
# The app has no notion of "user" until this point: every visitor shared the
# same conversations table with no owner column, so anyone's sidebar listed
# -- and could open -- every other visitor's chat history. This adds real
# accounts (bcrypt-hashed passwords) and scopes all conversation/audit
# queries by the logged-in username.
#
# Deliberately NOT using a cookie-persistence library: login is real, but it
# only survives st.session_state (i.e. the current browser tab's session) --
# a hard refresh or a new tab requires logging in again. That is a UX
# limitation, not a security hole; the leak this fixes is data crossing
# between different people, which this closes completely.
# ---------------------------------------------------------------------------

def _ensure_users_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "  username      TEXT PRIMARY KEY,"
        "  name          TEXT NOT NULL,"
        "  email         TEXT,"
        "  password_hash TEXT NOT NULL,"
        "  is_admin      INTEGER NOT NULL DEFAULT 0,"
        "  created_at    TEXT NOT NULL"
        ")"
    )
    conn.commit()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _any_users_exist(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def _get_user(conn: sqlite3.Connection, username: str) -> dict | None:
    row = conn.execute(
        "SELECT username, name, email, password_hash, is_admin FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if row is None:
        return None
    return {
        "username": row[0], "name": row[1], "email": row[2],
        "password_hash": row[3], "is_admin": bool(row[4]),
    }


def _create_user(conn: sqlite3.Connection, username: str, name: str, email: str, password: str) -> None:
    # First account registered on a fresh deployment becomes admin, so the
    # Compliance Audit Log (Section 5.4) has at least one viewer without
    # requiring manual DB surgery.
    is_admin = not _any_users_exist(conn)
    conn.execute(
        "INSERT INTO users (username, name, email, password_hash, is_admin, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (username, name, email, _hash_password(password), int(is_admin), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _auth_gate() -> dict:
    """Render a login/registration screen until the visitor is authenticated;
    returns the current user's {username, name, is_admin} once they are.

    Halts the script (st.stop()) while unauthenticated, so nothing past the
    call site in main() ever runs for a visitor who hasn't signed in.
    """
    if "auth_user" in st.session_state:
        return st.session_state.auth_user

    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _ensure_users_table(conn)

    st.title("💰 QuickLoan | FastFinance")
    st.caption("Sign in to continue -- your conversations are private to your account.")

    tab_login, tab_register = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            user = _get_user(conn, username.strip().lower())
            if user and _verify_password(password, user["password_hash"]):
                st.session_state.auth_user = {
                    "username": user["username"], "name": user["name"], "is_admin": user["is_admin"],
                }
                conn.close()
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with tab_register:
        with st.form("register_form"):
            reg_name      = st.text_input("Your name")
            reg_username  = st.text_input("Choose a username")
            reg_email     = st.text_input("Email (optional)")
            reg_password  = st.text_input("Choose a password", type="password")
            reg_password2 = st.text_input("Confirm password", type="password")
            reg_submitted = st.form_submit_button("Create account", use_container_width=True)
        if reg_submitted:
            uname = reg_username.strip().lower()
            if not reg_name.strip() or not uname or not reg_password:
                st.error("Name, username, and password are required.")
            elif reg_password != reg_password2:
                st.error("Passwords do not match.")
            elif len(reg_password) < 8:
                st.error("Password must be at least 8 characters.")
            elif _get_user(conn, uname):
                st.error("That username is already taken.")
            else:
                _create_user(conn, uname, reg_name.strip(), reg_email.strip(), reg_password)
                st.success("Account created -- please log in on the 'Log in' tab.")

    conn.close()
    st.stop()


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
        "SELECT product_id, product_name, min_tenure_months, max_tenure_months, "
        "max_loan_amount, processing_fee_pct FROM loan_products ORDER BY product_name"
    ).fetchall()
    conn.close()
    return [
        {
            "product_id":  r[0],
            "product_name": r[1],
            "min_tenure":  r[2],
            "max_tenure":  r[3],
            "max_amount":  r[4],
            "processing_fee_pct": r[5],
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
            apr_info  = calculate_apr(
                principal, slab["rate"], int(tenure_months), product["processing_fee_pct"]
            )
        except ValueError as e:
            st.error(str(e))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Monthly EMI", f"Rs. {breakdown['emi']:,.2f}")
            m2.metric("Total Payment", f"Rs. {breakdown['total_payment']:,.2f}")
            m3.metric("Total Interest", f"Rs. {breakdown['total_interest']:,.2f}")

            # RBI Key Fact Statement (KFS): the all-inclusive APR, not just
            # the nominal rate, must be shown alongside every quote.
            m4, m5, m6 = st.columns(3)
            m4.metric("Processing Fee", f"Rs. {apr_info['processing_fee']:,.2f}")
            m5.metric("Net Disbursed", f"Rs. {apr_info['net_disbursed']:,.2f}")
            m6.metric("APR (all-in cost)", f"{apr_info['apr']:.2f}% p.a.")

            st.caption(
                f"{product_name} · {slab_label} · {int(tenure_months)} months · "
                "Pre-qualification estimate only, subject to final approval."
            )
            st.info(
                "This is a pre-qualification estimate. Before final sign-up you will "
                "receive a Key Fact Statement (KFS) with the complete APR, all fees, "
                "and repayment schedule, as required under RBI's Digital Lending Directions."
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

def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    # SQLite has no atomic "ADD COLUMN IF NOT EXISTS", so two sessions could
    # both see the column missing and both attempt to add it right after a
    # fresh deploy; the loser of that race just hits "duplicate column",
    # which is harmless here since the goal (column exists) is already met.
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise


def _ensure_conversations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversations ("
        "  thread_id  TEXT PRIMARY KEY,"
        "  title      TEXT NOT NULL,"
        "  created_at TEXT NOT NULL"
        ")"
    )
    # Migration: conversations predates any notion of "owner". Existing rows
    # get customer_id='' and become invisible to every real account (queries
    # below always filter on the logged-in username) -- a deliberate
    # fail-closed choice, since we can't retroactively know who they belonged
    # to and showing them to the next person who logs in would repeat the
    # exact leak this migration exists to fix.
    _add_column_if_missing(conn, "conversations", "customer_id", "TEXT NOT NULL DEFAULT ''")
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
    # RBI compliance audit trail. Deliberately NOT touched by
    # _delete_conversation/_delete_all_conversations -- a regulator needs a
    # record of what the Compliance Agent flagged and how a human reviewer
    # disposed of it independent of whether the customer's own chat history
    # was later cleared from the sidebar.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS compliance_audit_log ("
        "  id                 INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  thread_id          TEXT NOT NULL,"
        "  created_at         TEXT NOT NULL,"
        "  compliance_status  TEXT NOT NULL,"
        "  compliance_reason  TEXT,"
        "  original_response  TEXT,"
        "  final_response     TEXT,"
        "  reviewer_action    TEXT NOT NULL"
        ")"
    )
    # Recorded for traceability (which customer a flagged turn belongs to)
    # but NOT used to filter the sidebar viewer -- that viewer is admin-only
    # (see _sidebar), so an admin legitimately sees every customer's entries.
    _add_column_if_missing(conn, "compliance_audit_log", "customer_id", "TEXT NOT NULL DEFAULT ''")
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


def _record_conversation_start(
    conn: sqlite3.Connection, thread_id: str, first_message: str, customer_id: str
) -> None:
    title = first_message.strip().replace("\n", " ")[:60]
    conn.execute(
        "INSERT OR IGNORE INTO conversations (thread_id, title, created_at, customer_id) "
        "VALUES (?, ?, ?, ?)",
        (thread_id, title, datetime.now(timezone.utc).isoformat(), customer_id),
    )
    conn.commit()


def _list_past_conversations(
    conn: sqlite3.Connection, exclude_thread_id: str, customer_id: str, search: str = ""
) -> list[dict]:
    search = search.strip()
    if search:
        rows = conn.execute(
            "SELECT DISTINCT c.thread_id, c.title, c.created_at FROM conversations c "
            "LEFT JOIN conversation_messages m ON m.thread_id = c.thread_id "
            "WHERE c.thread_id != ? AND c.customer_id = ? AND (c.title LIKE ? OR m.content LIKE ?) "
            "ORDER BY c.created_at DESC LIMIT 20",
            (exclude_thread_id, customer_id, f"%{search}%", f"%{search}%"),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT thread_id, title, created_at FROM conversations "
            "WHERE thread_id != ? AND customer_id = ? ORDER BY created_at DESC LIMIT 20",
            (exclude_thread_id, customer_id),
        ).fetchall()
    return [{"thread_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


def _delete_conversation(conn: sqlite3.Connection, thread_id: str, customer_id: str) -> None:
    """Remove a conversation's sidecar rows and its LangGraph checkpoint state.

    Requires customer_id and verifies ownership before deleting anything --
    defense in depth so a thread_id alone (however obtained) is never enough
    to delete someone else's conversation. Deliberately does NOT touch
    compliance_audit_log -- see its CREATE TABLE comment above.
    """
    owned = conn.execute(
        "SELECT 1 FROM conversations WHERE thread_id = ? AND customer_id = ?",
        (thread_id, customer_id),
    ).fetchone()
    if not owned:
        return
    conn.execute("DELETE FROM conversation_messages WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    conn.commit()


def _delete_all_conversations(conn: sqlite3.Connection, exclude_thread_id: str, customer_id: str) -> None:
    """Delete every one of this customer's past conversations except the
    currently active thread."""
    thread_ids = [
        r[0]
        for r in conn.execute(
            "SELECT thread_id FROM conversations WHERE thread_id != ? AND customer_id = ?",
            (exclude_thread_id, customer_id),
        ).fetchall()
    ]
    for thread_id in thread_ids:
        _delete_conversation(conn, thread_id, customer_id)


def _log_compliance_audit(
    conn: sqlite3.Connection,
    thread_id: str,
    customer_id: str,
    compliance_status: str,
    compliance_reason: str,
    original_response: str,
    final_response: str,
    reviewer_action: str,
) -> None:
    """Append one row to the RBI compliance audit trail. Every PASS is logged
    too (reviewer_action='auto') so the log is a complete record of every
    compliance-checked turn, not just the ones a human touched."""
    conn.execute(
        "INSERT INTO compliance_audit_log "
        "(thread_id, created_at, compliance_status, compliance_reason, "
        " original_response, final_response, reviewer_action, customer_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            thread_id,
            datetime.now(timezone.utc).isoformat(),
            compliance_status,
            compliance_reason,
            original_response,
            final_response,
            reviewer_action,
            customer_id,
        ),
    )
    conn.commit()


def _recent_compliance_audit(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT created_at, compliance_status, compliance_reason, reviewer_action "
        "FROM compliance_audit_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {"created_at": r[0], "compliance_status": r[1], "compliance_reason": r[2], "reviewer_action": r[3]}
        for r in rows
    ]


def _load_conversation(thread_id: str, customer_id: str) -> None:
    """Rebuild the displayed transcript (messages + route captions) for an
    existing thread from conversation_messages, and make it the active thread.

    Verifies the thread belongs to customer_id first -- defense in depth on
    top of the sidebar only ever listing the customer's own threads, so a
    thread_id alone (however it was obtained) can never load someone else's
    conversation.
    """
    owned = st.session_state.db_conn.execute(
        "SELECT 1 FROM conversations WHERE thread_id = ? AND customer_id = ?",
        (thread_id, customer_id),
    ).fetchone()
    if not owned:
        st.error("That conversation could not be found.")
        return

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


def _sidebar(user: dict) -> None:
    with st.sidebar:
        st.header("💰 QuickLoan")
        st.caption("FastFinance AI Loan Assistant")
        st.caption(f"Signed in as **{user['name']}** ({user['username']})" + (" · admin" if user["is_admin"] else ""))
        if st.button("Log out", use_container_width=True):
            for key in ["auth_user", "thread_id", "messages", "routes", "pending_hitl", "graph", "db_conn", "conv_search"]:
                st.session_state.pop(key, None)
            st.rerun()
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

        search = st.text_input(
            "Search conversations",
            key="conv_search",
            placeholder="Search by title or message…",
            label_visibility="collapsed",
        )
        past = _list_past_conversations(
            st.session_state.db_conn, st.session_state.thread_id, user["username"], search
        )

        if not past:
            st.caption("No matching conversations." if search else "No past conversations yet.")
        else:
            for conv in past:
                label = conv["title"] or "(untitled)"
                col_load, col_del = st.columns([5, 1])
                if col_load.button(f"💬 {label}", key=f"conv_{conv['thread_id']}", use_container_width=True):
                    _load_conversation(conv["thread_id"], user["username"])
                    st.rerun()
                if col_del.button("🗑️", key=f"del_{conv['thread_id']}", help="Delete this conversation"):
                    _delete_conversation(st.session_state.db_conn, conv["thread_id"], user["username"])
                    st.rerun()

            with st.popover("🗑️ Delete all conversations", use_container_width=True):
                st.warning("This permanently deletes every past conversation. This can't be undone.")
                if st.button("Confirm delete all", type="primary", use_container_width=True):
                    _delete_all_conversations(st.session_state.db_conn, st.session_state.thread_id, user["username"])
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
        st.subheader("Regulatory & Grievance Info")
        st.caption(
            f"**{GRIEVANCE_OFFICER_CONTACT}**\n\n{RBI_OMBUDSMAN_NOTE}"
        )
        # Internal QA/ops tool, not a customer-facing feature -- it spans
        # every customer's flagged responses, so only an admin account sees
        # it (see the "first registered user becomes admin" bootstrap in
        # _create_user).
        if user["is_admin"]:
            with st.expander("🔍 Compliance Audit Log (admin)"):
                audit_rows = _recent_compliance_audit(st.session_state.db_conn)
                if not audit_rows:
                    st.caption("No compliance events logged yet.")
                else:
                    for row in audit_rows:
                        icon = "✅" if row["compliance_status"] == "PASS" else "⚠️"
                        st.caption(
                            f"{icon} {row['created_at'][:19]} · {row['compliance_status']} · "
                            f"reviewer: {row['reviewer_action']}"
                            + (f" · {row['compliance_reason']}" if row["compliance_reason"] else "")
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


def _handle_hitl(user: dict) -> bool:
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
        _log_compliance_audit(
            st.session_state.db_conn, st.session_state.thread_id, user["username"],
            "REVISED", pending.get("compliance_reason", ""), pending.get("original_response", ""),
            edited, "approved_edited" if edited != pending["response"] else "approved",
        )
        del st.session_state.pending_hitl
        st.rerun()
    elif discarded:
        _log_compliance_audit(
            st.session_state.db_conn, st.session_state.thread_id, user["username"],
            "REVISED", pending.get("compliance_reason", ""), pending.get("original_response", ""),
            "", "discarded",
        )
        del st.session_state.pending_hitl
        st.rerun()

    return True


def main() -> None:
    st.set_page_config(
        page_title="QuickLoan | FastFinance",
        page_icon="💰",
        layout="wide",
    )

    user = _auth_gate()  # blocks (st.stop()) until the visitor logs in

    st.title("💰 QuickLoan | FastFinance")
    st.caption("AI-powered loan assistant — Session 15: Cloud Deployment")

    _init_session()
    _sidebar(user)

    tab_chat, tab_emi = st.tabs(["💬 Chat", "🧮 EMI Calculator"])

    with tab_chat:
        _render_history()

        hitl_active = _handle_hitl(user)

        if not hitl_active:
            prompt = st.chat_input("Ask about loan rates, eligibility, or our policies…")
            if prompt:
                if not st.session_state.messages:
                    _record_conversation_start(
                        st.session_state.db_conn, st.session_state.thread_id, prompt, user["username"]
                    )

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
                        "response":           result["response"],
                        "route_label":        route_label,
                        "compliance_reason":  result.get("compliance_reason", ""),
                        "original_response":  result.get("original_response", ""),
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
                    if result.get("compliance_status", "") == "PASS":
                        _log_compliance_audit(
                            st.session_state.db_conn, st.session_state.thread_id, user["username"],
                            "PASS", "", "", response, "auto",
                        )

    with tab_emi:
        _emi_calculator_tab()


if __name__ == "__main__":
    main()
