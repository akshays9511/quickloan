"""
s13/tests/test_s13.py
---------------------
Tests for Session 13: Streamlit UI (QuickLoan).

Run with:
    pytest s13/tests/ -v

All tests are pure Python — no Streamlit context needed.
The app helper functions are imported directly from app.py.

Test groups:
  TestBuildInputState    -- build_input_state() returns correct graph input dict
  TestGetThreadConfig    -- get_thread_config() returns correct LangGraph config
  TestComplianceBadge    -- compliance_badge() returns correct display text (RBI flavour)
  TestIsEscalated        -- is_escalated() detects escalated specialist
  TestFormatRouteLabel   -- format_route_label() formats routing info correctly
  TestAgentGraph         -- build_graph() compiles; S12 nodes are present
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "quickloan" or _k.startswith("quickloan."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

_spec = importlib.util.spec_from_file_location("app", SOLUTION_DIR / "app.py")
_app  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app)

build_input_state  = _app.build_input_state
get_thread_config  = _app.get_thread_config
compliance_badge   = _app.compliance_badge
is_escalated       = _app.is_escalated
format_route_label = _app.format_route_label

from quickloan.agent import build_graph  # noqa: E402
import quickloan.nodes as _nodes         # noqa: E402


# ---------------------------------------------------------------------------
# TestBuildInputState
# ---------------------------------------------------------------------------

class TestBuildInputState:
    def test_has_customer_message(self):
        state = build_input_state("What is the home loan rate?")
        assert state["customer_message"] == "What is the home loan rate?"

    def test_has_empty_response(self):
        assert build_input_state("test")["response"] == ""

    def test_has_empty_specialist(self):
        assert build_input_state("test")["specialist"] == ""

    def test_has_empty_retrieved_docs(self):
        assert build_input_state("test")["retrieved_docs"] == []

    def test_has_empty_compliance_status(self):
        assert build_input_state("test")["compliance_status"] == ""

    def test_all_required_keys_present(self):
        state    = build_input_state("test")
        required = {"customer_message", "response", "specialist", "retrieved_docs", "compliance_status"}
        assert required.issubset(set(state.keys()))

    def test_different_messages_differ(self):
        s1 = build_input_state("rates question")
        s2 = build_input_state("policy question")
        assert s1["customer_message"] != s2["customer_message"]

    def test_empty_message_accepted(self):
        assert build_input_state("")["customer_message"] == ""


# ---------------------------------------------------------------------------
# TestGetThreadConfig
# ---------------------------------------------------------------------------

class TestGetThreadConfig:
    def test_returns_dict(self):
        assert isinstance(get_thread_config("abc"), dict)

    def test_has_configurable_key(self):
        assert "configurable" in get_thread_config("abc")

    def test_configurable_has_thread_id(self):
        assert get_thread_config("loan-session")["configurable"]["thread_id"] == "loan-session"

    def test_different_ids_differ(self):
        c1 = get_thread_config("s1")
        c2 = get_thread_config("s2")
        assert c1["configurable"]["thread_id"] != c2["configurable"]["thread_id"]


# ---------------------------------------------------------------------------
# TestComplianceBadge
# ---------------------------------------------------------------------------

class TestComplianceBadge:
    def test_pass_returns_checkmark(self):
        assert "✅" in compliance_badge("PASS")

    def test_revised_returns_warning(self):
        assert "⚠️" in compliance_badge("REVISED")

    def test_fail_returns_cross(self):
        assert "❌" in compliance_badge("FAIL: banned phrase")

    def test_empty_returns_empty_string(self):
        assert compliance_badge("") == ""

    def test_unknown_returns_empty_string(self):
        assert compliance_badge("UNKNOWN") == ""

    def test_pass_mentions_rbi(self):
        badge = compliance_badge("PASS")
        assert "RBI" in badge or "Compliant" in badge

    def test_revised_text(self):
        assert "Revised" in compliance_badge("REVISED")


# ---------------------------------------------------------------------------
# TestIsEscalated
# ---------------------------------------------------------------------------

class TestIsEscalated:
    def test_escalated_returns_true(self):
        assert is_escalated({"specialist": "escalated"}) is True

    def test_rates_agent_returns_false(self):
        assert is_escalated({"specialist": "rates_agent"}) is False

    def test_policy_agent_returns_false(self):
        assert is_escalated({"specialist": "policy_agent"}) is False

    def test_empty_specialist_returns_false(self):
        assert is_escalated({"specialist": ""}) is False

    def test_missing_specialist_returns_false(self):
        assert is_escalated({}) is False

    def test_declined_is_not_escalated(self):
        assert is_escalated({"specialist": "declined"}) is False


# ---------------------------------------------------------------------------
# TestFormatRouteLabel
# ---------------------------------------------------------------------------

class TestFormatRouteLabel:
    def test_includes_query_type(self):
        result = {"query_type": "RATES", "specialist": "rates_agent", "compliance_status": "PASS"}
        assert "RATES" in format_route_label(result)

    def test_includes_specialist(self):
        result = {"query_type": "POLICY", "specialist": "policy_agent", "compliance_status": "PASS"}
        assert "policy_agent" in format_route_label(result)

    def test_includes_badge_for_pass(self):
        result = {"query_type": "RATES", "specialist": "rates_agent", "compliance_status": "PASS"}
        assert "✅" in format_route_label(result)

    def test_no_badge_for_empty_status(self):
        result = {"query_type": "COMPLEX", "specialist": "escalated", "compliance_status": ""}
        label  = format_route_label(result)
        assert "✅" not in label and "⚠️" not in label

    def test_dash_for_missing_keys(self):
        assert "—" in format_route_label({})

    def test_revised_badge_shown(self):
        result = {"query_type": "RATES", "specialist": "rates_agent", "compliance_status": "REVISED"}
        assert "⚠️" in format_route_label(result)


# ---------------------------------------------------------------------------
# TestAgentGraph
# ---------------------------------------------------------------------------

class TestAgentGraph:
    def test_build_graph_compiles(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert build_graph(checkpointer=MemorySaver()) is not None

    def test_graph_has_classify_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "classify" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_has_compliance_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "call_compliance_agent" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_has_rates_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "call_rates_agent" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_has_policy_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "call_policy_agent" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_invocable(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_rates_agent") as mock_ra, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="RATES")
            mock_ra.invoke.return_value  = {"response": "Home loan rate is 8.5% p.a.", "history": []}
            mock_ca.invoke.return_value  = {"response": "Home loan rate is 8.5% p.a.", "compliance_status": "PASS"}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                build_input_state("What is the home loan rate?"),
                config=get_thread_config("test-s13-quickloan"),
            )
        assert "response" in result
        assert "compliance_status" in result

    def test_compliance_status_in_result(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_rates_agent") as mock_ra, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="RATES")
            mock_ra.invoke.return_value  = {"response": "Guaranteed 5% rate!", "history": []}
            mock_ca.invoke.return_value  = {
                "response":          "Current home loan rate is 8.5% p.a.",
                "compliance_status": "REVISED",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                build_input_state("What rate will I get?"),
                config=get_thread_config("test-s13-revised"),
            )
        assert result["compliance_status"] == "REVISED"
