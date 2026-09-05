"""
s12/tests/test_s12.py
---------------------
Tests for Session 12: Multi-Agent Architecture Part 2 (Compliance Agent).

Run with:
    pytest s12/tests/ -v

All tests mock the LLM, vectorstore, and SQLite DB -- no real Groq, ChromaDB, or DB calls required.

Test groups:
  TestState                -- QuickLoanState has both specialist and compliance_status fields
  TestComplianceHelpers    -- _load_valid_rates, _extract_rates, _check_compliance_logic
  TestCheckRbiNode         -- check_rbi sets PASS/FAIL in compliance_status
  TestReviseResponseNode   -- revise_response rewrites draft; falls back to SAFE_COMPLIANCE_RESPONSE
  TestRouteCompliance      -- route_compliance returns "revise" on FAIL, END on PASS
  TestComplianceAgentFactory -- create_compliance_agent() builds correct sub-graph
  TestCallComplianceAgentNode -- call_compliance_agent passes state; returns response+status
  TestSupervisorGraph      -- POLICY/RATES go through compliance; COMPLEX/OOS bypass it
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "quickloan" or _k.startswith("quickloan."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from quickloan.state import QuickLoanState          # noqa: E402
import quickloan.nodes as _nodes                    # noqa: E402
from quickloan.nodes import (                       # noqa: E402
    call_compliance_agent,
    call_policy_agent,
    call_rates_agent,
    check_rbi,
    classify,
    create_compliance_agent,
    create_policy_agent,
    create_rates_agent,
    decline,
    escalate,
    revise_response,
    route_compliance,
    route_supervisor,
)
from quickloan.agent import build_graph             # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    message: str = "test",
    response: str = "",
    history: list | None = None,
    query_type: str = "RATES",
    retrieved_docs: list | None = None,
    specialist: str = "",
    compliance_status: str = "",
) -> QuickLoanState:
    return {
        "customer_message":  message,
        "response":          response,
        "history":           history or [],
        "query_type":        query_type,
        "retrieved_docs":    retrieved_docs or [],
        "specialist":        specialist,
        "compliance_status": compliance_status,
    }


# ---------------------------------------------------------------------------
# TestState
# ---------------------------------------------------------------------------

class TestState:
    def test_state_has_specialist_field(self):
        state = _make_state()
        assert "specialist" in state

    def test_state_has_compliance_status_field(self):
        state = _make_state()
        assert "compliance_status" in state

    def test_compliance_status_accepts_pass(self):
        state = _make_state(compliance_status="PASS")
        assert state["compliance_status"] == "PASS"

    def test_compliance_status_accepts_fail(self):
        state = _make_state(compliance_status="FAIL: banned phrase: 'guaranteed approval'")
        assert state["compliance_status"].startswith("FAIL")

    def test_compliance_status_accepts_revised(self):
        state = _make_state(compliance_status="REVISED")
        assert state["compliance_status"] == "REVISED"


# ---------------------------------------------------------------------------
# TestComplianceHelpers
# ---------------------------------------------------------------------------

class TestComplianceHelpers:
    def test_load_valid_rates_returns_set(self):
        with patch("sqlite3.connect") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [(8.5,), (10.0,), (12.75,)]
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_conn.return_value)
            mock_conn.return_value.execute.return_value = mock_cursor
            result = _nodes._load_valid_rates()
        assert isinstance(result, set)

    def test_load_valid_rates_returns_empty_on_error(self):
        with patch("sqlite3.connect", side_effect=Exception("DB not found")):
            result = _nodes._load_valid_rates()
        assert result == set()

    def test_extract_rates_finds_single_rate(self):
        result = _nodes._extract_rates("The home loan rate is 8.5% p.a. for salaried applicants.")
        assert 8.5 in result

    def test_extract_rates_finds_multiple_rates(self):
        result = _nodes._extract_rates("Rates range from 8.5% p.a. to 12.75% p.a.")
        assert len(result) == 2
        assert 8.5 in result
        assert 12.75 in result

    def test_extract_rates_empty_on_no_rates(self):
        result = _nodes._extract_rates("Please contact our loan officer for details.")
        assert result == []

    def test_extract_rates_case_insensitive(self):
        result = _nodes._extract_rates("Rate: 9.25% P.A.")
        assert 9.25 in result

    def test_check_compliance_logic_pass_on_clean_response(self):
        passed, reason = _nodes._check_compliance_logic(
            "Our home loan interest rate is subject to eligibility criteria. "
            "Please visit a branch for personalised assessment."
        )
        assert passed is True
        assert reason == "PASS"

    def test_check_compliance_logic_fail_on_banned_phrase(self):
        passed, reason = _nodes._check_compliance_logic(
            "Congratulations! Your loan is guaranteed approval pending document verification."
        )
        assert passed is False
        assert "banned phrase" in reason

    def test_check_compliance_logic_fail_on_hallucinated_rate(self):
        with patch.object(_nodes, "_load_valid_rates", return_value={8.5, 10.0}):
            passed, reason = _nodes._check_compliance_logic(
                "Our personal loan rate is 99.99% p.a."
            )
        assert passed is False
        assert "hallucinated rate" in reason

    def test_check_compliance_logic_pass_on_valid_rate(self):
        with patch.object(_nodes, "_load_valid_rates", return_value={8.5, 10.0}):
            passed, reason = _nodes._check_compliance_logic(
                "Our home loan rate starts at 8.5% p.a. for eligible applicants."
            )
        assert passed is True

    def test_check_compliance_logic_pass_when_no_valid_rates_in_db(self):
        with patch.object(_nodes, "_load_valid_rates", return_value=set()):
            passed, reason = _nodes._check_compliance_logic(
                "Our rate is 99.99% p.a."
            )
        assert passed is True

    def test_check_compliance_logic_banned_phrase_case_insensitive(self):
        passed, reason = _nodes._check_compliance_logic(
            "Your application is PRE-APPROVED subject to verification."
        )
        assert passed is False


# ---------------------------------------------------------------------------
# TestCheckRbiNode
# ---------------------------------------------------------------------------

class TestCheckRbiNode:
    def test_check_rbi_sets_pass(self):
        state = _make_state(response="Our loan requires income verification.")
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")):
            result = check_rbi(state)
        assert result["compliance_status"] == "PASS"

    def test_check_rbi_sets_fail(self):
        state = _make_state(response="Your loan is guaranteed approval!")
        with patch.object(_nodes, "_check_compliance_logic",
                          return_value=(False, "banned phrase: 'guaranteed approval'")):
            result = check_rbi(state)
        assert result["compliance_status"].startswith("FAIL")
        assert "guaranteed approval" in result["compliance_status"]

    def test_check_rbi_fail_includes_reason(self):
        state = _make_state(response="The rate is 99.99% p.a.")
        with patch.object(_nodes, "_check_compliance_logic",
                          return_value=(False, "hallucinated rate: 99.99% p.a.")):
            result = check_rbi(state)
        assert "hallucinated rate" in result["compliance_status"]

    def test_check_rbi_passes_draft_to_logic(self):
        state = _make_state(response="specific draft text")
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")) as mock:
            check_rbi(state)
        mock.assert_called_once_with("specific draft text")


# ---------------------------------------------------------------------------
# TestReviseResponseNode
# ---------------------------------------------------------------------------

class TestReviseResponseNode:
    def test_revise_response_returns_llm_output(self):
        state = _make_state(
            response="Your loan is guaranteed approval!",
            compliance_status="FAIL: banned phrase: 'guaranteed approval'",
        )
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Compliant rewrite here.")
            result = revise_response(state)
        assert result["response"] == "Compliant rewrite here."

    def test_revise_response_sets_revised_status(self):
        state = _make_state(
            response="Your loan is pre-approved.",
            compliance_status="FAIL: banned phrase: 'pre-approved'",
        )
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Revised response.")
            result = revise_response(state)
        assert result["compliance_status"] == "REVISED"

    def test_revise_response_falls_back_on_llm_error(self):
        from quickloan.config import SAFE_COMPLIANCE_RESPONSE
        state = _make_state(
            response="Your loan is guaranteed!",
            compliance_status="FAIL: banned phrase: 'guaranteed approval'",
        )
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.side_effect = Exception("LLM unavailable")
            result = revise_response(state)
        assert result["response"] == SAFE_COMPLIANCE_RESPONSE

    def test_revise_response_falls_back_on_empty_content(self):
        from quickloan.config import SAFE_COMPLIANCE_RESPONSE
        state = _make_state(
            response="draft",
            compliance_status="FAIL: banned phrase: 'no credit check'",
        )
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="   ")
            result = revise_response(state)
        assert result["response"] == SAFE_COMPLIANCE_RESPONSE


# ---------------------------------------------------------------------------
# TestRouteCompliance
# ---------------------------------------------------------------------------

class TestRouteCompliance:
    def test_fail_routes_to_revise(self):
        state = _make_state(compliance_status="FAIL: banned phrase: 'pre-approved'")
        assert route_compliance(state) == "revise"

    def test_pass_routes_to_end(self):
        state = _make_state(compliance_status="PASS")
        assert route_compliance(state) == END

    def test_revised_routes_to_end(self):
        state = _make_state(compliance_status="REVISED")
        assert route_compliance(state) == END

    def test_empty_status_routes_to_end(self):
        state = _make_state(compliance_status="")
        assert route_compliance(state) == END


# ---------------------------------------------------------------------------
# TestComplianceAgentFactory
# ---------------------------------------------------------------------------

class TestComplianceAgentFactory:
    def test_factory_returns_compiled_graph(self):
        agent = create_compliance_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        a1 = create_compliance_agent()
        a2 = create_compliance_agent()
        assert a1 is not a2

    def test_agent_has_check_rbi_node(self):
        agent = create_compliance_agent()
        assert "check_rbi" in agent.get_graph().nodes

    def test_agent_has_revise_node(self):
        agent = create_compliance_agent()
        assert "revise" in agent.get_graph().nodes

    def test_agent_passes_clean_response(self):
        agent = create_compliance_agent()
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")):
            result = agent.invoke(_make_state(response="Clean compliant response."))
        assert result["compliance_status"] == "PASS"
        assert result["response"] == "Clean compliant response."

    def test_agent_revises_failing_response(self):
        agent = create_compliance_agent()
        with patch.object(_nodes, "_check_compliance_logic",
                          return_value=(False, "banned phrase: 'guaranteed approval'")), \
             patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Revised compliant text.")
            result = agent.invoke(_make_state(
                response="Your loan is guaranteed approval!",
                compliance_status="",
            ))
        assert result["compliance_status"] == "REVISED"
        assert result["response"] == "Revised compliant text."


# ---------------------------------------------------------------------------
# TestCallComplianceAgentNode
# ---------------------------------------------------------------------------

class TestCallComplianceAgentNode:
    def test_call_compliance_agent_returns_response(self):
        state = _make_state(response="Clean policy answer.", specialist="policy_agent")
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Clean policy answer.",
                "compliance_status": "PASS",
            }
            result = call_compliance_agent(state)
        assert result["response"] == "Clean policy answer."

    def test_call_compliance_agent_returns_compliance_status(self):
        state = _make_state(response="Revised response.", specialist="rates_agent")
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Revised response.",
                "compliance_status": "REVISED",
            }
            result = call_compliance_agent(state)
        assert result["compliance_status"] == "REVISED"

    def test_call_compliance_agent_passes_compliance_status_empty(self):
        state = _make_state(response="Some response.", specialist="policy_agent")
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "r", "compliance_status": "PASS"}
            call_compliance_agent(state)
            invoke_kwargs = mock_agent.invoke.call_args[0][0]
        assert invoke_kwargs.get("compliance_status") == ""

    def test_call_compliance_agent_passes_response(self):
        state = _make_state(response="Original draft.", specialist="rates_agent")
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "r", "compliance_status": "PASS"}
            call_compliance_agent(state)
            invoke_kwargs = mock_agent.invoke.call_args[0][0]
        assert invoke_kwargs.get("response") == "Original draft."


# ---------------------------------------------------------------------------
# TestSupervisorGraph
# ---------------------------------------------------------------------------

class TestSupervisorGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_has_compliance_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_compliance_agent" in graph.get_graph().nodes

    def test_graph_has_classify_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "classify" in graph.get_graph().nodes

    def test_policy_goes_through_compliance_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_policy_agent") as mock_policy, \
             patch.object(_nodes, "_compliance_agent") as mock_comp:
            mock_clf.invoke.return_value   = MagicMock(content="POLICY")
            mock_policy.invoke.return_value = {
                "response": "Policy answer.", "history": [], "retrieved_docs": []
            }
            mock_comp.invoke.return_value = {
                "response": "Policy answer.", "compliance_status": "PASS"
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _make_state(message="What documents do I need?"),
                config={"configurable": {"thread_id": "test-policy-compliance"}},
            )
        mock_comp.invoke.assert_called_once()
        assert result["specialist"] == "policy_agent"

    def test_rates_goes_through_compliance_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_rates_agent") as mock_rates, \
             patch.object(_nodes, "_compliance_agent") as mock_comp:
            mock_clf.invoke.return_value  = MagicMock(content="RATES")
            mock_rates.invoke.return_value = {"response": "Rate answer.", "history": []}
            mock_comp.invoke.return_value  = {
                "response": "Rate answer.", "compliance_status": "PASS"
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _make_state(message="What is the home loan rate?"),
                config={"configurable": {"thread_id": "test-rates-compliance"}},
            )
        mock_comp.invoke.assert_called_once()
        assert result["specialist"] == "rates_agent"

    def test_complex_bypasses_compliance_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_compliance_agent") as mock_comp:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _make_state(message="Which loan is best for me?"),
                config={"configurable": {"thread_id": "test-complex-bypass"}},
            )
        mock_comp.invoke.assert_not_called()
        assert result["specialist"] == "escalated"

    def test_oos_bypasses_compliance_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_compliance_agent") as mock_comp:
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _make_state(message="What is the weather today?"),
                config={"configurable": {"thread_id": "test-oos-bypass"}},
            )
        mock_comp.invoke.assert_not_called()
        assert result["specialist"] == "declined"

    def test_compliance_status_in_final_result(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_policy_agent") as mock_policy, \
             patch.object(_nodes, "_compliance_agent") as mock_comp:
            mock_clf.invoke.return_value   = MagicMock(content="POLICY")
            mock_policy.invoke.return_value = {
                "response": "Clean answer.", "history": [], "retrieved_docs": []
            }
            mock_comp.invoke.return_value  = {
                "response": "Clean answer.", "compliance_status": "PASS"
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _make_state(message="How do I apply?"),
                config={"configurable": {"thread_id": "test-status-field"}},
            )
        assert "compliance_status" in result
        assert result["compliance_status"] == "PASS"

    def test_graph_compiles_without_checkpointer(self):
        graph = build_graph()
        assert graph is not None
