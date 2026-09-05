"""
s10/tests/test_s10.py
---------------------
Tests for Session 10: Multi-Agent Architecture Part 1.

Run with:
    pytest s10/tests/ -v

All tests mock the LLM and vectorstore -- no real Groq or ChromaDB calls required.

Test groups:
  TestState              -- QuickLoanState has specialist field; no compliance_status
  TestClassifyNode       -- returns RATES/POLICY/COMPLEX/OUT_OF_SCOPE; safe default
  TestPolicyAgent        -- factory returns compiled graph; RAG retrieve+respond nodes; invocable
  TestRatesAgent         -- factory returns compiled graph; respond-only node; invocable
  TestSupervisorNodes    -- call_policy_agent/call_rates_agent return correct state updates
  TestRouting            -- route_supervisor maps all 4 categories correctly
  TestSupervisorGraph    -- graph compiles; POLICY→policy; RATES→rates; COMPLEX/OOS handled
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "quickloan" or _k.startswith("quickloan."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from quickloan.state import QuickLoanState          # noqa: E402
import quickloan.nodes as _nodes                    # noqa: E402
from quickloan.nodes import (                       # noqa: E402
    call_policy_agent, call_rates_agent, classify,
    create_policy_agent, create_rates_agent,
    decline, escalate, route_supervisor,
)
from quickloan.agent import build_graph             # noqa: E402


# ---------------------------------------------------------------------------
# TestState
# ---------------------------------------------------------------------------

class TestState:
    def test_state_has_specialist_field(self):
        state: QuickLoanState = {
            "customer_message": "test",
            "response":         "",
            "history":          [],
            "query_type":       "RATES",
            "retrieved_docs":   [],
            "specialist":       "",
        }
        assert "specialist" in state

    def test_specialist_accepts_rates_agent(self):
        state: QuickLoanState = {
            "customer_message": "test", "response": "", "history": [],
            "query_type": "RATES", "retrieved_docs": [], "specialist": "rates_agent",
        }
        assert state["specialist"] == "rates_agent"

    def test_specialist_accepts_policy_agent(self):
        state: QuickLoanState = {
            "customer_message": "test", "response": "", "history": [],
            "query_type": "POLICY", "retrieved_docs": [], "specialist": "policy_agent",
        }
        assert state["specialist"] == "policy_agent"

    def test_state_has_no_compliance_status(self):
        assert "compliance_status" not in QuickLoanState.__annotations__


# ---------------------------------------------------------------------------
# TestClassifyNode
# ---------------------------------------------------------------------------

class TestClassifyNode:
    def _state(self, message: str) -> QuickLoanState:
        return {
            "customer_message": message, "response": "", "history": [],
            "query_type": "", "retrieved_docs": [], "specialist": "",
        }

    def test_rates_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="RATES")
            result = classify(self._state("What is the home loan rate?"))
        assert result["query_type"] == "RATES"

    def test_policy_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="POLICY")
            result = classify(self._state("What documents do I need for a home loan?"))
        assert result["query_type"] == "POLICY"

    def test_complex_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="COMPLEX")
            result = classify(self._state("Which loan is best for me?"))
        assert result["query_type"] == "COMPLEX"

    def test_oos_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            result = classify(self._state("What is the weather today?"))
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_invalid_response_defaults_to_rates(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="BANANA")
            result = classify(self._state("test"))
        assert result["query_type"] == "RATES"

    def test_classify_error_defaults_to_rates(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.side_effect = Exception("API error")
            result = classify(self._state("test"))
        assert result["query_type"] == "RATES"

    def test_classify_strips_whitespace(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="  POLICY  ")
            result = classify(self._state("test"))
        assert result["query_type"] == "POLICY"


# ---------------------------------------------------------------------------
# TestPolicyAgent
# ---------------------------------------------------------------------------

class TestPolicyAgent:
    def test_factory_returns_compiled_graph(self):
        agent = create_policy_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        a1 = create_policy_agent()
        a2 = create_policy_agent()
        assert a1 is not a2

    def test_agent_has_retrieve_docs_node(self):
        agent = create_policy_agent()
        assert "retrieve_docs" in agent.get_graph().nodes

    def test_agent_has_respond_node(self):
        agent = create_policy_agent()
        assert "respond" in agent.get_graph().nodes

    def test_agent_is_invocable(self):
        agent = create_policy_agent()
        with patch.object(_nodes, "vectorstore", None), \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Policy answer.")
            result = agent.invoke({
                "customer_message": "What documents do I need?",
                "history": [], "response": "",
                "query_type": "POLICY", "retrieved_docs": [], "specialist": "",
            })
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_agent_returns_retrieved_docs(self):
        agent = create_policy_agent()
        mock_doc = MagicMock()
        mock_doc.page_content = "FastFinance policy text"
        mock_doc.metadata = {"source": "fastfinance_policy.md"}
        with patch.object(_nodes, "vectorstore") as mock_vs, \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "llm") as mock_llm:
            mock_vs.similarity_search.return_value = [mock_doc]
            mock_llm.invoke.return_value = MagicMock(content="Policy answer.")
            result = agent.invoke({
                "customer_message": "What is the maximum loan amount?",
                "history": [], "response": "",
                "query_type": "POLICY", "retrieved_docs": [], "specialist": "",
            })
        assert len(result.get("retrieved_docs", [])) > 0

    def test_agent_updates_history(self):
        agent = create_policy_agent()
        with patch.object(_nodes, "vectorstore", None), \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Answer.")
            result = agent.invoke({
                "customer_message": "test", "history": [],
                "response": "", "query_type": "POLICY",
                "retrieved_docs": [], "specialist": "",
            })
        assert len(result.get("history", [])) == 2


# ---------------------------------------------------------------------------
# TestRatesAgent
# ---------------------------------------------------------------------------

class TestRatesAgent:
    def test_factory_returns_compiled_graph(self):
        agent = create_rates_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        a1 = create_rates_agent()
        a2 = create_rates_agent()
        assert a1 is not a2

    def test_agent_has_respond_node(self):
        agent = create_rates_agent()
        assert "respond" in agent.get_graph().nodes

    def test_agent_does_not_have_retrieve_docs_node(self):
        agent = create_rates_agent()
        assert "retrieve_docs" not in agent.get_graph().nodes

    def test_agent_is_invocable(self):
        agent = create_rates_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Rate answer.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "What is the home loan rate?",
                "history": [], "response": "",
                "query_type": "RATES", "retrieved_docs": [], "specialist": "",
            })
        assert "response" in result

    def test_agent_calls_tools(self):
        agent = create_rates_agent()
        tool_call = {"name": "query_rates", "args": {"product_id": "home_loan"}, "id": "tc1"}
        with patch.object(_nodes, "llm_with_tools") as mock_llm_tools, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Home Loan: 8.75% p.a."):
            mock_llm_tools.invoke.return_value = MagicMock(
                content="", tool_calls=[tool_call]
            )
            mock_llm.invoke.return_value = MagicMock(content="Home loan rate is 8.75% p.a.")
            result = agent.invoke({
                "customer_message": "What is the home loan rate?",
                "history": [], "response": "",
                "query_type": "RATES", "retrieved_docs": [], "specialist": "",
            })
        assert "response" in result

    def test_agent_updates_history(self):
        agent = create_rates_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Answer.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "test", "history": [],
                "response": "", "query_type": "RATES",
                "retrieved_docs": [], "specialist": "",
            })
        assert len(result.get("history", [])) == 2


# ---------------------------------------------------------------------------
# TestSupervisorNodes
# ---------------------------------------------------------------------------

class TestSupervisorNodes:
    def _state(self, message: str = "test", qt: str = "POLICY") -> QuickLoanState:
        return {
            "customer_message": message, "response": "", "history": [],
            "query_type": qt, "retrieved_docs": [], "specialist": "",
        }

    def test_call_policy_agent_sets_specialist(self):
        with patch.object(_nodes, "_policy_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Policy info.", "history": [], "retrieved_docs": []
            }
            result = call_policy_agent(self._state())
        assert result["specialist"] == "policy_agent"

    def test_call_policy_agent_returns_response(self):
        with patch.object(_nodes, "_policy_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Policy info.", "history": [], "retrieved_docs": []
            }
            result = call_policy_agent(self._state())
        assert result["response"] == "Policy info."

    def test_call_policy_agent_returns_retrieved_docs(self):
        with patch.object(_nodes, "_policy_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "info.", "history": [],
                "retrieved_docs": ["[policy.md]\nchunk text"]
            }
            result = call_policy_agent(self._state())
        assert len(result["retrieved_docs"]) == 1

    def test_call_rates_agent_sets_specialist(self):
        with patch.object(_nodes, "_rates_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Rate info.", "history": []}
            result = call_rates_agent(self._state(qt="RATES"))
        assert result["specialist"] == "rates_agent"

    def test_call_rates_agent_returns_response(self):
        with patch.object(_nodes, "_rates_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Rate info.", "history": []}
            result = call_rates_agent(self._state(qt="RATES"))
        assert result["response"] == "Rate info."

    def test_escalate_sets_specialist(self):
        result = escalate(self._state())
        assert result["specialist"] == "escalated"

    def test_decline_sets_specialist(self):
        result = decline(self._state())
        assert result["specialist"] == "declined"


# ---------------------------------------------------------------------------
# TestRouting
# ---------------------------------------------------------------------------

class TestRouting:
    def _state(self, qt: str) -> QuickLoanState:
        return {
            "customer_message": "test", "response": "", "history": [],
            "query_type": qt, "retrieved_docs": [], "specialist": "",
        }

    def test_policy_routes_to_policy_agent(self):
        assert route_supervisor(self._state("POLICY")) == "call_policy_agent"

    def test_rates_routes_to_rates_agent(self):
        assert route_supervisor(self._state("RATES")) == "call_rates_agent"

    def test_complex_routes_to_escalate(self):
        assert route_supervisor(self._state("COMPLEX")) == "escalate"

    def test_oos_routes_to_decline(self):
        assert route_supervisor(self._state("OUT_OF_SCOPE")) == "decline"

    def test_unknown_defaults_to_rates_agent(self):
        assert route_supervisor(self._state("UNKNOWN")) == "call_rates_agent"


# ---------------------------------------------------------------------------
# TestSupervisorGraph
# ---------------------------------------------------------------------------

class TestSupervisorGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_has_classify_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "classify" in graph.get_graph().nodes

    def test_graph_has_call_policy_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_policy_agent" in graph.get_graph().nodes

    def test_graph_has_call_rates_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_rates_agent" in graph.get_graph().nodes

    def test_policy_query_routes_to_policy_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_policy_agent") as mock_policy_agent:
            mock_clf.invoke.return_value = MagicMock(content="POLICY")
            mock_policy_agent.invoke.return_value = {
                "response": "Policy answer.", "history": [], "retrieved_docs": []
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What documents do I need?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-policy"}},
            )
        assert result["specialist"] == "policy_agent"

    def test_rates_query_routes_to_rates_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_rates_agent") as mock_rates_agent:
            mock_clf.invoke.return_value = MagicMock(content="RATES")
            mock_rates_agent.invoke.return_value = {
                "response": "Rate answer.", "history": []
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What is the home loan rate?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-rates"}},
            )
        assert result["specialist"] == "rates_agent"

    def test_complex_query_escalates(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Which loan is best for me?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-complex"}},
            )
        assert result["specialist"] == "escalated"
        assert "loan officer" in result["response"]

    def test_oos_query_declines(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What is the weather?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-oos"}},
            )
        assert result["specialist"] == "declined"

    def test_graph_result_has_specialist_field(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "test", "response": "",
                 "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-field"}},
            )
        assert "specialist" in result
