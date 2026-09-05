"""
s05/tests/test_s05.py
---------------------
Tests for Session 5: SQLite tool calls.

Run with:
    pytest s05/tests/ -v

Test groups:
  TestQuickLoanState       -- state TypedDict has all five fields (unchanged from S04)
  TestQueryRatesTool       -- query_rates() SQL correctness, filtering, output format
  TestQueryEligibilityTool -- query_eligibility() SQL correctness, filtering
  TestToolSQLSafety        -- SQL injection protection and parameterised-query enforcement
  TestToolsBinding         -- llm_with_tools exists; tools are @tool decorated; prompt updated
  TestRunToolDispatch      -- _run_tool dispatches correctly; handles unknown names
  TestRespondWithTools     -- respond() calls llm_with_tools; executes tool calls; calls llm again
  TestGraphRouting         -- SIMPLE goes through retrieve_docs->respond; COMPLEX/OOS skip tools
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

import quickloan  # noqa: E402
import quickloan.nodes as _nodes  # noqa: E402
import quickloan.tools as _tools  # noqa: E402
from quickloan.config import SYSTEM_PROMPT  # noqa: E402
from quickloan.state import QuickLoanState  # noqa: E402
from quickloan.tools import _run_tool, query_rates, query_eligibility  # noqa: E402
from quickloan.nodes import respond  # noqa: E402
from quickloan.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# TestQuickLoanState
# ---------------------------------------------------------------------------

class TestQuickLoanState:
    def test_state_has_customer_message_field(self):
        assert "customer_message" in QuickLoanState.__annotations__

    def test_state_has_response_field(self):
        assert "response" in QuickLoanState.__annotations__

    def test_state_has_history_field(self):
        assert "history" in QuickLoanState.__annotations__

    def test_state_has_query_type_field(self):
        assert "query_type" in QuickLoanState.__annotations__

    def test_state_has_retrieved_docs_field(self):
        assert "retrieved_docs" in QuickLoanState.__annotations__

    def test_state_has_exactly_five_fields(self):
        assert len(QuickLoanState.__annotations__) == 5


# ---------------------------------------------------------------------------
# TestQueryRatesTool
# ---------------------------------------------------------------------------

class TestQueryRatesTool:
    def test_query_rates_all_returns_home_loan(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_id": "all"})
        assert "Home Loan" in result

    def test_query_rates_all_returns_personal_loan(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_id": "all"})
        assert "Personal Loan" in result

    def test_query_rates_personal_loan_filter_includes_rate(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_id": "personal_loan"})
        assert "11.5" in result

    def test_query_rates_personal_loan_includes_cibil_range(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_id": "personal_loan"})
        assert "CIBIL" in result
        assert "750" in result

    def test_query_rates_gold_loan_flat_rate(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_id": "gold_loan"})
        assert "10.5" in result

    def test_query_rates_home_loan_rate(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_id": "home_loan"})
        assert "8.75" in result

    def test_query_rates_no_match_returns_message(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_id": "nonexistent_product"})
        assert "No rate data found" in result or "nonexistent_product" in result

    def test_query_rates_default_is_all(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({})
        assert "Home Loan" in result
        assert "Personal Loan" in result

    def test_query_rates_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        assert isinstance(query_rates.invoke({"product_id": "all"}), str)

    def test_query_rates_filter_excludes_other_products(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_id": "gold_loan"})
        assert "Personal Loan" not in result


# ---------------------------------------------------------------------------
# TestQueryEligibilityTool
# ---------------------------------------------------------------------------

class TestQueryEligibilityTool:
    def test_query_eligibility_all_returns_multiple(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_eligibility.invoke({"product_id": "all"})
        assert "Personal Loan" in result
        assert "Home Loan" in result

    def test_query_eligibility_filter_by_product(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_eligibility.invoke({"product_id": "personal_loan"})
        assert "Personal Loan" in result

    def test_query_eligibility_filter_excludes_others(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_eligibility.invoke({"product_id": "personal_loan"})
        assert "Home Loan" not in result

    def test_query_eligibility_includes_min_cibil(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_eligibility.invoke({"product_id": "all"})
        assert "700" in result or "720" in result

    def test_query_eligibility_includes_income(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_eligibility.invoke({"product_id": "all"})
        assert "25000" in result or "40000" in result

    def test_query_eligibility_no_match_returns_message(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_eligibility.invoke({"product_id": "nonexistent_xyz"})
        assert "No eligibility data found" in result or "nonexistent_xyz" in result

    def test_query_eligibility_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        assert isinstance(query_eligibility.invoke({"product_id": "all"}), str)


# ---------------------------------------------------------------------------
# TestToolSQLSafety
# ---------------------------------------------------------------------------

class TestToolSQLSafety:
    def test_query_eligibility_sql_injection_safe(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_eligibility.invoke({"product_id": "'; DROP TABLE eligibility_rules; --"})
        assert isinstance(result, str)
        normal = query_eligibility.invoke({"product_id": "personal_loan"})
        assert "Personal Loan" in normal

    def test_query_eligibility_uses_question_mark_placeholder(self):
        import inspect
        source = inspect.getsource(query_eligibility.func)
        assert "= ?" in source, (
            "query_eligibility must use a ? placeholder for the product_id parameter. "
            "Never interpolate user input directly into a SQL string."
        )


# ---------------------------------------------------------------------------
# TestToolsBinding
# ---------------------------------------------------------------------------

class TestToolsBinding:
    def test_llm_with_tools_exists(self):
        assert hasattr(_tools, "llm_with_tools"), (
            "llm_with_tools not found. Create it with llm.bind_tools([query_rates, query_eligibility])."
        )

    def test_query_rates_is_tool_decorated(self):
        assert hasattr(query_rates, "name"), (
            "query_rates does not appear to be decorated with @tool."
        )

    def test_query_eligibility_is_tool_decorated(self):
        assert hasattr(query_eligibility, "name"), (
            "query_eligibility does not appear to be decorated with @tool."
        )

    def test_query_rates_tool_name(self):
        assert query_rates.name == "query_rates"

    def test_query_eligibility_tool_name(self):
        assert query_eligibility.name == "query_eligibility"

    def test_system_prompt_has_no_hardcoded_rates(self):
        assert "10.5%" not in SYSTEM_PROMPT and "8.75%" not in SYSTEM_PROMPT, (
            "Session 5 removes hardcoded rates from SYSTEM_PROMPT. "
            "Rates now come from query_rates(). Remove the rate table."
        )

    def test_system_prompt_mentions_tools_or_database(self):
        assert "tool" in SYSTEM_PROMPT.lower() or "database" in SYSTEM_PROMPT.lower(), (
            "SYSTEM_PROMPT should instruct the LLM to use database tools for rates."
        )


# ---------------------------------------------------------------------------
# TestRunToolDispatch
# ---------------------------------------------------------------------------

class TestRunToolDispatch:
    def test_run_tool_dispatches_query_rates(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_rates", {"product_id": "personal_loan"})
        assert "11.5" in result

    def test_run_tool_dispatches_query_eligibility(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_eligibility", {"product_id": "all"})
        assert "Personal Loan" in result

    def test_run_tool_unknown_name_returns_error_string(self):
        result = _run_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    def test_run_tool_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_rates", {"product_id": "home_loan"})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestRespondWithTools
# ---------------------------------------------------------------------------

class TestRespondWithTools:
    def _make_tool_call_result(self, tool_name, args, call_id="call_abc123"):
        result = MagicMock()
        result.content = ""
        result.tool_calls = [{"id": call_id, "name": tool_name, "args": args}]
        return result

    def _make_text_result(self, content):
        result = MagicMock()
        result.content = content
        result.tool_calls = []
        return result

    def _base_state(self):
        return {
            "customer_message": "What is the interest rate for a home loan?",
            "response": "", "history": [], "query_type": "SIMPLE", "retrieved_docs": [],
        }

    def test_respond_calls_llm_with_tools_first(self):
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm:
            mock_wt.invoke.return_value = self._make_text_result("Home loan starts at 8.75%.")
            respond(self._base_state())
        mock_wt.invoke.assert_called_once()
        mock_llm.invoke.assert_not_called()

    def test_respond_no_tool_calls_returns_first_result(self):
        expected = "FastFinance offers home, personal, business, and gold loans."
        state    = {**self._base_state(), "customer_message": "What loans do you offer?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm"):
            mock_wt.invoke.return_value = self._make_text_result(expected)
            result = respond(state)
        assert result["response"] == expected

    def test_respond_makes_second_call_when_tool_requested(self):
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Home Loan: 8.75% p.a. (CIBIL 750-900)"):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_rates", {"product_id": "home_loan"}
            )
            mock_llm.invoke.return_value = self._make_text_result(
                "Home loan rate is 8.75% p.a. for CIBIL 750+. QuickLoan | FastFinance India"
            )
            respond(self._base_state())
        mock_llm.invoke.assert_called_once()

    def test_respond_executes_tool_via_run_tool(self):
        state = {**self._base_state(), "customer_message": "What is the min CIBIL for a personal loan?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Personal Loan\n  Min CIBIL: 700...") as mock_rt:
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_eligibility", {"product_id": "personal_loan"}
            )
            mock_llm.invoke.return_value = self._make_text_result("Min CIBIL is 700.")
            respond(state)
        mock_rt.assert_called_once_with("query_eligibility", {"product_id": "personal_loan"})

    def test_respond_uses_second_call_content_as_response(self):
        final_answer = "The gold loan rate is 10.5% p.a. QuickLoan | FastFinance India"
        state = {**self._base_state(), "customer_message": "What is the gold loan interest rate?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Gold Loan: 10.50% p.a. (CIBIL 0-900)"):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_rates", {"product_id": "gold_loan"}
            )
            mock_llm.invoke.return_value = self._make_text_result(final_answer)
            result = respond(state)
        assert result["response"] == final_answer

    def test_respond_history_grows_by_two(self):
        state = {**self._base_state(), "customer_message": "What is the home loan rate?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm"):
            mock_wt.invoke.return_value = self._make_text_result("Home loan from 8.75%.")
            result = respond(state)
        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["role"] == "assistant"

    def test_respond_appends_tool_message_to_conversation(self):
        captured_messages = []

        def capture_invoke(msgs):
            captured_messages.extend(msgs)
            return MagicMock(content="Home loan: 8.75% p.a. QuickLoan | FastFinance India", tool_calls=[])

        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Home Loan: 8.75% p.a. (CIBIL 750-900)"):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_rates", {"product_id": "home_loan"}
            )
            mock_llm.invoke.side_effect = capture_invoke
            respond(self._base_state())

        from langchain_core.messages import ToolMessage as TM
        tool_messages = [m for m in captured_messages if isinstance(m, TM)]
        assert len(tool_messages) == 1
        assert "8.75" in tool_messages[0].content or "Home Loan" in tool_messages[0].content


# ---------------------------------------------------------------------------
# TestGraphRouting
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def _mock_vectorstore(self):
        vs = MagicMock()
        vs.similarity_search.return_value = []
        return vs

    def test_simple_path_calls_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "llm"), \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="SIMPLE")
            mock_wt.invoke.return_value = MagicMock(
                content="Home loan starts at 8.75%.", tool_calls=[]
            )
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-simple"}}
            graph.invoke(
                {"customer_message": "What is the home loan rate?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_called_once()

    def test_complex_path_skips_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-complex"}}
            result = graph.invoke(
                {"customer_message": "Should I take a personal loan or use my savings?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "loan officer" in result["response"] or "1800-456-7890" in result["response"]

    def test_out_of_scope_path_skips_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-oos"}}
            result = graph.invoke(
                {"customer_message": "What is the weather today?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "only help with FastFinance" in result["response"]
