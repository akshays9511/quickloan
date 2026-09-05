"""
s08/tests/test_s08.py
---------------------
Tests for Session 8: MCP Agent Integration, using MultiServerMCPClient.

Run with:
    pytest s08/tests/ -v
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "quickloan" or _k.startswith("quickloan."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from quickloan.config import MCP_SERVER_PATH    # noqa: E402
from quickloan.state import QuickLoanState      # noqa: E402
import quickloan.tools as _tools                # noqa: E402
import quickloan.nodes as _nodes                # noqa: E402
from quickloan.tools import _extract_text, _run_tool  # noqa: E402
from quickloan.nodes import classify, decline, escalate  # noqa: E402
from quickloan.agent import build_graph         # noqa: E402


class TestMCPServerPath:
    def test_mcp_server_path_is_path_object(self):
        assert isinstance(MCP_SERVER_PATH, Path)

    def test_mcp_server_path_points_to_s07(self):
        assert "s07" in str(MCP_SERVER_PATH)

    def test_mcp_server_path_filename(self):
        assert MCP_SERVER_PATH.name == "mcp_server.py"

    def test_mcp_server_path_exists(self):
        assert MCP_SERVER_PATH.exists(), (
            f"S07 MCP server not found at {MCP_SERVER_PATH}. "
            "Complete Session 7 before running Session 8 tests."
        )


class TestMCPToolLoading:
    def test_mcp_tools_has_two_tools(self):
        assert len(_tools.mcp_tools) == 2

    def test_mcp_tools_contains_query_rates(self):
        names = [t.name for t in _tools.mcp_tools]
        assert "query_rates" in names

    def test_mcp_tools_contains_query_eligibility(self):
        names = [t.name for t in _tools.mcp_tools]
        assert "query_eligibility" in names

    def test_tool_registry_maps_names_to_tools(self):
        assert set(_tools._tool_registry.keys()) == {"query_rates", "query_eligibility"}

    def test_tools_have_descriptions(self):
        for t in _tools.mcp_tools:
            assert t.description and len(t.description) > 10

    def test_llm_with_tools_is_bound(self):
        assert _tools.llm_with_tools is not None
        assert _tools.llm_with_tools is not _tools.llm


class TestExtractText:
    def test_single_text_block(self):
        assert _extract_text([{"type": "text", "text": "Home Loan: 8.50% p.a."}]) == "Home Loan: 8.50% p.a."

    def test_multiple_text_blocks_joined_with_newline(self):
        result = _extract_text([{"type": "text", "text": "A"}, {"type": "text", "text": "B"}])
        assert result == "A\nB"

    def test_empty_list_returns_empty_string(self):
        assert _extract_text([]) == ""

    def test_non_list_falls_back_to_str(self):
        assert _extract_text("already a string") == "already a string"

    def test_block_missing_text_key_defaults_to_empty(self):
        assert _extract_text([{"type": "text"}]) == ""


class TestRunTool:
    def _fake_tool(self, return_value=None, side_effect=None):
        fake = MagicMock()
        if side_effect is not None:
            fake.ainvoke = AsyncMock(side_effect=side_effect)
        else:
            fake.ainvoke = AsyncMock(return_value=return_value)
        return fake

    def test_dispatches_query_rates(self):
        fake = self._fake_tool(return_value=[{"type": "text", "text": "8.50% p.a."}])
        with patch.dict(_tools._tool_registry, {"query_rates": fake}):
            result = _run_tool("query_rates", {"product_id": "home_loan"})
        assert result == "8.50% p.a."
        fake.ainvoke.assert_called_once_with({"product_id": "home_loan"})

    def test_dispatches_query_eligibility(self):
        fake = self._fake_tool(return_value=[{"type": "text", "text": "Min CIBIL: 700"}])
        with patch.dict(_tools._tool_registry, {"query_eligibility": fake}):
            result = _run_tool("query_eligibility", {"product_id": "home_loan"})
        assert result == "Min CIBIL: 700"

    def test_unknown_tool_returns_error_string(self):
        result = _run_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_tool_exception_returns_error_string(self):
        fake = self._fake_tool(side_effect=RuntimeError("crash"))
        with patch.dict(_tools._tool_registry, {"query_rates": fake}):
            result = _run_tool("query_rates", {})
        assert "Tool error" in result


class TestGraphNodes:
    def _make_state(self, message="test") -> QuickLoanState:
        return {"customer_message": message, "response": "", "history": [],
                "query_type": "SIMPLE", "retrieved_docs": []}

    def test_escalate_response_mentions_loan_officer(self):
        result = escalate(self._make_state())
        assert "loan officer" in result["response"]

    def test_escalate_response_includes_phone(self):
        result = escalate(self._make_state())
        assert "1800-456-7890" in result["response"]

    def test_escalate_updates_history(self):
        result = escalate(self._make_state("which loan?"))
        assert len(result["history"]) == 2

    def test_decline_response_mentions_fastfinance(self):
        result = decline(self._make_state())
        assert "FastFinance" in result["response"]

    def test_decline_updates_history(self):
        result = decline(self._make_state("off-topic"))
        assert len(result["history"]) == 2


class TestBuildGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert build_graph(checkpointer=MemorySaver()) is not None

    def test_graph_invoke_complex_escalates(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Which loan is best for me?", "response": ""},
                config={"configurable": {"thread_id": "test-complex"}},
            )
        assert "loan officer" in result["response"]
        assert result["query_type"] == "COMPLEX"

    def test_graph_invoke_oos_declines(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Write me a poem", "response": ""},
                config={"configurable": {"thread_id": "test-oos"}},
            )
        assert "FastFinance" in result["response"]
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_graph_invoke_returns_expected_keys(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "test", "response": ""},
                config={"configurable": {"thread_id": "test-keys"}},
            )
        assert "response" in result and "query_type" in result
