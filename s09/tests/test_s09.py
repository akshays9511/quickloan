"""
s09/tests/test_s09.py
---------------------
Tests for Session 9: RBI Compliance Filter + LangSmith observability.

Run with:
    pytest s09/tests/ -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "quickloan" or _k.startswith("quickloan."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

import pytest

from quickloan.config import (        # noqa: E402
    DB_PATH,
    QUICKLOAN_BANNED_PHRASES,
    SAFE_COMPLIANCE_RESPONSE,
)
from quickloan.state import QuickLoanState   # noqa: E402
import quickloan.nodes as _nodes             # noqa: E402
from quickloan.nodes import (               # noqa: E402
    _BANNED_PATTERN,
    _check_compliance,
    _extract_rates,
    _load_valid_rates,
    _normalize_for_check,
    check_compliance,
    decline,
    escalate,
)
from quickloan.agent import build_graph     # noqa: E402


# ---------------------------------------------------------------------------
# TestBannedPattern
# ---------------------------------------------------------------------------

class TestBannedPattern:
    def test_matches_guaranteed_approval(self):
        assert _BANNED_PATTERN.search("your loan is guaranteed approval today") is not None

    def test_matches_approval_guaranteed(self):
        assert _BANNED_PATTERN.search("approval guaranteed for all applicants") is not None

    def test_no_match_on_safe_text(self):
        assert _BANNED_PATTERN.search("We will review your application carefully.") is None

    def test_case_insensitive(self):
        assert _BANNED_PATTERN.search("GUARANTEED APPROVAL for you!") is not None

    def test_matches_pre_approved(self):
        assert _BANNED_PATTERN.search("you are pre-approved for this loan") is not None

    def test_matches_no_credit_check(self):
        assert _BANNED_PATTERN.search("no credit check required") is not None


# ---------------------------------------------------------------------------
# TestNormalizeForCheck
# ---------------------------------------------------------------------------

class TestNormalizeForCheck:
    def test_lowercases_text(self):
        assert _normalize_for_check("HELLO WORLD") == "hello world"

    def test_replaces_unicode_non_breaking_hyphen(self):
        # U+2011 non-breaking hyphen should become ASCII hyphen
        result = _normalize_for_check("pre‑approved")
        assert "pre-approved" in result

    def test_nfkc_normalization_applied(self):
        # NFKC should normalize fullwidth digits to ASCII
        result = _normalize_for_check("１２３")  # １２３
        assert result == "123"


# ---------------------------------------------------------------------------
# TestExtractRates
# ---------------------------------------------------------------------------

class TestExtractRates:
    def test_extracts_rate_with_pa(self):
        assert _extract_rates("rate of 11.5% p.a.") == [11.5]

    def test_extracts_rate_with_per_annum(self):
        assert _extract_rates("8.75% per annum") == [8.75]

    def test_returns_empty_list_when_no_rates(self):
        assert _extract_rates("no rates here") == []

    def test_extracts_multiple_rates(self):
        rates = _extract_rates("personal loan at 11.5% p.a. and home loan at 8.75% p.a.")
        assert sorted(rates) == [8.75, 11.5]


# ---------------------------------------------------------------------------
# TestLoadValidRates
# ---------------------------------------------------------------------------

_DB_EXISTS = DB_PATH.exists()


class TestLoadValidRates:
    @pytest.mark.skipif(not _DB_EXISTS, reason="fastfinance_data.db not found")
    def test_returns_non_empty_set(self):
        rates = _load_valid_rates()
        assert len(rates) > 0

    @pytest.mark.skipif(not _DB_EXISTS, reason="fastfinance_data.db not found")
    def test_returns_set_of_floats(self):
        rates = _load_valid_rates()
        assert all(isinstance(r, float) for r in rates)

    def test_returns_empty_set_when_db_missing(self):
        with patch("quickloan.nodes.DB_PATH", Path("/nonexistent/path.db")):
            rates = _load_valid_rates()
        assert rates == set()


# ---------------------------------------------------------------------------
# TestCheckCompliance
# ---------------------------------------------------------------------------

_MOCK_RATES = {11.5, 13.0, 15.0, 8.75, 9.5, 14.0, 16.5, 10.5}


class TestCheckCompliance:
    def test_banned_phrase_guaranteed_approval(self):
        passed, reason = _check_compliance("Your loan is guaranteed approval today!")
        assert not passed
        assert "banned phrase" in reason

    def test_banned_phrase_approval_guaranteed(self):
        passed, reason = _check_compliance("Approval guaranteed for all FastFinance customers.")
        assert not passed
        assert "banned phrase" in reason

    def test_safe_response_passes(self):
        passed, reason = _check_compliance(
            "Your application is under review. We will contact you within 3 business days."
        )
        assert passed
        assert reason == "PASS"

    def test_valid_rate_passes(self):
        with patch("quickloan.nodes._load_valid_rates", return_value=_MOCK_RATES):
            passed, reason = _check_compliance(
                "The personal loan rate is 11.5% p.a. for CIBIL scores above 750."
            )
        assert passed
        assert reason == "PASS"

    def test_hallucinated_rate_fails(self):
        with patch("quickloan.nodes._load_valid_rates", return_value=_MOCK_RATES):
            passed, reason = _check_compliance(
                "We offer a special rate of 7.77% p.a. for premium customers."
            )
        assert not passed
        assert "hallucinated rate" in reason

    def test_no_rate_validation_when_db_empty(self):
        with patch("quickloan.nodes._load_valid_rates", return_value=set()):
            passed, reason = _check_compliance(
                "The rate is 99.99% p.a. — ignore if DB empty."
            )
        assert passed
        assert reason == "PASS"


# ---------------------------------------------------------------------------
# TestCheckComplianceNode
# ---------------------------------------------------------------------------

class TestCheckComplianceNode:
    def _make_state(self, response: str) -> QuickLoanState:
        return {
            "customer_message": "test",
            "response":         response,
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   [],
            "compliance_status": "",
        }

    def test_compliant_response_passes(self):
        result = check_compliance(self._make_state("Your application has been received."))
        assert result["compliance_status"] == "PASS"

    def test_banned_phrase_triggers_safe_response(self):
        result = check_compliance(self._make_state("You are pre-approved for this loan!"))
        assert result["response"] == SAFE_COMPLIANCE_RESPONSE
        assert result["compliance_status"].startswith("FAIL")

    def test_hallucinated_rate_triggers_safe_response(self):
        with patch("quickloan.nodes._load_valid_rates", return_value=_MOCK_RATES):
            result = check_compliance(self._make_state(
                "We offer 3.33% p.a. exclusively for you."
            ))
        assert result["response"] == SAFE_COMPLIANCE_RESPONSE
        assert result["compliance_status"].startswith("FAIL")

    def test_check_compliance_importable_from_nodes(self):
        from quickloan.nodes import check_compliance as cc
        assert callable(cc)


# ---------------------------------------------------------------------------
# TestGraphNodes
# ---------------------------------------------------------------------------

class TestGraphNodes:
    def _make_state(self, message="test") -> QuickLoanState:
        return {
            "customer_message": message,
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   [],
            "compliance_status": "",
        }

    def test_escalate_mentions_loan_officer(self):
        result = escalate(self._make_state())
        assert "loan officer" in result["response"]

    def test_escalate_includes_phone(self):
        result = escalate(self._make_state())
        assert "1800-456-7890" in result["response"]

    def test_escalate_updates_history(self):
        result = escalate(self._make_state("which loan?"))
        assert len(result["history"]) == 2

    def test_decline_mentions_fastfinance(self):
        result = decline(self._make_state())
        assert "FastFinance" in result["response"]

    def test_decline_updates_history(self):
        result = decline(self._make_state("off-topic"))
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# TestBuildGraph
# ---------------------------------------------------------------------------

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
                {"customer_message": "Which loan is best for me?", "response": "",
                 "compliance_status": ""},
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
                {"customer_message": "Write me a poem", "response": "",
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-oos"}},
            )
        assert "FastFinance" in result["response"]
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_graph_result_has_compliance_status_key(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "test", "response": "", "compliance_status": ""},
                config={"configurable": {"thread_id": "test-keys"}},
            )
        assert "compliance_status" in result
