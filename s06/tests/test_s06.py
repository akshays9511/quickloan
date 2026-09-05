"""
s06/tests/test_s06.py
---------------------
Tests for Session 6: Baseline Evaluation.

Run with:
    pytest s06/tests/ -v

These tests do not invoke the real QuickLoan graph or make live LLM calls.
All graph invocations and judge calls are mocked. The tests verify:
  - golden dataset structure and content
  - judge output parsing (including edge cases)
  - response evaluation logic for SIMPLE, COMPLEX, and OUT_OF_SCOPE
  - report aggregation arithmetic
  - run_evaluation calls the graph and evaluate_response for each item

Test groups:
  TestGoldenDataset      -- dataset loads; has 40 items; required fields present
  TestLoadDataset        -- file loading and validation errors
  TestParseJudgeResponse -- score extraction, clamping, malformed output
  TestEvaluateResponse   -- deterministic scoring for all three route types
  TestGenerateReport     -- pass rate, avg score, by_category, failures list
  TestRunEvaluation      -- orchestration: graph invoked, results returned
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

import evaluate  # noqa: E402
from evaluate import (  # noqa: E402
    DATASET_PATH,
    PASS_SCORE,
    REQUIRED_FIELDS,
    evaluate_response,
    generate_report,
    load_dataset,
    parse_judge_response,
    run_evaluation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset():
    """The real golden dataset loaded from disk."""
    return load_dataset(DATASET_PATH)


@pytest.fixture
def simple_item():
    return {
        "id": "S01",
        "query": "What is the personal loan interest rate?",
        "expected_route": "SIMPLE",
        "category": "loan_rates",
        "criteria": ["11.5", "Personal Loan"],
        "must_not_contain": ["HDFC"],
    }


@pytest.fixture
def complex_item():
    return {
        "id": "C01",
        "query": "I earn Rs. 55,000. Can I get a home loan?",
        "expected_route": "COMPLEX",
        "category": "complex",
        "criteria": ["loan officer"],
        "must_not_contain": [],
    }


@pytest.fixture
def oos_item():
    return {
        "id": "O01",
        "query": "Write me a poem.",
        "expected_route": "OUT_OF_SCOPE",
        "category": "oos",
        "criteria": ["FastFinance India"],
        "must_not_contain": [],
    }


# ---------------------------------------------------------------------------
# TestGoldenDataset
# ---------------------------------------------------------------------------

class TestGoldenDataset:
    def test_dataset_loads(self, dataset):
        assert isinstance(dataset, list)

    def test_dataset_has_40_items(self, dataset):
        assert len(dataset) == 40, f"Expected 40 items, got {len(dataset)}"

    def test_dataset_has_20_simple(self, dataset):
        simple = [d for d in dataset if d["expected_route"] == "SIMPLE"]
        assert len(simple) == 20

    def test_dataset_has_10_complex(self, dataset):
        complex_ = [d for d in dataset if d["expected_route"] == "COMPLEX"]
        assert len(complex_) == 10

    def test_dataset_has_10_oos(self, dataset):
        oos = [d for d in dataset if d["expected_route"] == "OUT_OF_SCOPE"]
        assert len(oos) == 10

    def test_all_items_have_required_fields(self, dataset):
        for item in dataset:
            missing = REQUIRED_FIELDS - set(item.keys())
            assert not missing, f"Item {item.get('id')} missing: {missing}"

    def test_all_ids_are_unique(self, dataset):
        ids = [d["id"] for d in dataset]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in dataset"

    def test_expected_route_values_are_valid(self, dataset):
        valid = {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}
        for item in dataset:
            assert item["expected_route"] in valid, (
                f"Item {item['id']} has invalid expected_route: {item['expected_route']}"
            )

    def test_criteria_is_a_list(self, dataset):
        for item in dataset:
            assert isinstance(item["criteria"], list), (
                f"Item {item['id']}: criteria must be a list"
            )

    def test_simple_items_have_non_empty_criteria(self, dataset):
        for item in dataset:
            if item["expected_route"] == "SIMPLE":
                assert len(item["criteria"]) > 0, (
                    f"SIMPLE item {item['id']} has empty criteria"
                )


# ---------------------------------------------------------------------------
# TestLoadDataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_load_returns_list(self):
        result = load_dataset(DATASET_PATH)
        assert isinstance(result, list)

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.json")

    def test_load_raises_on_missing_fields(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([{"id": "X01", "query": "hello"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="missing fields"):
            load_dataset(bad)

    def test_load_raises_with_item_id_in_message(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps([{"id": "MISSING_FIELDS", "query": "q"}]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="MISSING_FIELDS"):
            load_dataset(bad)

    def test_load_valid_file_returns_correct_count(self, tmp_path):
        items = [
            {"id": f"T{i}", "query": "q", "expected_route": "SIMPLE",
             "category": "cat", "criteria": ["x"]}
            for i in range(5)
        ]
        path = tmp_path / "valid.json"
        path.write_text(json.dumps(items), encoding="utf-8")
        result = load_dataset(path)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# TestParseJudgeResponse
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def test_parses_well_formed_output(self):
        output = "SCORE: 4\nREASON: The response is accurate."
        score, reason = parse_judge_response(output)
        assert score == 4
        assert "accurate" in reason

    def test_parses_score_1(self):
        score, _ = parse_judge_response("SCORE: 1\nREASON: Wrong rate given.")
        assert score == 1

    def test_parses_score_5(self):
        score, _ = parse_judge_response("SCORE: 5\nREASON: Perfect answer.")
        assert score == 5

    def test_clamps_score_above_5(self):
        score, _ = parse_judge_response("SCORE: 9\nREASON: impossible.")
        assert score == 5

    def test_clamps_score_below_1(self):
        score, _ = parse_judge_response("SCORE: -3\nREASON: bad.")
        assert score == 1

    def test_returns_zero_on_missing_score(self):
        score, _ = parse_judge_response("REASON: No score line.")
        assert score == 0

    def test_returns_default_reason_on_missing_reason(self):
        score, reason = parse_judge_response("SCORE: 3")
        assert score == 3
        assert "parse" in reason.lower() or reason

    def test_handles_empty_output(self):
        score, reason = parse_judge_response("")
        assert score == 0
        assert isinstance(reason, str)

    def test_handles_extra_whitespace(self):
        output = "  SCORE:  4  \n  REASON:  Good response.  "
        score, reason = parse_judge_response(output)
        assert score == 4
        assert "Good response" in reason

    def test_handles_non_integer_score(self):
        score, _ = parse_judge_response("SCORE: four\nREASON: Spelled out.")
        assert score == 0

    def test_reason_with_colon_in_text(self):
        output = "SCORE: 3\nREASON: Rate is correct: 11.5% p.a."
        _, reason = parse_judge_response(output)
        assert "11.5" in reason


# ---------------------------------------------------------------------------
# TestEvaluateResponse
# ---------------------------------------------------------------------------

class TestEvaluateResponse:
    def test_complex_passes_when_criteria_met(self, complex_item):
        result = {
            "query_type": "COMPLEX",
            "response": "Please speak with a loan officer for personalised advice.",
        }
        with patch("evaluate.llm_judge") as mock_judge:
            out = evaluate_response(complex_item, result)
        mock_judge.assert_not_called()
        assert out["passed"] is True
        assert out["score"] == 5

    def test_complex_fails_when_criteria_missing(self, complex_item):
        result = {
            "query_type": "COMPLEX",
            "response": "Please call us for help.",
        }
        out = evaluate_response(complex_item, result)
        assert out["passed"] is False
        assert out["score"] == 1

    def test_complex_fails_on_wrong_route(self, complex_item):
        result = {
            "query_type": "SIMPLE",
            "response": "Speak with a loan officer.",
        }
        out = evaluate_response(complex_item, result)
        assert out["route_correct"] is False
        assert out["passed"] is False

    def test_oos_passes_when_criteria_met(self, oos_item):
        result = {
            "query_type": "OUT_OF_SCOPE",
            "response": "I can only help with FastFinance India loan products and services.",
        }
        out = evaluate_response(oos_item, result)
        assert out["passed"] is True
        assert out["score"] == 5

    def test_oos_fails_when_criteria_missing(self, oos_item):
        result = {
            "query_type": "OUT_OF_SCOPE",
            "response": "That is outside my expertise.",
        }
        out = evaluate_response(oos_item, result)
        assert out["passed"] is False

    def test_simple_passes_with_high_score(self, simple_item):
        result = {
            "query_type": "SIMPLE",
            "response": "The Personal Loan rate is 11.5% p.a. QuickLoan | FastFinance India",
        }
        with patch("evaluate.llm_judge", return_value=(5, "Perfect.")):
            out = evaluate_response(simple_item, result)
        assert out["passed"] is True
        assert out["score"] == 5

    def test_simple_fails_with_low_score(self, simple_item):
        result = {
            "query_type": "SIMPLE",
            "response": "I am not sure about the rate.",
        }
        with patch("evaluate.llm_judge", return_value=(2, "Incomplete.")):
            out = evaluate_response(simple_item, result)
        assert out["passed"] is False

    def test_simple_fails_on_wrong_route(self, simple_item):
        result = {
            "query_type": "COMPLEX",
            "response": "The Personal Loan rate is 11.5%.",
        }
        with patch("evaluate.llm_judge", return_value=(5, "Good.")):
            out = evaluate_response(simple_item, result)
        assert out["route_correct"] is False
        assert out["passed"] is False

    def test_simple_fails_on_forbidden_content(self, simple_item):
        result = {
            "query_type": "SIMPLE",
            "response": "FastFinance Personal Loan is 11.5%. HDFC offers 10.5%.",
        }
        with patch("evaluate.llm_judge", return_value=(4, "Good but mentions HDFC.")):
            out = evaluate_response(simple_item, result)
        assert out["forbidden_found"] == ["HDFC"]
        assert out["passed"] is False

    def test_result_has_all_required_keys(self, simple_item):
        result = {
            "query_type": "SIMPLE",
            "response": "Personal Loan: 11.5% p.a.",
        }
        with patch("evaluate.llm_judge", return_value=(4, "Good.")):
            out = evaluate_response(simple_item, result)
        for key in ("id", "query", "category", "expected_route", "actual_route",
                    "route_correct", "score", "reason", "forbidden_found", "passed", "response"):
            assert key in out, f"Missing key in evaluate_response output: {key}"

    def test_actual_route_is_from_result(self, simple_item):
        result = {"query_type": "SIMPLE", "response": "11.5% Personal Loan."}
        with patch("evaluate.llm_judge", return_value=(4, "Good.")):
            out = evaluate_response(simple_item, result)
        assert out["actual_route"] == "SIMPLE"

    def test_handles_missing_query_type_in_result(self, simple_item):
        result = {"response": "The rate is 11.5%."}
        with patch("evaluate.llm_judge", return_value=(4, "Good.")):
            out = evaluate_response(simple_item, result)
        assert out["actual_route"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def _make_results(self, specs):
        """Build a results list from (category, passed, score) tuples."""
        out = []
        for i, (cat, passed, score) in enumerate(specs):
            route = "COMPLEX" if cat == "complex" else ("oos" if cat == "oos" else "SIMPLE")
            out.append({
                "id": f"T{i:02d}",
                "query": f"Question {i}",
                "category": cat,
                "expected_route": route,
                "actual_route": route,
                "route_correct": True,
                "score": score,
                "reason": "test reason",
                "forbidden_found": [],
                "passed": passed,
                "response": "some response",
            })
        return out

    def test_total_count(self):
        results = self._make_results([("loan_rates", True, 5)] * 10)
        report = generate_report(results)
        assert report["total"] == 10

    def test_passed_count(self):
        results = self._make_results(
            [("loan_rates", True, 5)] * 7 + [("loan_rates", False, 1)] * 3
        )
        report = generate_report(results)
        assert report["passed"] == 7

    def test_failed_count(self):
        results = self._make_results(
            [("loan_rates", True, 5)] * 7 + [("loan_rates", False, 1)] * 3
        )
        report = generate_report(results)
        assert report["failed"] == 3

    def test_pass_rate_calculation(self):
        results = self._make_results(
            [("loan_rates", True, 5)] * 3 + [("loan_rates", False, 1)] * 1
        )
        report = generate_report(results)
        assert abs(report["pass_rate"] - 0.75) < 0.01

    def test_average_score_excludes_complex_and_oos(self):
        results = self._make_results([
            ("loan_rates", True, 4),
            ("loan_rates", True, 4),
            ("complex",    True, 5),
            ("oos",        True, 5),
        ])
        report = generate_report(results)
        assert report["average_score"] == 4.0

    def test_by_category_contains_all_categories(self):
        results = self._make_results([
            ("loan_rates", True, 5),
            ("eligibility", False, 2),
            ("loan_products", True, 4),
        ])
        report = generate_report(results)
        assert "loan_rates" in report["by_category"]
        assert "eligibility" in report["by_category"]
        assert "loan_products" in report["by_category"]

    def test_by_category_pass_rate(self):
        results = self._make_results([
            ("loan_rates", True, 5),
            ("loan_rates", True, 4),
            ("loan_rates", False, 1),
        ])
        report = generate_report(results)
        cat = report["by_category"]["loan_rates"]
        assert abs(cat["pass_rate"] - 2/3) < 0.01

    def test_failures_list_contains_failed_items(self):
        results = self._make_results([
            ("loan_rates", True, 5),
            ("loan_rates", False, 2),
        ])
        report = generate_report(results)
        assert len(report["failures"]) == 1
        assert report["failures"][0]["id"] == "T01"

    def test_failures_list_has_required_keys(self):
        results = self._make_results([("loan_rates", False, 1)])
        report = generate_report(results)
        for key in ("id", "query", "reason", "score", "actual_route"):
            assert key in report["failures"][0]

    def test_empty_results_does_not_crash(self):
        report = generate_report([])
        assert report["total"] == 0
        assert report["pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# TestRunEvaluation
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    def _make_dataset(self, n=3):
        return [
            {
                "id": f"T{i:02d}",
                "query": f"Question {i}",
                "expected_route": "SIMPLE",
                "category": "loan_rates",
                "criteria": ["test"],
                "must_not_contain": [],
            }
            for i in range(n)
        ]

    def test_run_evaluation_returns_list(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"query_type": "SIMPLE", "response": "test response"}
        dataset = self._make_dataset(2)
        with patch("evaluate.evaluate_response") as mock_eval:
            mock_eval.return_value = {
                "id": "T00", "query": "q", "category": "loan_rates",
                "expected_route": "SIMPLE", "actual_route": "SIMPLE",
                "route_correct": True, "score": 4, "reason": "good",
                "forbidden_found": [], "passed": True, "response": "r",
            }
            results = run_evaluation(mock_graph, dataset)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_run_evaluation_calls_graph_for_each_item(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"query_type": "SIMPLE", "response": "ok"}
        dataset = self._make_dataset(3)
        with patch("evaluate.evaluate_response") as mock_eval:
            mock_eval.return_value = {
                "id": "x", "query": "q", "category": "c", "expected_route": "SIMPLE",
                "actual_route": "SIMPLE", "route_correct": True, "score": 4,
                "reason": "r", "forbidden_found": [], "passed": True, "response": "r",
            }
            run_evaluation(mock_graph, dataset)
        assert mock_graph.invoke.call_count == 3

    def test_run_evaluation_uses_unique_thread_ids(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"query_type": "SIMPLE", "response": "ok"}
        dataset = self._make_dataset(2)
        with patch("evaluate.evaluate_response") as mock_eval:
            mock_eval.return_value = {
                "id": "x", "query": "q", "category": "c", "expected_route": "SIMPLE",
                "actual_route": "SIMPLE", "route_correct": True, "score": 4,
                "reason": "r", "forbidden_found": [], "passed": True, "response": "r",
            }
            run_evaluation(mock_graph, dataset)
        thread_ids = [
            call.kwargs["config"]["configurable"]["thread_id"]
            for call in mock_graph.invoke.call_args_list
        ]
        assert len(thread_ids) == len(set(thread_ids)), "Thread IDs must be unique per item"

    def test_run_evaluation_handles_graph_exception(self):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("Graph crashed")
        dataset = self._make_dataset(1)
        with patch("evaluate.evaluate_response") as mock_eval:
            mock_eval.return_value = {
                "id": "x", "query": "q", "category": "c", "expected_route": "SIMPLE",
                "actual_route": "ERROR", "route_correct": False, "score": 0,
                "reason": "error", "forbidden_found": [], "passed": False, "response": "Graph error",
            }
            results = run_evaluation(mock_graph, dataset)
        assert len(results) == 1
