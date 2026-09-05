"""
QuickLoan -- Session 6: Baseline Evaluation (US-05)
====================================================

What you build this session
  A structured evaluation pipeline that runs 40 questions from a golden
  dataset through the Session 5 QuickLoan agent, scores each response,
  and produces a pass-rate report broken down by question category.

  Routing is evaluated deterministically (COMPLEX and OUT_OF_SCOPE have
  canned responses with known keywords). SIMPLE responses are scored 1-5
  by an LLM judge. A response passes if it is correctly routed, scores
  >= 3 out of 5, and contains no forbidden content.

Run
  python s06/starter/evaluate.py

Golden dataset
  s06/data/golden_dataset.json  --  40 Q&A items:
    20 SIMPLE  (loan rates, eligibility, loan products, loan policy)
    10 COMPLEX (personalised eligibility, comparisons, EMI calculations)
    10 OUT_OF_SCOPE (off-topic, competitor comparisons)
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

MODEL_NAME  = "openai/gpt-oss-120b"  # updated from llama-4-scout (no longer available)
PASS_SCORE  = 3

DATA_DIR     = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_DIR / "golden_dataset.json"

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

judge_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0.0,
    max_tokens=100,
)

JUDGE_PROMPT = """You are evaluating a loan AI assistant's response to a customer question.

Customer question:
{question}

The response should cover these points:
{criteria_list}

Assistant response:
{response}

Score the response on a scale of 1 to 5:
  5 = Excellent: all required points covered, factually accurate, professional
  4 = Good: most points covered, minor gaps
  3 = Acceptable: the key information is present but incomplete
  2 = Poor: missing important information or contains inaccuracies
  1 = Fail: refuses to answer, wrong information, or off-topic

Reply in exactly this format (two lines, no other text):
SCORE: <integer 1-5>
REASON: <one sentence explaining the score>"""


def parse_judge_response(output: str) -> tuple[int, str]:
    """Extract the integer score and one-line reason from the judge LLM output.

    # TODO 1: Implement this function.
    #
    # Parse the judge LLM output and return (score, reason).
    #
    # Rules:
    #   - Scan each line; if it starts with "SCORE:" (case-insensitive), parse
    #     the integer after the colon and clamp it to [1, 5].
    #   - If it starts with "REASON:" (case-insensitive), capture the text after
    #     the colon as the reason.
    #   - If parsing fails or SCORE line is absent, return score=0.
    #   - Default reason: "Could not parse judge output"
    #
    # Hints:
    #   line.upper().startswith("SCORE:")
    #   int(line.split(":", 1)[1].strip())
    #   max(1, min(5, raw))
    """
    raise NotImplementedError("TODO 1: implement parse_judge_response")


def llm_judge(question: str, criteria: list[str], response: str) -> tuple[int, str]:
    """Ask the judge LLM to score a SIMPLE response."""
    criteria_list = "\n".join(f"  - {c}" for c in criteria) if criteria else "  - (none specified)"
    prompt = JUDGE_PROMPT.format(
        question=question,
        criteria_list=criteria_list,
        response=response,
    )
    try:
        result = judge_llm.invoke([
            SystemMessage(content="You are a strict but fair evaluation judge."),
            HumanMessage(content=prompt),
        ])
        return parse_judge_response(result.content)
    except Exception as e:
        return 0, f"Judge error: {e}"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"id", "query", "expected_route", "category", "criteria"}


def load_dataset(path: Path) -> list[dict]:
    """Load and validate the golden dataset from a JSON file.

    # TODO 2: Implement this function.
    #
    # Steps:
    #   1. Open and json.load the file at `path`.
    #   2. For each item, compute missing = REQUIRED_FIELDS - set(item.keys()).
    #   3. If missing is non-empty, raise ValueError with a message that includes
    #      the item index, item id (use item.get('id', '?')), and the missing fields.
    #   4. Return the dataset list.
    #
    # FileNotFoundError bubbles up naturally from open() -- do not catch it.
    """
    raise NotImplementedError("TODO 2: implement load_dataset")


# ---------------------------------------------------------------------------
# Response evaluation
# ---------------------------------------------------------------------------

def evaluate_response(item: dict, result: dict) -> dict:
    """Score a single graph result against its golden dataset entry.

    # TODO 3: Implement this function.
    #
    # Steps:
    #   1. Extract actual_route from result.get("query_type", "UNKNOWN").
    #   2. Extract response from result.get("response", "").
    #   3. route_correct = (actual_route == item["expected_route"])
    #   4. criteria = item.get("criteria", [])
    #   5. must_not = item.get("must_not_contain", [])
    #   6. criteria_met = all(c.lower() in response.lower() for c in criteria)
    #   7. forbidden_found = [f for f in must_not if f.lower() in response.lower()]
    #   8. If expected_route is COMPLEX or OUT_OF_SCOPE:
    #        score = 5 if criteria_met else 1
    #        reason = "Canned response criteria met." or "Canned response keyword missing."
    #        passed = route_correct and criteria_met and not forbidden_found
    #      Else (SIMPLE):
    #        score, reason = llm_judge(item["query"], criteria, response)
    #        passed = route_correct and score >= PASS_SCORE and not forbidden_found
    #   9. Return a dict with keys:
    #        id, query, category, expected_route, actual_route,
    #        route_correct, score, reason, forbidden_found, passed, response
    """
    raise NotImplementedError("TODO 3: implement evaluate_response")


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(graph, dataset: list[dict]) -> list[dict]:
    """Invoke the graph on every dataset item and return a list of eval results."""
    results = []
    for item in dataset:
        config = {"configurable": {"thread_id": f"eval-{item['id']}"}}
        try:
            graph_result = graph.invoke(
                {"customer_message": item["query"], "response": ""},
                config=config,
            )
        except Exception as e:
            graph_result = {
                "query_type": "ERROR",
                "response": f"Graph error: {e}",
            }
        eval_result = evaluate_response(item, graph_result)
        status = "PASS" if eval_result["passed"] else "FAIL"
        print(f"  [{status}] {item['id']}: {item['query'][:60]}")
        results.append(eval_result)
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict]) -> dict:
    """Aggregate evaluation results into a structured report.

    # TODO 4: Implement this function.
    #
    # Steps:
    #   1. total = len(results), passed = count of passed, failed = total - passed
    #   2. simple_scores: scores from items whose category is NOT "complex" or "oos"
    #      AND score > 0. avg_score = mean of simple_scores (0.0 if empty).
    #   3. by_category: dict of {cat: {"total": n, "passed": n, "pass_rate": float}}
    #      Build it by iterating results and grouping by r["category"].
    #   4. failures: list of dicts for items where passed is False.
    #      Each dict has: id, query, reason, score, actual_route.
    #   5. Return dict with keys:
    #        total, passed, failed, pass_rate, average_score, by_category, failures
    #      Use round(avg_score, 2) for average_score.
    """
    raise NotImplementedError("TODO 4: implement generate_report")


def print_report(report: dict) -> None:
    """Print the evaluation report to stdout in a readable format."""
    print("\n" + "=" * 60)
    print("  QuickLoan Baseline Evaluation Report")
    print("=" * 60)
    print(f"  Total questions : {report['total']}")
    print(f"  Passed          : {report['passed']}")
    print(f"  Failed          : {report['failed']}")
    print(f"  Pass rate       : {report['pass_rate']:.0%}")
    print(f"  Avg SIMPLE score: {report['average_score']} / 5")
    print()
    print("  By category:")
    for cat, data in sorted(report["by_category"].items()):
        bar = "#" * data["passed"] + "-" * (data["total"] - data["passed"])
        print(f"    {cat:<15} [{bar}] {data['passed']}/{data['total']} ({data['pass_rate']:.0%})")

    if report["failures"]:
        print()
        print(f"  Failed items ({len(report['failures'])}):")
        for f in report["failures"]:
            print(f"    {f['id']}: (route={f['actual_route']}, score={f['score']}) {f['query'][:55]}")
            print(f"         {f['reason']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load the golden dataset, build the S05 graph, run evaluation, print report."""
    s05_dir = Path(__file__).parent.parent.parent / "s05" / "starter"
    sys.path.insert(0, str(s05_dir))

    from langgraph.checkpoint.memory import MemorySaver
    from quickloan.agent import build_graph

    graph = build_graph(checkpointer=MemorySaver())

    dataset = load_dataset(DATASET_PATH)
    print(f"\nRunning evaluation on {len(dataset)} questions...")
    print("-" * 60)

    results = run_evaluation(graph, dataset)
    report  = generate_report(results)
    print_report(report)


if __name__ == "__main__":
    main()
