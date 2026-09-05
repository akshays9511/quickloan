"""
s06/tests/live_eval.py — Live evaluation for QuickLoan S06 solution
---------------------------------------------------------------------
Runs the full 40-question golden dataset through the S05 QuickLoan agent
using the REAL Groq LLM. No mocks.

This is the same pipeline as `python s06/solution/evaluate.py` but called
from the tests/ directory so it fits the pattern participants already know
from S04 and S05.

Why this exists alongside pytest
─────────────────────────────────
`pytest test_s06.py` mocks the LLM and the graph — it catches structural
bugs in evaluate.py (missing functions, wrong return shapes, etc.).

This script catches defects that only appear with a real LLM:
  • Judge prompt brittleness — giving high scores to evasive answers
  • Rate answers that don't cite query_rates() output
  • COMPLEX/OOS queries that slip through the classifier
  • Low pass rate on a category — signals a prompt or data issue

Run this script (from the quickloan/ directory):
    python s06/tests/live_eval.py

Expected output: overall pass rate >= 80% (32/40)
The detailed per-category breakdown tells you where defects are.

Cost: ~40 Groq API calls for agent + ~20 for LLM judge (~60 total, ~30s).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_DIR      = Path(__file__).parent
S06_DIR        = TESTS_DIR.parent
S05_DIR        = S06_DIR.parent / "s05"
QUICKLOAN_ROOT = S06_DIR.parent   # cohort-1/quickloan/

load_dotenv(QUICKLOAN_ROOT / ".env")

# Add s06/solution (for evaluate.py) and s05/solution (for quickloan package)
sys.path.insert(0, str(S06_DIR / "solution"))
sys.path.insert(0, str(S05_DIR / "solution"))

from evaluate import load_dataset, run_evaluation, generate_report, print_report, DATASET_PATH
from langgraph.checkpoint.memory import MemorySaver
from quickloan.agent import build_graph

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  QuickLoan S06 — Live Evaluation  (real Groq, 40-question golden dataset)")
print("=" * 80)

graph   = build_graph(checkpointer=MemorySaver())
dataset = load_dataset(DATASET_PATH)
print(f"\nRunning evaluation on {len(dataset)} questions...")
print("-" * 60)

results = run_evaluation(graph, dataset)
report  = generate_report(results)
print_report(report)

overall = report.get("overall", {})
passed  = overall.get("passed", 0)
total   = overall.get("total", len(dataset))
rate    = overall.get("pass_rate", 0)

print("\n" + "=" * 80)
if rate >= 0.80:
    print(f"  ✓ PASS — {passed}/{total} ({rate:.0%}) — meets the >=80% release threshold")
else:
    print(f"  ✗ FAIL — {passed}/{total} ({rate:.0%}) — below 80% release threshold")
    print("  Fix the categories with low pass rates above before releasing S06.")
print("=" * 80 + "\n")
