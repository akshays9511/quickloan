"""
s05/tests/live_eval.py — Live evaluation for QuickLoan S05 solution
---------------------------------------------------------------------
Runs 15 test queries directly against graph.invoke() using the REAL Groq
LLM, the REAL ChromaDB vectorstore, and the REAL SQLite tools database.
No mocks.

Why this exists alongside pytest
─────────────────────────────────
`pytest test_s05.py` mocks the LLM, tools, and vectorstore — it runs in
~2 seconds and catches structural bugs.

This script catches defects that only appear with a real LLM:
  • Classifier brittleness — rate query classified COMPLEX or OUT_OF_SCOPE
  • Tool not called    — LLM answers a rate query from memory instead of
                         calling query_rates() (violates Rule 3)
  • Wrong tool args    — query_rates("home") instead of query_rates("home_loan")
  • Stale history      — previous tool output leaks into follow-up answer
  • Personal advice not escalated — system prompt rule violation

Run this script (from the quickloan/ directory):
    python s05/tests/live_eval.py

Expected output: 15/15 passed
If any test fails, the response snippet is printed so you can diagnose.

Cost: ~15 Groq API calls (~8–12 seconds total, well within free tier).
"""

import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_DIR      = Path(__file__).parent
S05_DIR        = TESTS_DIR.parent
SOLUTION_DIR   = S05_DIR / "solution"
QUICKLOAN_ROOT = S05_DIR.parent   # cohort-1/quickloan/

load_dotenv(QUICKLOAN_ROOT / ".env")
sys.path.insert(0, str(SOLUTION_DIR))

from langgraph.checkpoint.memory import MemorySaver
from quickloan.agent import build_graph
from quickloan.config import DECLINE_RESPONSE, ESCALATE_RESPONSE
from quickloan.tools import query_rates   # used to get live rates for validation

graph = build_graph(checkpointer=MemorySaver())

# ── Pull live rates from the DB for validation ────────────────────────────────
# Proves tool calls return real data, not hallucinated values.
try:
    _all_rates   = query_rates.invoke({"product_id": "all"})
    _rate_values = set(re.findall(r'\d+\.\d+', _all_rates))   # e.g. "10.5", "12.0"
except Exception:
    _rate_values = set()

# ── Test cases ────────────────────────────────────────────────────────────────
# (label, query, expected_route, expected_behaviour)
#
# expected_route     : "SIMPLE" | "COMPLEX" | "OUT_OF_SCOPE"
# expected_behaviour :
#     "answer"           — substantive reply (not escalate, not decline)
#     "answer_with_rate" — not escalate/decline AND response contains a rate
#                          value that matches the live DB (tool called)
#     "escalate"         — response == ESCALATE_RESPONSE
#     "decline"          — response == DECLINE_RESPONSE

TEST_CASES = [
    # ── Tool: interest rates ─────────────────────────────────────────────────
    ("Home loan rate",  "What is the home loan interest rate at FastFinance?",
     "SIMPLE", "answer_with_rate"),

    ("Personal rate",   "What are the personal loan interest rates?",
     "SIMPLE", "answer_with_rate"),

    ("Gold rate",       "Tell me the gold loan interest rate at FastFinance.",
     "SIMPLE", "answer_with_rate"),

    ("Business rate",   "What is the interest rate for a business loan?",
     "SIMPLE", "answer_with_rate"),

    # ── Tool: eligibility ────────────────────────────────────────────────────
    ("Eligibility",     "What salary do I need to qualify for a home loan?",
     "SIMPLE", "answer"),

    ("Income check",    "Am I eligible for a personal loan with a salary of Rs. 25,000?",
     "SIMPLE", "answer"),

    # ── RAG policy questions (no tool expected) ──────────────────────────────
    ("Gold loan docs",  "What documents are required for a gold loan?",
     "SIMPLE", "answer"),

    ("Prepayment",      "What is FastFinance's policy on loan prepayment charges?",
     "SIMPLE", "answer"),

    # ── Complex → escalate ───────────────────────────────────────────────────
    ("Best loan",       "Which loan product is best for my business expansion?",
     "COMPLEX", "escalate"),

    ("Advice",          "Should I take a personal loan or use my savings?",
     "COMPLEX", "escalate"),

    # ── Out of scope → decline ───────────────────────────────────────────────
    ("Stocks",          "What stocks should I invest in right now?",
     "OUT_OF_SCOPE", "decline"),

    ("Savings",         "Which bank has the best savings account rate?",
     "OUT_OF_SCOPE", "decline"),

    ("Cricket",         "Who won the cricket match yesterday?",
     "OUT_OF_SCOPE", "decline"),

    # ── Follow-up memory ─────────────────────────────────────────────────────
    ("Follow-up 1",     "What is the home loan interest rate at FastFinance?",
     "SIMPLE", "answer_with_rate"),

    ("Follow-up 2",     "And what about the business loan rate?",
     "SIMPLE", "answer_with_rate"),
]

FOLLOW_UP_START = 13   # index of "Follow-up 1" — share one thread from here
SHARED_THREAD   = "live-eval-memory-thread"


# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  QuickLoan S05 — Live Evaluation  (real Groq + real SQLite tools)")
print(f"  DB rate values found : {sorted(_rate_values) or 'none — DB may be missing'}")
print("=" * 80)

results = []
for i, (label, query, exp_route, exp_behaviour) in enumerate(TEST_CASES):
    thread = SHARED_THREAD if i >= FOLLOW_UP_START else f"eval-{i}"
    cfg    = {"configurable": {"thread_id": thread}}

    result   = graph.invoke({"customer_message": query, "response": ""}, config=cfg)
    route    = result.get("query_type", "?")
    response = result["response"]

    if response == ESCALATE_RESPONSE:
        actual = "escalate"
    elif response == DECLINE_RESPONSE:
        actual = "decline"
    else:
        actual = "answer"

    # For answer_with_rate: verify response contains a rate matching the live DB
    if exp_behaviour == "answer_with_rate" and actual == "answer":
        resp_rates = set(re.findall(r'\d+\.\d+', response))
        if _rate_values and not resp_rates.isdisjoint(_rate_values):
            actual = "answer_with_rate"        # rate matches DB ✓
        elif _rate_values and resp_rates.isdisjoint(_rate_values):
            actual = "answer_hallucinated"     # rate doesn't match DB ✗
        elif not _rate_values and "%" in response:
            actual = "answer_with_rate"        # DB unavailable, has % at least
        # else: stays "answer" — no rate found at all

    route_ok = route == exp_route
    act_ok   = actual == exp_behaviour
    passed   = route_ok and act_ok
    results.append(passed)

    mark = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{mark}  [{label}]")
    print(f"     Q      : {query[:72]}")
    print(f"     Route  : {route} (expected {exp_route}) {'✓' if route_ok else '✗'}")
    print(f"     Action : {actual} (expected {exp_behaviour}) {'✓' if act_ok else '✗'}")
    if not passed:
        snippet = response[:200].replace("\n", " ")
        print(f"     Resp   : {snippet}...")

total  = len(results)
passed = sum(results)
print("\n" + "=" * 80)
print(f"  Result : {passed}/{total} passed")
if passed < total:
    print(f"  {'─' * 40}")
    print(f"  {total - passed} failure(s) above need fixing before S05 is release-ready.")
    print()
    print("  Common causes:")
    print("    answer_hallucinated → LLM answered rate question without calling query_rates()")
    print("    wrong route         → CLASSIFY_SYSTEM prompt needs tightening")
    print("    escalate got answer → COMPLEX case fell through to SIMPLE path")
print("=" * 80 + "\n")
