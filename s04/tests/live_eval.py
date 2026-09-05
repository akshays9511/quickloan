"""
s04/tests/live_eval.py — Live evaluation for QuickLoan S04 solution
---------------------------------------------------------------------
Runs 14 test queries directly against graph.invoke() using the REAL Groq
LLM and the REAL ChromaDB vectorstore. No mocks.

Why this exists alongside pytest
─────────────────────────────────
`pytest test_s04.py` mocks the LLM and vectorstore — it runs in ~2 seconds
and catches structural bugs (wrong node wired, state field missing, etc.).

This script catches a different class of defects that only appear with a
real LLM:
  • Classifier prompt brittleness  — e.g. loan policy query classified OUT_OF_SCOPE
  • Stale state leaking across turns — retrieved_docs from turn 1 in turn 2
  • Personal advice not being escalated (system prompt rule violation)
  • Fragment not escalating (no docs → should escalate)

Run this script (from the quickloan/ directory):
    python s04/tests/live_eval.py

Expected output: 14/14 passed
If any test fails, the response snippet is printed so you can diagnose.

Cost: ~14 Groq API calls (~5–10 seconds total, well within free tier).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_DIR     = Path(__file__).parent
S04_DIR       = TESTS_DIR.parent
SOLUTION_DIR  = S04_DIR / "solution"
QUICKLOAN_ROOT = S04_DIR.parent   # cohort-1/quickloan/

load_dotenv(QUICKLOAN_ROOT / ".env")
sys.path.insert(0, str(SOLUTION_DIR))

from langgraph.checkpoint.memory import MemorySaver
from quickloan.agent import build_graph
from quickloan.config import DECLINE_RESPONSE, ESCALATE_RESPONSE

graph = build_graph(checkpointer=MemorySaver())

# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry: (label, query, expected_route, expected_behaviour)
# expected_route    : "IN_SCOPE" | "OUT_OF_SCOPE"
# expected_behaviour: "answer" | "escalate" | "decline"
TEST_CASES = [
    # Factual RAG — retrieved docs should drive the answer
    ("Home loan docs",  "What documents are required to apply for a home loan?",  "IN_SCOPE",     "answer"),
    ("Loan tenure",     "What is the maximum loan tenure for a personal loan?",   "IN_SCOPE",     "answer"),
    ("Prepayment",      "What is FastFinance's policy on loan prepayment?",       "IN_SCOPE",     "answer"),
    ("Gold loan",       "How do I apply for a gold loan at FastFinance?",         "IN_SCOPE",     "answer"),
    ("Processing fee",  "What is the processing fee for a business loan?",        "IN_SCOPE",     "answer"),

    # Personal advice — IN_SCOPE (about FastFinance) but respond() must escalate
    ("Advice 1",        "Which loan product is best for me?",                     "IN_SCOPE",     "escalate"),
    ("Advice 2",        "Should I take a personal loan or sell my assets?",       "IN_SCOPE",     "escalate"),

    # Fragment — single word, very low cosine score → no docs → escalate
    ("Fragment",        "Which",                                                   "IN_SCOPE",     "escalate"),

    # Out of scope
    ("Weather",         "What is the weather in Mumbai today?",                   "OUT_OF_SCOPE", "decline"),
    ("Stocks",          "Recommend me a good stock to invest in",                 "OUT_OF_SCOPE", "decline"),
    ("Cricket",         "Who won the cricket match yesterday?",                   "OUT_OF_SCOPE", "decline"),
    ("Savings",         "Which bank has the best savings account interest rate?", "OUT_OF_SCOPE", "decline"),

    # Follow-up memory — same thread, second query should use prior context
    ("Follow-up 1",     "What documents do I need for a personal loan?",          "IN_SCOPE",     "answer"),
    ("Follow-up 2",     "And what about for a business loan?",                    "IN_SCOPE",     "answer"),
]

FOLLOW_UP_START = 12   # index of "Follow-up 1" — share one thread from here
SHARED_THREAD   = "live-eval-memory-thread"

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  QuickLoan S04 — Live Evaluation  (real Groq, no mocks)")
print("=" * 80)

results = []
for i, (label, query, exp_route, exp_behaviour) in enumerate(TEST_CASES):
    thread = SHARED_THREAD if i >= FOLLOW_UP_START else f"eval-{i}"
    cfg    = {"configurable": {"thread_id": thread}}

    result   = graph.invoke({"customer_message": query, "response": ""}, config=cfg)
    route    = result.get("query_type", "?")
    docs     = result.get("retrieved_docs", [])
    response = result["response"]

    if response == ESCALATE_RESPONSE:
        actual = "escalate"
    elif response == DECLINE_RESPONSE:
        actual = "decline"
    else:
        actual = "answer"

    route_ok = route == exp_route
    act_ok   = actual == exp_behaviour
    passed   = route_ok and act_ok
    results.append(passed)

    sources = ""
    if docs and response != ESCALATE_RESPONSE:
        src_set = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
        sources = f"  [{len(docs)} chunk(s): {', '.join(sorted(src_set))}]"

    mark = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{mark}  [{label}]")
    print(f"     Q      : {query[:70]}")
    print(f"     Route  : {route} (expected {exp_route}) {'✓' if route_ok else '✗'}")
    print(f"     Action : {actual} (expected {exp_behaviour}) {'✓' if act_ok else '✗'}{sources}")
    if not passed:
        snippet = response[:150].replace("\n", " ")
        print(f"     Resp   : {snippet}...")

total  = len(results)
passed = sum(results)
print("\n" + "=" * 80)
print(f"  Result : {passed}/{total} passed")
if passed < total:
    print(f"  {'─' * 40}")
    print(f"  {total - passed} failure(s) above need fixing before this session is release-ready.")
print("=" * 80 + "\n")
