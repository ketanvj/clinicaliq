"""
s05/tests/live_eval.py — Live evaluation for ClinicalIQ S05 solution
----------------------------------------------------------------------
Runs 14 test queries directly against graph.invoke() using the REAL Groq
LLM, the REAL ChromaDB vectorstore, and the REAL SQLite tools database.
No mocks.

Why this exists alongside pytest
─────────────────────────────────
`pytest test_s05.py` mocks the LLM, tools, and vectorstore — it runs in
~2 seconds and catches structural bugs.

This script catches defects that only appear with a real LLM:
  • Classifier brittleness — doctor query classified COMPLEX or OUT_OF_SCOPE
  • Tool not called    — LLM answers a fee query from memory instead of
                         calling query_doctors() (violates Rule 3)
  • Wrong tool args    — query_doctors("heart") instead of query_doctors("Cardiology")
  • Stale history      — previous tool output leaks into follow-up answer
  • Clinical advice not escalated — system prompt rule violation

Run this script (from the clinicaliq/ directory):
    python s05/tests/live_eval.py

Expected output: 14/14 passed
If any test fails, the response snippet is printed so you can diagnose.

Cost: ~14 Groq API calls (~8–12 seconds total, well within free tier).
"""

import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_DIR       = Path(__file__).parent
S05_DIR         = TESTS_DIR.parent
SOLUTION_DIR    = S05_DIR / "solution"
CLINICALIQ_ROOT = S05_DIR.parent   # cohort-1/clinicaliq/

load_dotenv(CLINICALIQ_ROOT / ".env")
sys.path.insert(0, str(SOLUTION_DIR))

from langgraph.checkpoint.memory import MemorySaver
from clinicaliq.agent import build_graph
from clinicaliq.config import DECLINE_RESPONSE, ESCALATE_RESPONSE
from clinicaliq.tools import query_doctors   # used to get live fees for validation

graph = build_graph(checkpointer=MemorySaver())

# ── Pull live consultation fees from the DB for validation ────────────────────
# Proves tool calls return real data, not hallucinated values.
try:
    _all_doctors = query_doctors.invoke({"specialty": "all"})
    _fee_values  = set(re.findall(r'\d+', _all_doctors))   # fee integers e.g. "500", "800"
except Exception:
    _fee_values = set()

# ── Test cases ────────────────────────────────────────────────────────────────
# (label, query, expected_route, expected_behaviour)
#
# expected_route     : "SIMPLE" | "COMPLEX" | "OUT_OF_SCOPE"
# expected_behaviour :
#     "answer"          — substantive reply (not escalate, not decline)
#     "answer_with_fee" — not escalate/decline AND response contains a fee
#                         value that matches the live DB (tool called)
#     "escalate"        — response == ESCALATE_RESPONSE
#     "decline"         — response == DECLINE_RESPONSE

TEST_CASES = [
    # ── Tool: doctors ────────────────────────────────────────────────────────
    ("Cardiologists",   "Are there any cardiologists available at Apollo?",
     "SIMPLE", "answer_with_fee"),

    ("Ortho doctors",   "Who are the orthopedic doctors at Apollo Health Clinic?",
     "SIMPLE", "answer_with_fee"),

    ("Consult fees",    "What are the doctor consultation fees at Apollo?",
     "SIMPLE", "answer_with_fee"),

    # ── Tool: services ───────────────────────────────────────────────────────
    ("Diagnostics",     "What diagnostic services does Apollo offer?",
     "SIMPLE", "answer_with_fee"),

    ("Radiology",       "What tests are available in the radiology department?",
     "SIMPLE", "answer_with_fee"),

    # ── RAG policy questions (no tool expected) ──────────────────────────────
    ("MRI prep",        "How do I prepare for an MRI scan at Apollo?",
     "SIMPLE", "answer"),

    ("Appointments",    "How do I book an appointment at Apollo Health Clinic?",
     "SIMPLE", "answer"),

    # ── Complex → escalate ───────────────────────────────────────────────────
    ("Which doctor",    "Which doctor should I see for my back pain?",
     "COMPLEX", "escalate"),

    ("Symptoms",        "What treatment do you recommend for my chest tightness?",
     "COMPLEX", "escalate"),

    # ── Out of scope → decline ───────────────────────────────────────────────
    ("Weather",         "What is the weather today?",
     "OUT_OF_SCOPE", "decline"),

    ("Pharmacy",        "Which pharmacy near me has the best prices?",
     "OUT_OF_SCOPE", "decline"),

    ("Cricket",         "Who won the IPL this year?",
     "OUT_OF_SCOPE", "decline"),

    # ── Follow-up memory ─────────────────────────────────────────────────────
    ("Follow-up 1",     "Who are the cardiologists at Apollo?",
     "SIMPLE", "answer_with_fee"),

    ("Follow-up 2",     "What are their consultation fees?",
     "SIMPLE", "answer_with_fee"),
]

FOLLOW_UP_START = 12   # index of "Follow-up 1" — share one thread from here
SHARED_THREAD   = "live-eval-memory-thread"


# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  ClinicalIQ S05 — Live Evaluation  (real Groq + real SQLite tools)")
print(f"  DB fee values found : {sorted(_fee_values)[:8] or 'none — DB may be missing'}")
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

    # For answer_with_fee: verify the response contains a number matching the DB
    if exp_behaviour == "answer_with_fee" and actual == "answer":
        resp_nums = set(re.findall(r'\d+', response))
        if _fee_values and not resp_nums.isdisjoint(_fee_values):
            actual = "answer_with_fee"         # DB value confirmed ✓
        elif _fee_values and resp_nums.isdisjoint(_fee_values):
            actual = "answer_hallucinated"     # number doesn't match DB ✗
        elif not _fee_values and re.search(r'₹\d+|\d+\s*(?:per consultation|fee)', response):
            actual = "answer_with_fee"         # DB unavailable, has fee at least
        # else: stays "answer" — no fee found at all

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
    print("    answer_hallucinated → LLM answered fee question without calling query_doctors()")
    print("    wrong route         → CLASSIFY_SYSTEM prompt needs tightening")
    print("    escalate got answer → COMPLEX case fell through to SIMPLE path")
print("=" * 80 + "\n")
