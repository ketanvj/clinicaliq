"""
clinicaliq/agent.py
-------------------
STARTER FILE -- wire check_compliance into the graph (TODO 3 below).

Goal
  Route the respond node's output through a compliance filter before returning
  to the user. Non-compliant responses (banned medical-advice phrases) must be
  replaced with SAFE_COMPLIANCE_RESPONSE inside check_compliance().

What is already done for you
  - All imports including check_compliance from nodes
  - Graph nodes for classify, retrieve_docs, respond, escalate, decline
  - Conditional routing from classify

Your task
  TODO 3: Add check_compliance as a graph node and update the edges so that
          respond routes to check_compliance, and check_compliance routes to END.

Run when done
  python -m clinicaliq.agent   (from inside s09/starter/)
"""
import os
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB, MCP_SERVER_PATH
from .nodes import (
    check_compliance,
    classify,
    decline,
    escalate,
    respond,
    retrieve_docs,
    route_query,
)
from .state import ClinicalIQState


def build_graph(checkpointer=None):
    builder = StateGraph(ClinicalIQState)

    builder.add_node("classify",      classify)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("respond",       respond)
    builder.add_node("escalate",      escalate)
    builder.add_node("decline",       decline)

    builder.add_node("check_compliance", check_compliance)

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_query, {
        "retrieve_docs": "retrieve_docs",
        "escalate":      "escalate",
        "decline":       "decline",
    })

    builder.add_edge("retrieve_docs",    "respond")
    builder.add_edge("respond",          "check_compliance")
    builder.add_edge("check_compliance", END)
    builder.add_edge("escalate",         END)
    builder.add_edge("decline",          END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def run() -> None:
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    if not MCP_SERVER_PATH.exists():
        print(f"[ClinicalIQ] WARNING: MCP server not found at {MCP_SERVER_PATH}")
        print("  Complete Session 7 first.")

    print("=" * 55)
    print("  ClinicalIQ | Apollo Health Clinic")
    print("  Tools: MCP server (query_doctors, query_services)")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print(f"  Session: {thread_id[:8]}...")
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        project = os.getenv("LANGSMITH_PROJECT", "batch1-clinicaliq")
        print(f"  Tracing: LangSmith ({project})")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nClinicalIQ: Session ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nClinicalIQ: Thank you for choosing Apollo Health Clinic. Goodbye!")
            break

        result = _graph.invoke(
            {"customer_message": user_input, "response": "", "compliance_status": ""},
            config=config,
        )
        route      = result.get("query_type", "?")
        compliance = result.get("compliance_status", "")
        docs       = result.get("retrieved_docs", [])
        print(f"\n[Routed: {route}]", end="")
        if docs:
            sources = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
            print(f"  [Retrieved {len(docs)} chunk(s) from: {', '.join(sorted(sources))}]", end="")
        if compliance:
            print(f"  [Compliance: {compliance}]", end="")
        print()
        print(f"\nClinicalIQ: {result['response']}")


if __name__ == "__main__":
    run()
