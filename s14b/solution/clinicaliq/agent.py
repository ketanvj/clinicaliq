"""
clinicaliq/agent.py
-------------------
Builds and runs the ClinicalIQ multi-agent LangGraph supervisor.

Session 14b: Same guard architecture as S14 but with LlamaGuard 3 8B.
  - Guard (two-layer) → Supervisor → [Doctors|Services] → Compliance
"""
import os
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB, MCP_SERVER_PATH
from .nodes import (
    blocked,
    call_compliance_agent,
    call_doctors_agent,
    call_services_agent,
    classify,
    decline,
    escalate,
    guard,
    route_guard,
    route_supervisor,
)
from .state import ClinicalIQState


def build_graph(checkpointer=None):
    builder = StateGraph(ClinicalIQState)

    builder.add_node("guard",                 guard)
    builder.add_node("blocked",               blocked)
    builder.add_node("classify",              classify)
    builder.add_node("call_doctors_agent",    call_doctors_agent)
    builder.add_node("call_services_agent",   call_services_agent)
    builder.add_node("call_compliance_agent", call_compliance_agent)
    builder.add_node("escalate",              escalate)
    builder.add_node("decline",               decline)

    builder.set_entry_point("guard")
    builder.add_conditional_edges("guard", route_guard, {
        "classify": "classify",
        "blocked":  "blocked",
    })
    builder.add_edge("blocked", END)

    builder.add_conditional_edges("classify", route_supervisor, {
        "call_doctors_agent":  "call_doctors_agent",
        "call_services_agent": "call_services_agent",
        "escalate":            "escalate",
        "decline":             "decline",
    })

    builder.add_edge("call_doctors_agent",    "call_compliance_agent")
    builder.add_edge("call_services_agent",   "call_compliance_agent")
    builder.add_edge("call_compliance_agent", END)

    builder.add_edge("escalate", END)
    builder.add_edge("decline",  END)

    return builder.compile(checkpointer=checkpointer)  # None = no checkpointer, Studio-safe


graph = build_graph()


def run() -> None:
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    g         = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    if not MCP_SERVER_PATH.exists():
        print(f"[ClinicalIQ] WARNING: MCP server not found at {MCP_SERVER_PATH}")

    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project    = os.environ.get("LANGCHAIN_PROJECT", "batch1-clinicaliq")

    print("=" * 60)
    print("  ClinicalIQ | Apollo Health Clinic")
    print("  Architecture: Guard (LlamaGuard 3 8B) → Supervisor → [Doctors|Services] → Compliance")
    print(f"  Tracing: {'LangSmith (' + project + ')' if tracing_on else 'off'}")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print(f"  Session: {thread_id[:8]}...")
    print("=" * 60)

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

        result = g.invoke(
            {
                "customer_message":  user_input,
                "response":          "",
                "specialist":        "",
                "retrieved_docs":    [],
                "compliance_status": "",
                "blocked_reason":    "",
            },
            config=config,
        )

        specialist = result.get("specialist", "?")
        blocked_r  = result.get("blocked_reason", "")
        compliance = result.get("compliance_status", "")

        if blocked_r:
            print(f"\n[Guard: BLOCKED ({blocked_r})]")
        else:
            print(f"\n[Route: {result.get('query_type','?')} → {specialist}]", end="")
            if compliance:
                print(f"  [Compliance: {compliance}]", end="")
            print()
        print(f"\nClinicalIQ: {result['response']}")


if __name__ == "__main__":
    run()
