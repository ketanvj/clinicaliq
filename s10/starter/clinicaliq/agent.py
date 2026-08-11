"""
clinicaliq/agent.py
-------------------
Builds and runs the ClinicalIQ multi-agent LangGraph supervisor.

Session 10: Supervisor + Specialist Agent architecture.
  - Supervisor classifies into DOCTORS / SERVICES / COMPLEX / OUT_OF_SCOPE
  - Doctors Agent  uses MCP query_doctors tool
  - Services Agent uses MCP query_services tool

What you need to implement (in nodes.py first):
  TODO 1: create_doctors_agent() and create_services_agent() factory functions
  TODO 2: call_doctors_agent() and call_services_agent() supervisor caller nodes
  TODO 3: route_supervisor() routing function

This agent.py already wires the graph correctly -- once the TODOs in nodes.py
are implemented this will work end-to-end.

Run when done:
  python -m clinicaliq.agent   (from inside s10/starter/)
"""
import os
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB, MCP_SERVER_PATH
from .nodes import (
    call_doctors_agent,
    call_services_agent,
    classify,
    decline,
    escalate,
    route_supervisor,
)
from .state import ClinicalIQState


def build_graph(checkpointer=None):
    builder = StateGraph(ClinicalIQState)

    builder.add_node("classify",            classify)
    builder.add_node("call_doctors_agent",  call_doctors_agent)
    builder.add_node("call_services_agent", call_services_agent)
    builder.add_node("escalate",            escalate)
    builder.add_node("decline",             decline)

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_supervisor, {
        "call_doctors_agent":  "call_doctors_agent",
        "call_services_agent": "call_services_agent",
        "escalate":            "escalate",
        "decline":             "decline",
    })

    builder.add_edge("call_doctors_agent",  END)
    builder.add_edge("call_services_agent", END)
    builder.add_edge("escalate",            END)
    builder.add_edge("decline",             END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def run() -> None:
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    g         = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    if not MCP_SERVER_PATH.exists():
        print(f"[ClinicalIQ] WARNING: MCP server not found at {MCP_SERVER_PATH}")
        print("  Complete Session 7 first.")

    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project    = os.environ.get("LANGCHAIN_PROJECT", "batch1-clinicaliq")

    print("=" * 60)
    print("  ClinicalIQ | Apollo Health Clinic")
    print("  Architecture: Supervisor + Doctors Agent + Services Agent")
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
            {"customer_message": user_input, "response": "",
             "specialist": "", "retrieved_docs": []},
            config=config,
        )
        specialist = result.get("specialist", "?")
        print(f"\n[Route: {result.get('query_type','?')} → {specialist}]")
        print(f"\nClinicalIQ: {result['response']}")


if __name__ == "__main__":
    run()
