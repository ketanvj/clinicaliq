"""
clinicaliq/agent.py
-------------------
Builds and runs the ClinicalIQ multi-agent LangGraph supervisor.

Session 12: Supervisor + Specialist Agent + Compliance Agent architecture.
  - Supervisor classifies into DOCTORS / SERVICES / COMPLEX / OUT_OF_SCOPE
  - Doctors Agent    uses MCP query_doctors tool
  - Services Agent   uses MCP query_services tool
  - Compliance Agent checks medical ethics; revises non-compliant responses

What you need to implement (in nodes.py first):
  TODO 1: create_compliance_agent() -- wire up the compliance sub-graph
  TODO 2: call_compliance_agent()   -- supervisor node that calls _compliance_agent

Once the TODOs in nodes.py are done:
  TODO in this file: wire specialists → compliance → END (see the TODO block below)

Run when done:
  python -m clinicaliq.agent   (from inside s12/starter/)
"""
import os
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB, MCP_SERVER_PATH
from .nodes import (
    call_compliance_agent,
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

    builder.add_node("classify",              classify)
    builder.add_node("call_doctors_agent",    call_doctors_agent)
    builder.add_node("call_services_agent",   call_services_agent)
    builder.add_node("call_compliance_agent", call_compliance_agent)
    builder.add_node("escalate",              escalate)
    builder.add_node("decline",               decline)

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_supervisor, {
        "call_doctors_agent":  "call_doctors_agent",
        "call_services_agent": "call_services_agent",
        "escalate":            "escalate",
        "decline":             "decline",
    })

    # ---------------------------------------------------------------------------
    # TODO -- Wire specialists to compliance, then compliance to END
    # ---------------------------------------------------------------------------
    # Replace these two direct-to-END edges:
    #   builder.add_edge("call_doctors_agent",  END)
    #   builder.add_edge("call_services_agent", END)
    #
    # With edges that route through the compliance agent:
    #   builder.add_edge("call_doctors_agent",   "call_compliance_agent")
    #   builder.add_edge("call_services_agent",  "call_compliance_agent")
    #   builder.add_edge("call_compliance_agent", END)
    # ---------------------------------------------------------------------------
    builder.add_edge("call_doctors_agent",  END)   # TODO: change to "call_compliance_agent"
    builder.add_edge("call_services_agent", END)   # TODO: change to "call_compliance_agent"
    # builder.add_edge("call_compliance_agent", END)  # TODO: uncomment this line

    builder.add_edge("escalate", END)
    builder.add_edge("decline",  END)

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
    print("  Architecture: Supervisor + Doctors/Services + Compliance Agent")
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
            },
            config=config,
        )

        specialist = result.get("specialist", "?")
        compliance = result.get("compliance_status", "")

        print(f"\n[Route: {result.get('query_type','?')} → {specialist}]", end="")
        if compliance:
            print(f"  [Compliance: {compliance}]", end="")
        print()
        print(f"\nClinicalIQ: {result['response']}")


if __name__ == "__main__":
    run()
