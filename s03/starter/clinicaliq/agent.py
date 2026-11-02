"""
clinicaliq/agent.py
-----------------
Graph construction and the terminal loop.

Run the agent from the repo root:
    cd cohort-1/clinicaliq/s03/starter
    python -m clinicaliq.agent

Session 3 graph:
    START --> classify --> route_query --> {respond | escalate | decline} --> END

What is new in Session 3:
  - build_graph() wires three new nodes (classify, escalate, decline)
  - Entry point moves from 'respond' to 'classify'
  - add_conditional_edges() replaces the single add_edge out of the entry point
  - route_query (from nodes.py) is passed as the routing function
"""
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .config import CHECKPOINT_DB
from .nodes import classify, decline, escalate, respond, route_query
from .state import ClinicalIQState


# ---------------------------------------------------------------------------
# TODO 4 of 4 -- implement build_graph()
# ---------------------------------------------------------------------------
# Build the Session 3 graph inside build_graph():
#
#   1. Create the builder:
#        builder = StateGraph(ClinicalIQState)
#
#   2. Register all four nodes:
#        builder.add_node("classify", classify)
#        builder.add_node("respond",  respond)
#        builder.add_node("escalate", escalate)
#        builder.add_node("decline",  decline)
#
#   3. Set the entry point using add_edge from START (not set_entry_point):
#        builder.add_edge(START, "classify")
#
#   4. Add the conditional edge out of "classify":
#        builder.add_conditional_edges("classify", route_query, {
#            "respond":  "respond",
#            "escalate": "escalate",
#            "decline":  "decline",
#        })
#      The third argument (the path map dict) is required for LangGraph Studio
#      to draw the routing arrows. Without it the graph still runs, but Studio
#      cannot visualise the conditional branches.
#
#   5. Connect all three terminal nodes to END:
#        builder.add_edge("respond",  END)
#        builder.add_edge("escalate", END)
#        builder.add_edge("decline",  END)
#
#   6. Compile with the checkpointer and return:
#        return builder.compile(checkpointer=checkpointer)
#
# WHY checkpointer=None as the default?
# The module-level `graph = build_graph()` below must have NO checkpointer.
# LangGraph Studio rejects graphs compiled with a checkpointer attached.
# The run() function passes SqliteSaver explicitly for the terminal app.
#
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Build and compile the ClinicalIQ LangGraph graph."""
    # TODO 4: implement build_graph() -- see instructions above
    raise NotImplementedError("TODO 4: implement build_graph() in clinicaliq/agent.py")


# Module-level graph instance required by langgraph.json for LangGraph Studio.
graph = build_graph()


def run() -> None:
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    print("=" * 55)
    print("  ClinicalIQ | Apollo Health Clinic")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print(f"  Session: {thread_id[:8]}...")
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
            {"customer_message": user_input, "response": ""},
            config=config,
        )
        route = result.get("query_type", "?")
        print(f"\n[Routed: {route}]")
        print(f"\nClinicalIQ: {result['response']}")


if __name__ == "__main__":
    run()
