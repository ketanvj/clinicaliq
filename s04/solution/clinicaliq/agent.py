"""
clinicaliq/agent.py
-------------------
Graph construction and the terminal loop.

Run the agent from the repo root:
    cd cohort-1/clinicaliq/s04/solution
    python -m clinicaliq.agent

Session 4 graph:
    START --> classify --> route_query --> retrieve_docs --> respond --> END
                                      --> decline --> END

What is new in Session 4:
  - retrieve_docs node added between classify and respond
  - route_query is now 2-way (IN_SCOPE / OUT_OF_SCOPE)
  - respond() uses retrieved chunks to answer from policy documents
"""
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .config import CHECKPOINT_DB, ESCALATE_RESPONSE
from .nodes import classify, decline, escalate, respond, retrieve_docs, route_query
from .state import ClinicalIQState


def build_graph(checkpointer=None):
    builder = StateGraph(ClinicalIQState)

    builder.add_node("classify",      classify)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("respond",       respond)
    builder.add_node("decline",       decline)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges("classify", route_query, {
        "retrieve_docs": "retrieve_docs",
        "decline":       "decline",
    })

    builder.add_edge("retrieve_docs", "respond")
    builder.add_edge("respond",       END)
    builder.add_edge("decline",       END)

    return builder.compile(checkpointer=checkpointer)


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
    print(f"  Session : {thread_id[:8]}...")
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
        route    = result.get("query_type", "?")
        docs     = result.get("retrieved_docs", [])
        response = result["response"]
        print(f"\n[Routed: {route}]", end="")
        if docs and response != ESCALATE_RESPONSE:
            sources = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
            print(f"  [Retrieved {len(docs)} chunk(s) from: {', '.join(sorted(sources))}]")
        else:
            print()
        print(f"\nClinicalIQ: {result['response']}")


if __name__ == "__main__":
    run()
