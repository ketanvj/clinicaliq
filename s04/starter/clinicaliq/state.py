"""
clinicaliq/state.py
-------------------
The shared state that flows through the LangGraph graph.

Session 4 adds retrieved_docs so RAG chunks can flow from
retrieve_docs() into respond().
"""
from typing import TypedDict


class ClinicalIQState(TypedDict):
    customer_message: str
    response:         str
    history:          list[dict]
    query_type:       str

    retrieved_docs:   list[str]
