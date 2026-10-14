"""
clinicaliq/state.py
-------------------
The shared state that flows through the LangGraph graph.

Session 2 adds conversation history so the agent can remember
previous turns within the same session.
"""
from typing import TypedDict


class ClinicalIQState(TypedDict):
    customer_message: str    # the question the patient typed
    response:         str    # the answer ClinicalIQ will return

    history:          list[dict]
