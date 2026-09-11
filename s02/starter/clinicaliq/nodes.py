"""
clinicaliq/nodes.py
-------------------
Graph nodes for ClinicalIQ.

Session 2: the respond() node now carries conversation history across
turns so the LLM can refer back to earlier messages.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import SYSTEM_PROMPT
from .state import ClinicalIQState
from .tools import llm


def respond(state: ClinicalIQState) -> dict:
    history = state.get("history", [])

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm.invoke(messages)
        response_text = result.content
    except Exception as e:
        print(f"[ClinicalIQ] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}
