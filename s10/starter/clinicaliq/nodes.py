"""
clinicaliq/nodes.py
-------------------
STARTER FILE -- Session 10: Multi-Agent Architecture Part 1.

What is already provided:
  - _agent_respond()      shared respond logic for specialist agents
  - _doctors_respond()    wraps _agent_respond with DOCTORS_SYSTEM_PROMPT
  - _services_respond()   wraps _agent_respond with SERVICES_SYSTEM_PROMPT
  - classify()            supervisor classifier (already uses 4-category prompt)
  - escalate() / decline()

Your task (3 TODOs):
  TODO 1: Implement create_doctors_agent() and create_services_agent()
  TODO 2: Implement call_doctors_agent() and call_services_agent() supervisor callers
  TODO 3: Implement route_supervisor()

Run when done:
  python -m clinicaliq.agent   (from inside s10/starter/)
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from .config import (
    CLASSIFY_SYSTEM,
    DECLINE_RESPONSE,
    DOCTORS_SYSTEM_PROMPT,
    ESCALATE_RESPONSE,
    SERVICES_SYSTEM_PROMPT,
)
from .state import ClinicalIQState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools


def _agent_respond(state: ClinicalIQState, system_prompt: str, label: str) -> dict:
    """Shared respond logic for both specialist agents. Already implemented -- no changes needed."""
    history  = state.get("history", [])
    messages = [SystemMessage(content=system_prompt)]
    for turn in history:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"])
        )
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm_with_tools.invoke(messages)

        if result.tool_calls:
            messages.append(result)
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(
                    f"[ClinicalIQ] {label} MCP: {tc['name']}({tc['args']}) "
                    f"-> {str(tool_output)[:80]}"
                )
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
            result = llm.invoke(messages)

        response_text = result.content

    except Exception as e:
        print(f"[ClinicalIQ] {label} LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    return {
        "response": response_text,
        "history":  history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response_text},
        ],
    }


def _doctors_respond(state: ClinicalIQState) -> dict:
    return _agent_respond(state, DOCTORS_SYSTEM_PROMPT, "Doctors Agent")


def _services_respond(state: ClinicalIQState) -> dict:
    return _agent_respond(state, SERVICES_SYSTEM_PROMPT, "Services Agent")


# ---------------------------------------------------------------------------
# TODO 1 of 3 -- Implement the agent factory functions
# ---------------------------------------------------------------------------
# create_doctors_agent():  StateGraph with a single "respond" node (_doctors_respond) → END
# create_services_agent(): StateGraph with a single "respond" node (_services_respond) → END
#
# Template:
#   def create_doctors_agent():
#       builder = StateGraph(ClinicalIQState)
#       builder.add_node("respond", _doctors_respond)
#       builder.set_entry_point("respond")
#       builder.add_edge("respond", END)
#       return builder.compile()
# ---------------------------------------------------------------------------
def create_doctors_agent():
    builder = StateGraph(ClinicalIQState)
    builder.add_node("respond", _doctors_respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)
    return builder.compile()


def create_services_agent():
    builder = StateGraph(ClinicalIQState)
    builder.add_node("respond", _services_respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)
    return builder.compile()


_doctors_agent  = create_doctors_agent()
_services_agent = create_services_agent()


# ---------------------------------------------------------------------------
# TODO 2 of 3 -- Implement call_doctors_agent() and call_services_agent()
# ---------------------------------------------------------------------------
# These are the supervisor's caller nodes. Each must:
#   1. Call _doctors_agent.invoke({...}) or _services_agent.invoke({...})
#      with the current message, history, and empty response/retrieved_docs/specialist
#   2. Return the specialist's response, updated history, and set "specialist" field
#      ("doctors_agent" or "services_agent")
# ---------------------------------------------------------------------------
def call_doctors_agent(state: ClinicalIQState) -> dict:
    result = _doctors_agent.invoke({
        "customer_message": state["customer_message"],
        "history":          state.get("history", []),
        "response":         "",
        "retrieved_docs":   [],
        "specialist":       "",
        "query_type":       state.get("query_type", ""),
    })
    return {
        "response":   result["response"],
        "history":    result["history"],
        "specialist": "doctors_agent",
    }


def call_services_agent(state: ClinicalIQState) -> dict:
    result = _services_agent.invoke({
        "customer_message": state["customer_message"],
        "history":          state.get("history", []),
        "response":         "",
        "retrieved_docs":   [],
        "specialist":       "",
        "query_type":       state.get("query_type", ""),
    })
    return {
        "response":   result["response"],
        "history":    result["history"],
        "specialist": "services_agent",
    }


# ---------------------------------------------------------------------------
# Supervisor classify node -- already implemented, no changes needed
# ---------------------------------------------------------------------------

def classify(state: ClinicalIQState) -> dict:
    messages = [SystemMessage(content=CLASSIFY_SYSTEM)]
    for turn in state.get("history", [])[-2:]:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"])
        )
    messages.append(HumanMessage(content=state["customer_message"]))
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"DOCTORS", "SERVICES", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "SERVICES"
    except Exception as e:
        print(f"[ClinicalIQ] Supervisor classification error: {e}")
        query_type = "SERVICES"
    return {"query_type": query_type}


def escalate(state: ClinicalIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history, "specialist": "escalated"}


def decline(state: ClinicalIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history, "specialist": "declined"}


# ---------------------------------------------------------------------------
# TODO 3 of 3 -- Implement route_supervisor()
# ---------------------------------------------------------------------------
# Map query_type → node name:
#   DOCTORS       → "call_doctors_agent"
#   COMPLEX       → "escalate"
#   OUT_OF_SCOPE  → "decline"
#   default       → "call_services_agent"
# ---------------------------------------------------------------------------
def route_supervisor(state: ClinicalIQState) -> str:
    qt = state.get("query_type", "SERVICES")
    if qt == "DOCTORS":
        return "call_doctors_agent"
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "call_services_agent"
