"""
clinicaliq/nodes.py
-------------------
STARTER FILE -- Session 12: Multi-Agent Architecture Part 2.

What is already provided (no changes needed):
  - _agent_respond()        shared respond logic for specialist agents
  - _doctors_respond()      wraps _agent_respond with DOCTORS_SYSTEM_PROMPT
  - _services_respond()     wraps _agent_respond with SERVICES_SYSTEM_PROMPT
  - create_doctors_agent()  / _doctors_agent
  - create_services_agent() / _services_agent
  - call_doctors_agent()    supervisor caller for doctors (already includes compliance_status)
  - call_services_agent()   supervisor caller for services (already includes compliance_status)
  - classify() / escalate() / decline() / route_supervisor()
  - _check_compliance_logic()  medical ethics phrase scanner
  - check_medical()            compliance check node
  - revise_response()          compliance revision node
  - route_compliance()         returns "revise" or END

Your task (2 TODOs):
  TODO 1: Implement create_compliance_agent()
          Wire up check_medical → route_compliance → [revise | END]
  TODO 2: Implement call_compliance_agent()
          Invoke _compliance_agent with the current state and return updated response

Run when done:
  python -m clinicaliq.agent   (from inside s12/starter/)
"""
from langsmith import traceable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from .config import (
    CLASSIFY_SYSTEM,
    CLINICALIQ_BANNED_PHRASES,
    DECLINE_RESPONSE,
    DOCTORS_SYSTEM_PROMPT,
    ESCALATE_RESPONSE,
    SAFE_COMPLIANCE_RESPONSE,
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
# Compliance helpers -- already implemented, no changes needed
# ---------------------------------------------------------------------------

@traceable(name="medical_compliance_check")
def _check_compliance_logic(draft: str) -> tuple:
    lower = draft.lower()

    for phrase in CLINICALIQ_BANNED_PHRASES:
        if phrase in lower:
            return False, f"banned phrase: '{phrase}'"

    return True, "PASS"


def check_medical(state: ClinicalIQState) -> dict:
    draft          = state["response"]
    passed, reason = _check_compliance_logic(draft)

    if not passed:
        print(f"[ClinicalIQ] Compliance FAIL: {reason}")
        return {"compliance_status": f"FAIL: {reason}"}

    print("[ClinicalIQ] Compliance PASS")
    return {"compliance_status": "PASS"}


def revise_response(state: ClinicalIQState) -> dict:
    draft  = state["response"]
    reason = state.get("compliance_status", "violation").replace("FAIL: ", "")

    prompt = (
        "You are a medical compliance officer reviewing an AI clinic assistant response.\n\n"
        f"The response was flagged for: {reason}\n\n"
        "Rewrite it to fix the violation while keeping the response helpful.\n\n"
        "Rules:\n"
        "  1. Never diagnose conditions, recommend medications, or give clinical advice.\n"
        "  2. Redirect clinical questions to speak with a nurse or doctor.\n"
        "  3. Keep the rewritten response under 150 words.\n"
        "  4. End with 'ClinicalIQ | Apollo Health Clinic'\n\n"
        f"Original response:\n{draft}\n\n"
        "Compliant rewrite:"
    )

    try:
        result       = llm.invoke([HumanMessage(content=prompt)])
        revised_text = result.content.strip() or SAFE_COMPLIANCE_RESPONSE
    except Exception as e:
        print(f"[ClinicalIQ] Compliance Agent revision error: {e}")
        revised_text = SAFE_COMPLIANCE_RESPONSE

    print("[ClinicalIQ] Compliance Agent: response revised")
    return {
        "response":          revised_text,
        "compliance_status": "REVISED",
    }


def route_compliance(state: ClinicalIQState) -> str:
    return "revise" if state.get("compliance_status", "").startswith("FAIL") else END


# ---------------------------------------------------------------------------
# TODO 1 of 2 -- Implement create_compliance_agent()
# ---------------------------------------------------------------------------
# Build a StateGraph that:
#   1. Adds a "check_medical" node and a "revise" node
#   2. Sets "check_medical" as the entry point
#   3. Adds conditional edges from "check_medical" using route_compliance:
#         {"revise": "revise", END: END}
#   4. Adds a direct edge from "revise" → END
#   5. Returns builder.compile()
#
# Template:
#   def create_compliance_agent():
#       builder = StateGraph(ClinicalIQState)
#       builder.add_node("check_medical", check_medical)
#       builder.add_node("revise",        revise_response)
#       builder.set_entry_point("check_medical")
#       builder.add_conditional_edges(
#           "check_medical", route_compliance,
#           {"revise": "revise", END: END},
#       )
#       builder.add_edge("revise", END)
#       return builder.compile()
# ---------------------------------------------------------------------------
def create_compliance_agent():
    builder = StateGraph(ClinicalIQState)
    builder.add_node("check_medical", check_medical)
    builder.add_node("revise",        revise_response)
    builder.set_entry_point("check_medical")
    builder.add_conditional_edges(
        "check_medical", route_compliance,
        {"revise": "revise", END: END},
    )
    builder.add_edge("revise", END)
    return builder.compile()


_compliance_agent = create_compliance_agent()


# ---------------------------------------------------------------------------
# Supervisor classify and route nodes -- already implemented, no changes needed
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


def call_doctors_agent(state: ClinicalIQState) -> dict:
    print("[ClinicalIQ] Supervisor → Doctors Agent")
    result = _doctors_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "DOCTORS"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "doctors_agent",
    }


def call_services_agent(state: ClinicalIQState) -> dict:
    print("[ClinicalIQ] Supervisor → Services Agent")
    result = _services_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "SERVICES"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "services_agent",
    }


# ---------------------------------------------------------------------------
# TODO 2 of 2 -- Implement call_compliance_agent()
# ---------------------------------------------------------------------------
# This supervisor node runs the compliance sub-graph on the current response.
#
# Steps:
#   1. Print "[ClinicalIQ] Supervisor → Compliance Agent"
#   2. Call _compliance_agent.invoke({...}) passing all state fields including
#      "compliance_status": "" to reset it for a fresh check
#   3. Return {"response": result["response"],
#              "compliance_status": result.get("compliance_status", "PASS")}
#
# Template:
#   def call_compliance_agent(state: ClinicalIQState) -> dict:
#       print("[ClinicalIQ] Supervisor → Compliance Agent")
#       result = _compliance_agent.invoke({
#           "customer_message":  state["customer_message"],
#           "response":          state["response"],
#           "history":           state.get("history", []),
#           "query_type":        state.get("query_type", ""),
#           "retrieved_docs":    state.get("retrieved_docs", []),
#           "specialist":        state.get("specialist", ""),
#           "compliance_status": "",
#       })
#       return {
#           "response":          result["response"],
#           "compliance_status": result.get("compliance_status", "PASS"),
#       }
# ---------------------------------------------------------------------------
def call_compliance_agent(state: ClinicalIQState) -> dict:
    print("[ClinicalIQ] Supervisor \u2192 Compliance Agent")
    result = _compliance_agent.invoke({
        "customer_message":  state["customer_message"],
        "response":          state["response"],
        "history":           state.get("history", []),
        "query_type":        state.get("query_type", ""),
        "retrieved_docs":    state.get("retrieved_docs", []),
        "specialist":        state.get("specialist", ""),
        "compliance_status": "",
    })
    return {
        "response":          result["response"],
        "compliance_status": result.get("compliance_status", "PASS"),
    }


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


def route_supervisor(state: ClinicalIQState) -> str:
    qt = state.get("query_type", "SERVICES")
    if qt == "DOCTORS":
        return "call_doctors_agent"
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "call_services_agent"
