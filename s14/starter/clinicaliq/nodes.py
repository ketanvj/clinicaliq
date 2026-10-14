"""
clinicaliq/nodes.py — Session 14 Starter
-----------------------------------------
Add the two-layer Input Guard to the ClinicalIQ agent.

TODO tasks for participants:
  1. Import INJECTION_PATTERNS, PII_PATTERNS, LLAMAGUARD_THRESHOLD and the guard
     response constants from config.py.
  2. Pre-compile the patterns (re.compile) once at module load.
  3. Implement _llamaguard_safe(message) → tuple[bool, float]:
       - Invoke llamaguard_llm with the message as a HumanMessage
       - Parse the float score from result.content
       - Return (score < LLAMAGUARD_THRESHOLD, score)
       - Fail-open on exception: return (True, -1.0)
  4. Implement guard(state) → dict:
       - Strip invisible Unicode (optional, see solution)
       - NFKD-normalise the message
       - Layer 1a: check PII patterns — return {"blocked_reason": "pii", "llamaguard_score": -1.0}
       - Layer 1b: check injection patterns — return {"blocked_reason": "injection", "llamaguard_score": -1.0}
       - Layer 2: call _llamaguard_safe — return {"blocked_reason": "llamaguard", "llamaguard_score": score}
       - Return {"blocked_reason": "", "llamaguard_score": score} when clean
  5. Implement blocked(state) → dict:
       - Choose GUARD_PII_RESPONSE, GUARD_UNSAFE_RESPONSE, or GUARD_BLOCKED_RESPONSE
         based on state["blocked_reason"]
       - Set specialist to "guard"
       - Update history
  6. Implement route_guard(state) → str:
       - Return "blocked" if state has a non-empty blocked_reason, else "classify"
  7. Update call_doctors_agent and call_services_agent to pass
     "blocked_reason": "" and "llamaguard_score": -1.0 to sub-agents.

Everything else (compliance, specialist agents, supervisor) is carried forward
from Session 13 unchanged.
"""
import re
import unicodedata
from typing import Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langgraph.graph import END, StateGraph

from .config import (
    CLASSIFY_SYSTEM,
    CLINICALIQ_BANNED_PHRASES,
    DECLINE_RESPONSE,
    DOCTORS_SYSTEM_PROMPT,
    ESCALATE_RESPONSE,
    GUARD_BLOCKED_RESPONSE,
    GUARD_PII_RESPONSE,
    GUARD_UNSAFE_RESPONSE,
    INJECTION_PATTERNS,
    LLAMAGUARD_THRESHOLD,
    PII_PATTERNS,
    SAFE_COMPLIANCE_RESPONSE,
    SERVICES_SYSTEM_PROMPT,
)
from .state import ClinicalIQState
from .tools import _run_tool, classifier_llm, llamaguard_llm, llm, llm_with_tools

_stream_callback: Optional[Callable[[str], None]] = None

_pii_compiled       = [re.compile(p)               for p in PII_PATTERNS]
_injection_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

_INVISIBLE_UNICODE_RE = re.compile(
    "[\U000E0000-\U000E007F︀-️​‌‍⁠]"
)


# ---------------------------------------------------------------------------
# TODO 1: Implement _llamaguard_safe
# ---------------------------------------------------------------------------

def _llamaguard_safe(message: str) -> tuple[bool, float]:
    try:
        result = llamaguard_llm.invoke([HumanMessage(content=message)])
        score = float(result.content.strip())
        return score < LLAMAGUARD_THRESHOLD, score
    except Exception:
        return True, -1.0


# ---------------------------------------------------------------------------
# TODO 2: Implement guard
# ---------------------------------------------------------------------------

@traceable(name="input_guard")
def guard(state: ClinicalIQState) -> dict:
    raw = state["customer_message"]
    msg = unicodedata.normalize("NFKD", _INVISIBLE_UNICODE_RE.sub("", raw))

    for rx in _pii_compiled:
        if rx.search(msg):
            print("[ClinicalIQ] Guard: PII detected — blocked")
            return {"blocked_reason": "pii", "llamaguard_score": -1.0}

    for rx in _injection_compiled:
        if rx.search(msg):
            print("[ClinicalIQ] Guard: injection (regex) detected — blocked")
            return {"blocked_reason": "injection", "llamaguard_score": -1.0}

    safe, score = _llamaguard_safe(msg)
    if not safe:
        print("[ClinicalIQ] Guard: LlamaGuard flagged message — blocked")
        return {"blocked_reason": "llamaguard", "llamaguard_score": score}

    return {"blocked_reason": "", "llamaguard_score": score}


# ---------------------------------------------------------------------------
# TODO 3: Implement blocked
# ---------------------------------------------------------------------------

def blocked(state: ClinicalIQState) -> dict:
    reason = state.get("blocked_reason", "injection")
    if reason == "pii":
        response = GUARD_PII_RESPONSE
    elif reason == "llamaguard":
        response = GUARD_UNSAFE_RESPONSE
    else:
        response = GUARD_BLOCKED_RESPONSE
    return {
        "response":   response,
        "specialist": "guard",
        "history": state.get("history", []) + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response},
        ],
    }


# ---------------------------------------------------------------------------
# TODO 4: Implement route_guard
# ---------------------------------------------------------------------------

def route_guard(state: ClinicalIQState) -> str:
    return "blocked" if state.get("blocked_reason") else "classify"


# ---------------------------------------------------------------------------
# Compliance helpers (unchanged from S13)
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
    return {"response": revised_text, "compliance_status": "REVISED"}


def route_compliance(state: ClinicalIQState) -> str:
    return "revise" if state.get("compliance_status", "").startswith("FAIL") else END


def create_compliance_agent():
    builder = StateGraph(ClinicalIQState)
    builder.add_node("check_medical", check_medical)
    builder.add_node("revise",        revise_response)
    builder.set_entry_point("check_medical")
    builder.add_conditional_edges(
        "check_medical",
        route_compliance,
        {"revise": "revise", END: END},
    )
    builder.add_edge("revise", END)
    return builder.compile()


_compliance_agent = create_compliance_agent()


# ---------------------------------------------------------------------------
# Specialist agents (unchanged from S13, but invoke state with S14 fields)
# ---------------------------------------------------------------------------

def _agent_respond(state: ClinicalIQState, system_prompt: str, label: str) -> dict:
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
                print(f"[ClinicalIQ] {label} MCP: {tc['name']}({tc['args']}) -> {str(tool_output)[:80]}")
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
            if _stream_callback is not None:
                response_text = ""
                for chunk in llm.stream(messages):
                    if chunk.content:
                        response_text += chunk.content
                        _stream_callback(chunk.content)
            else:
                response_text = llm.invoke(messages).content
        else:
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
# Supervisor nodes
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
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
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
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "services_agent",
    }


def call_compliance_agent(state: ClinicalIQState) -> dict:
    print("[ClinicalIQ] Supervisor → Compliance Agent")
    result = _compliance_agent.invoke({
        "customer_message":  state["customer_message"],
        "response":          state["response"],
        "history":           state.get("history", []),
        "query_type":        state.get("query_type", ""),
        "retrieved_docs":    state.get("retrieved_docs", []),
        "specialist":        state.get("specialist", ""),
        "compliance_status": "",
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
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
