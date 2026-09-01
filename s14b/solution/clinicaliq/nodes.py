"""
clinicaliq/nodes.py
-------------------
Graph nodes for ClinicalIQ multi-agent architecture (Session 14b).

Session 14b upgrades the input guard from Llama Prompt Guard 2 (S14) to
LlamaGuard 3 8B — Meta's full 13-category content safety model.

The two-layer guard architecture is unchanged from S14:
  Layer 1 (regex, deterministic): PII + injection patterns
  Layer 2 (semantic): LlamaGuard 3 8B via Ollama or Together AI

Key difference from S14: _llamaguard_safe() returns bool (not tuple[bool, float]).
LlamaGuard 3 8B outputs "safe" or "unsafe\\nS<n>" — not a probability score.

S6 and S9 exclusion (ClinicalIQ-specific):
  LlamaGuard S6 (Specialized Advice) catches "What medicine should I take?"
  LlamaGuard S9 (Self-Harm) catches mentions of self-harm or suicide.
  Both are EXCLUDED from blocking because in a medical context these queries
  should reach a human nurse via the COMPLEX → escalate path, not be
  wall-blocked with a generic "I can only assist with clinic services" message.
  Blocking vulnerable patients (S9) is actively harmful.
"""
import base64
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
# S14b: Input Guard
# ---------------------------------------------------------------------------

def _try_decode(text: str) -> str:
    """Decode Base64 or hex-encoded text before guard checks.
    Catches obfuscation attacks like base64("ignore previous instructions").
    Returns decoded text if decoding succeeds and changes the input; original text otherwise.
    """
    stripped = text.strip()
    # Base64
    try:
        padding = (4 - len(stripped) % 4) % 4
        decoded = base64.b64decode(stripped + "=" * padding).decode("utf-8", errors="strict")
        if decoded != stripped and len(decoded) >= 8 and decoded.isprintable():
            return decoded
    except Exception:
        pass
    # Hex
    try:
        hex_clean = stripped.replace(" ", "")
        if len(hex_clean) >= 16 and all(c in "0123456789abcdefABCDEF" for c in hex_clean):
            decoded = bytes.fromhex(hex_clean).decode("utf-8", errors="strict")
            if decoded.isprintable():
                return decoded
    except Exception:
        pass
    return text


def _llamaguard_safe(message: str) -> bool:
    """Call LlamaGuard 3 8B and return True if the message is safe.

    LlamaGuard 3 8B response format:
      "safe"            — message passes all 13 safety checks
      "unsafe\\nS6"     — flagged for one category
      "unsafe\\nS6,S9"  — flagged for multiple categories

    S6 + S9 exclusion — ClinicalIQ intentionally handles both:
      S6 (Specialized Advice) catches "What medicine should I take?"
        → classified COMPLEX → escalated to nurse. Better than a wall block.
      S9 (Self-Harm) catches self-harm and suicide mentions.
        → In a medical context, these patients need a nurse, not a wall block.
        Blocking a vulnerable patient reaching out is actively harmful.

    We block on everything else: S1 (Violence), S2 (Non-Violent Crimes),
    S7 (Privacy), S8 (Hate), S13 (System Prompt Issues / jailbreaks), etc.

    Fail-open: if LlamaGuard is unreachable, log and let the message through.
    """
    try:
        result  = llamaguard_llm.invoke([HumanMessage(content=message)])
        verdict = result.content.strip().lower()

        if verdict.startswith("safe"):
            print(f"[ClinicalIQ] LlamaGuard: {verdict!r} → safe")
            return True

        categories: set[str] = set()
        if "\n" in verdict:
            raw_cats   = verdict.split("\n", 1)[1]
            categories = {c.strip() for c in raw_cats.split(",")}

        # S6 (Specialized Advice) and S9 (Self-Harm) are handled by COMPLEX routing
        excluded     = {"s6", "s9"}
        non_excluded = categories - excluded
        safe = len(non_excluded) == 0
        print(
            f"[ClinicalIQ] LlamaGuard: {verdict!r} → "
            f"{'safe (S6/S9 handled by router)' if safe else 'UNSAFE'}"
        )
        return safe
    except Exception as e:
        print(f"[ClinicalIQ] LlamaGuard unavailable — defaulting to safe: {e}")
        return True


@traceable(name="input_guard")
def guard(state: ClinicalIQState) -> dict:
    """Inspect customer_message for PII, injection patterns, and unsafe content.

    Returns {"blocked_reason": ""} for a clean message, or
    {"blocked_reason": "pii"|"injection"|"llamaguard"} when blocked.
    """
    raw = state["customer_message"]
    msg = unicodedata.normalize("NFKD", _INVISIBLE_UNICODE_RE.sub("", raw))

    decoded = _try_decode(msg)
    if decoded != msg:
        print(f"[ClinicalIQ] Guard: obfuscated input decoded ({len(msg)}→{len(decoded)} chars)")
        msg = decoded

    for rx in _pii_compiled:
        if rx.search(msg):
            print("[ClinicalIQ] Guard: PII detected — blocked")
            return {"blocked_reason": "pii"}

    for rx in _injection_compiled:
        if rx.search(msg):
            print("[ClinicalIQ] Guard: injection (regex) detected — blocked")
            return {"blocked_reason": "injection"}

    if not _llamaguard_safe(msg):
        print("[ClinicalIQ] Guard: LlamaGuard flagged message — blocked")
        return {"blocked_reason": "llamaguard"}

    return {"blocked_reason": ""}


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
# Specialist agents (unchanged from S14, state dict without llamaguard_score)
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
