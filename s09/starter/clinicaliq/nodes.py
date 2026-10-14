"""
clinicaliq/nodes.py
-------------------
STARTER FILE -- implement the two TODO sections below.

Goal
  Add a compliance filter to the ClinicalIQ agent that detects banned medical-
  advice phrases in LLM responses and replaces them with a safe fallback.

What is already done for you
  - _BANNED_PATTERN compiled from CLINICALIQ_BANNED_PHRASES (fast regex scan)
  - _normalize_for_check() for Unicode normalisation before scanning
  - All existing graph nodes (classify, retrieve_docs, respond, escalate, decline)

Your task
  TODO 1: Implement _check_compliance() using _BANNED_PATTERN.search()
  TODO 2: Implement check_compliance() node that calls _check_compliance()
          and replaces non-compliant responses with SAFE_COMPLIANCE_RESPONSE

Run when done
  python -m clinicaliq.agent   (from inside s09/starter/)
"""
import re
import unicodedata

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

from .config import (
    CLASSIFY_SYSTEM,
    CLINICALIQ_BANNED_PHRASES,
    DECLINE_RESPONSE,
    EMBED_MODEL,
    ESCALATE_RESPONSE,
    RETRIEVAL_K,
    SAFE_COMPLIANCE_RESPONSE,
    SYSTEM_PROMPT,
    VECTORSTORE_DIR,
)
from .state import ClinicalIQState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools

vectorstore = None

_BANNED_PATTERN: re.Pattern = re.compile(
    "|".join(re.escape(p) for p in CLINICALIQ_BANNED_PHRASES),
    re.IGNORECASE,
)


def _normalize_for_check(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in "‐‑‒–—―−":
        text = text.replace(ch, "-")
    return text.lower()


# ---------------------------------------------------------------------------
# TODO 1 of 2 -- Implement _check_compliance()
# ---------------------------------------------------------------------------
# Steps:
#   1. normalized = _normalize_for_check(draft)
#   2. match = _BANNED_PATTERN.search(normalized)
#      if match: return False, f"banned phrase: '{match.group()}'"
#   3. return True, "PASS"
# ---------------------------------------------------------------------------
@traceable(name="medical_compliance_check")
def _check_compliance(draft: str) -> tuple:
    normalized = _normalize_for_check(draft)
    match = _BANNED_PATTERN.search(normalized)
    if match:
        return False, f"banned phrase: '{match.group()}'"
    return True, "PASS"


# ---------------------------------------------------------------------------
# TODO 2 of 2 -- Implement check_compliance() node
# ---------------------------------------------------------------------------
# Steps:
#   1. Call _check_compliance(state["response"])
#   2. If it fails (passed is False):
#        print(f"[ClinicalIQ] Compliance FAIL: {reason}")
#        return {"response": SAFE_COMPLIANCE_RESPONSE,
#                "compliance_status": f"FAIL: {reason}"}
#   3. If it passes:
#        print("[ClinicalIQ] Compliance PASS")
#        return {"compliance_status": "PASS"}
# ---------------------------------------------------------------------------
def check_compliance(state: ClinicalIQState) -> dict:
    passed, reason = _check_compliance(state["response"])
    if not passed:
        print(f"[ClinicalIQ] Compliance FAIL: {reason}")
        return {"response": SAFE_COMPLIANCE_RESPONSE, "compliance_status": f"FAIL: {reason}"}
    print("[ClinicalIQ] Compliance PASS")
    return {"compliance_status": "PASS"}


def _init_vectorstore() -> None:
    global vectorstore
    if vectorstore is not None:
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
    except Exception as e:
        print(f"[ClinicalIQ] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


def classify(state: ClinicalIQState) -> dict:
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state["customer_message"]),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "SIMPLE"
    except Exception as e:
        print(f"[ClinicalIQ] Classification error: {e}")
        query_type = "SIMPLE"
    return {"query_type": query_type}


def retrieve_docs(state: ClinicalIQState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        docs      = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
        retrieved = [
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ]
    except Exception as e:
        print(f"[ClinicalIQ] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}


def respond(state: ClinicalIQState) -> dict:
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    if retrieved:
        context_block  = "\n\n---\n\n".join(retrieved)
        system_content = (
            SYSTEM_PROMPT
            + "\n\nThe following sections from Apollo Health Clinic's documents are relevant "
              "to the patient's question. Use this information in your answer:\n\n"
            + context_block
        )
    else:
        system_content = SYSTEM_PROMPT

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm_with_tools.invoke(messages)

        if result.tool_calls:
            messages.append(result)
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(
                    f"[ClinicalIQ] MCP tool: {tc['name']}({tc['args']}) "
                    f"-> {str(tool_output)[:80]}"
                )
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
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


def escalate(state: ClinicalIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: ClinicalIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: ClinicalIQState) -> str:
    qt = state.get("query_type", "SIMPLE")
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"
