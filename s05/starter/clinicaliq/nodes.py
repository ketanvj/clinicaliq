"""
clinicaliq/nodes.py
-------------------
Node functions for the ClinicalIQ graph.

Session 5 adds tool calling to respond():
  - The LLM now calls query_doctors() and query_services() when it needs live data
  - RAG (retrieve_docs) and tools are two independent data sources; either can
    satisfy the query. If RAG chunks are present they are injected; the LLM then
    decides whether to also call a tool for live doctor or service info.

Changes from Session 4:
  - respond():  uses llm_with_tools instead of llm; adds tool execution while-loop
                (TODO 4 below); removes early ESCALATE_RESPONSE return when
                retrieved_docs is empty

Carries forward from Session 4:
  - classify(), retrieve_docs(), escalate(), decline(), route_query(): unchanged
  - _init_vectorstore(): unchanged
"""
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, RETRIEVAL_SCORE_THRESHOLD, SYSTEM_PROMPT, VECTORSTORE_DIR,
)
from .state import ClinicalIQState
from .tools import classifier_llm, llm_with_tools, _run_tool

vectorstore = None  # shared across calls; initialised once by _init_vectorstore()

BLOCKLIST = [
    "ignore all previous",
    "forget everything",
    "you are now",
    "disregard your system",
    "act as",
    "jailbreak",
]


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
        count = vectorstore._collection.count()
        if count == 0:
            print(f"[ClinicalIQ] Vectorstore opened but is EMPTY (0 chunks).")
            print(f"  Run 'python data/ingest.py' from the clinicaliq/ root to populate it.")
        else:
            print(f"[ClinicalIQ] Vectorstore ready — {count} chunks loaded.")
    except Exception as e:
        print(f"[ClinicalIQ] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


def classify(state: ClinicalIQState) -> dict:
    msg = state["customer_message"].strip()

    if not msg or len(msg) < 3 or len(msg) > 500:
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}

    if any(phrase in msg.lower() for phrase in BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}

    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=msg),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"IN_SCOPE", "OUT_OF_SCOPE"}:
            query_type = "IN_SCOPE"
    except Exception as e:
        print(f"[ClinicalIQ] Classification error: {e}")
        query_type = "IN_SCOPE"
    return {"query_type": query_type, "retrieved_docs": []}


def retrieve_docs(state: ClinicalIQState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        results   = vectorstore.similarity_search_with_relevance_scores(
            state["customer_message"], k=RETRIEVAL_K
        )
        if not results:
            print(f"[ClinicalIQ] No chunks returned — vectorstore may be empty. Run 'python data/ingest.py'.")
            return {"retrieved_docs": []}
        retrieved = []
        for doc, score in results:
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                retrieved.append(
                    f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                )
            else:
                print(
                    f"[ClinicalIQ] Chunk skipped (score {score:.2f} < {RETRIEVAL_SCORE_THRESHOLD}): "
                    f"{doc.metadata.get('source', 'unknown')}"
                )
        print(f"[ClinicalIQ] Retrieval: {len(results)} candidates, {len(retrieved)} passed threshold ({RETRIEVAL_SCORE_THRESHOLD}).")
    except Exception as e:
        print(f"[ClinicalIQ] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}


# ---------------------------------------------------------------------------
# TODO 4 of 4 -- add tool calling to respond()
# ---------------------------------------------------------------------------
# respond() now uses llm_with_tools instead of llm so the LLM can call
# query_doctors() and query_services() to fetch live data from the database.
#
# The function signature and the history / RAG context setup are the same as
# Session 4.  The only change is what happens AFTER llm_with_tools.invoke():
#
#   Step 1 -- invoke llm_with_tools (not llm):
#        result = llm_with_tools.invoke(messages)
#
#   Step 2 -- log any invalid tool calls (bad JSON, unknown name):
#        if result.invalid_tool_calls:
#            for itc in result.invalid_tool_calls:
#                print(f"[ClinicalIQ] Invalid tool call ignored: {itc.get('name', 'unknown')} — {itc.get('error', 'parse error')}")
#
#   Step 3 -- execute tools in a while loop (not an if):
#        max_tool_rounds = 5
#        tool_rounds     = 0
#        while result.tool_calls and tool_rounds < max_tool_rounds:
#            messages.append(result)
#            for tc in result.tool_calls:
#                tool_output = _run_tool(tc["name"], tc["args"])
#                print(f"[ClinicalIQ] Tool: {tc['name']}({tc['args']}) -> {str(tool_output)[:80]}")
#                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
#            tool_rounds += 1
#            result = llm_with_tools.invoke(messages)
#
#   Step 4 -- extract the final text response:
#        response_text = result.content or ""
#        if not response_text:
#            response_text = "I am temporarily unavailable. Please try again in a moment."
#
# WHY a while loop (not an if)?
#   The LLM may call tools in multiple rounds: first query_doctors, then (in the
#   next LLM call) query_services. Using `if` would switch to the plain `llm` on
#   the second call — Groq rejects that with "Tool choice is none, but model
#   called a tool". The while loop keeps using llm_with_tools until the LLM
#   stops requesting tools and returns a plain text response.
#
# About result.tool_calls (defined by LangChain's ToolCall TypedDict):
#   tc["name"] : str  -- tool name,  e.g. "query_doctors"
#   tc["args"] : dict -- arguments,  e.g. {"department": "Cardiology"}
#   tc["id"]   : str  -- unique ID,  e.g. "call_abc123"
#   tc["type"] : str  -- always "tool_call" in this list
# ---------------------------------------------------------------------------

def respond(state: ClinicalIQState) -> dict:
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    if retrieved:
        context_block  = "\n\n".join(f"<policy_chunk>\n{chunk}\n</policy_chunk>" for chunk in retrieved)
        system_content = (
            SYSTEM_PROMPT
            + f"\n\n<retrieved_documents>\n{context_block}\n</retrieved_documents>"
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
        # TODO 4: replace the single llm.invoke() call below with llm_with_tools.invoke()
        #         then add the tool execution while-loop (see instructions above).
        #         Keep the try/except, response_text extraction, and history update.
        raise NotImplementedError("TODO 4: add tool calling loop in respond() in clinicaliq/nodes.py")

    except Exception as e:
        print(f"[ClinicalIQ] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


def escalate(state: ClinicalIQState) -> dict:
    """Kept for Option A fallback — not wired into the graph under Option B."""
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
    qt = state.get("query_type", "IN_SCOPE")
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"
