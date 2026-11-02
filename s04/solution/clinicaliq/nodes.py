"""
clinicaliq/nodes.py
-------------------
Node functions for the ClinicalIQ graph.

Session 4 adds ChromaDB RAG:
  retrieve_docs() -- new node: fetches relevant policy chunks from ChromaDB
  respond()       -- updated: builds context-augmented system prompt from retrieved chunks

classify(), route_query(), decline() carry forward from Session 3,
with one change: classify() now returns IN_SCOPE / OUT_OF_SCOPE (2-way) instead
of SIMPLE / COMPLEX / OUT_OF_SCOPE (3-way). The respond() node handles escalation
internally when no docs are retrieved.
"""
from langchain_chroma import Chroma

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, RETRIEVAL_SCORE_THRESHOLD, SYSTEM_PROMPT, VECTORSTORE_DIR,
)
from .state import ClinicalIQState
from .tools import classifier_llm, llm

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


def respond(state: ClinicalIQState) -> dict:
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    if not retrieved:
        new_history = history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": ESCALATE_RESPONSE},
        ]
        return {"response": ESCALATE_RESPONSE, "history": new_history}

    context_block  = "\n\n".join(f"<policy_chunk>\n{chunk}\n</policy_chunk>" for chunk in retrieved)
    system_content = (
        SYSTEM_PROMPT
        + f"\n\n<retrieved_documents>\n{context_block}\n</retrieved_documents>"
    )

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result        = llm.invoke(messages)
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
