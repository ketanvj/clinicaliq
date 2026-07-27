from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, SYSTEM_PROMPT, VECTORSTORE_DIR,
)
from .state import ClinicalIQState
from .tools import classifier_llm, llm

vectorstore = None


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
            print("  Run 'python data/ingest.py' from the project root to populate it.")
        else:
            print(f"[ClinicalIQ] Vectorstore ready — {count} chunks loaded.")
    except Exception as e:
        print(f"[ClinicalIQ] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


def classify(state: ClinicalIQState) -> dict:
    # ── Option B (active) ───────────────────────────────────────────────────
    # Classifier answers one question only: is this about Apollo Health Clinic?
    # respond() decides whether to answer from docs or escalate.
    # Valid outputs: IN_SCOPE | OUT_OF_SCOPE
    #
    # ── Option A fallback ───────────────────────────────────────────────────
    # To revert, update CLASSIFY_SYSTEM in config.py (uncomment Option A)
    # and change valid_types below to {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}.
    # Also restore the escalate branch in route_query() and agent.py.
    valid_types = {"IN_SCOPE", "OUT_OF_SCOPE"}
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state["customer_message"]),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in valid_types:
            query_type = "IN_SCOPE"
    except Exception as e:
        print(f"[ClinicalIQ] Classification error: {e}")
        query_type = "IN_SCOPE"
    # Clear any retrieved_docs from the previous turn so stale results never leak.
    return {"query_type": query_type, "retrieved_docs": []}


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
        if not docs:
            print(f"[ClinicalIQ] No chunks returned — vectorstore may be empty. Run 'python data/ingest.py'.")
        else:
            print(f"[ClinicalIQ] Retrieval: {len(docs)} chunks returned.")
    except Exception as e:
        print(f"[ClinicalIQ] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}


def respond(state: ClinicalIQState) -> dict:
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    # ── Option B (active) ───────────────────────────────────────────────────
    # If no docs were retrieved the knowledge base cannot support an answer.
    # Escalate directly without an LLM call — fast and deterministic.
    if not retrieved:
        new_history = history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": ESCALATE_RESPONSE},
        ]
        return {"response": ESCALATE_RESPONSE, "history": new_history}
    #
    # ── Option A fallback ───────────────────────────────────────────────────
    # To restore: remove the early-return block above (keep the rest).
    # ────────────────────────────────────────────────────────────────────────

    context_block  = "\n\n---\n\n".join(retrieved)
    system_content = (
        SYSTEM_PROMPT
        + "\n\nThe following sections from Apollo Health Clinic's policy documents are relevant "
        "to the patient's question. Use this information in your answer:\n\n"
        + context_block
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


# ── Option A fallback node ─────────────────────────────────────────────────────
# escalate() is no longer wired into the graph under Option B.
# It is kept here so reverting to Option A only requires changes to agent.py.
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
    # ── Option B (active) ───────────────────────────────────────────────────
    # Two routes: OUT_OF_SCOPE → decline, everything else → retrieve_docs.
    qt = state.get("query_type", "IN_SCOPE")
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"
    #
    # ── Option A fallback ───────────────────────────────────────────────────
    # Three routes: SIMPLE → retrieve_docs, COMPLEX → escalate, OUT_OF_SCOPE → decline.
    # qt = state.get("query_type", "SIMPLE")
    # if qt == "COMPLEX":
    #     return "escalate"
    # if qt == "OUT_OF_SCOPE":
    #     return "decline"
    # return "retrieve_docs"
