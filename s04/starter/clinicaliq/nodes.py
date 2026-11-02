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
    """Load the ChromaDB vectorstore once. Subsequent calls are no-ops."""
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
    """Classify the patient's question as IN_SCOPE or OUT_OF_SCOPE."""
    msg = state["customer_message"].strip()

    if not msg or len(msg) < 3 or len(msg) > 500:
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}

    if any(phrase in msg.lower() for phrase in BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE", "retrieved_docs": []}

    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state["customer_message"]),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"IN_SCOPE", "OUT_OF_SCOPE"}:
            query_type = "IN_SCOPE"
    except Exception as e:
        print(f"[ClinicalIQ] Classification error: {e}")
        query_type = "IN_SCOPE"
    # Clear any retrieved_docs from the previous turn -- prevents stale chunks leaking across turns.
    return {"query_type": query_type, "retrieved_docs": []}


# ---------------------------------------------------------------------------
# TODO 2 of 4 -- implement retrieve_docs()
# ---------------------------------------------------------------------------
# retrieve_docs() is a new node that runs after classify() for IN_SCOPE queries.
# It fetches the most relevant policy chunks from ChromaDB and returns them
# as a list of strings, one per chunk.
#
# Steps:
#
#   1. Initialise the vectorstore (call _init_vectorstore() — it's a no-op if
#      the vectorstore is already loaded):
#        _init_vectorstore()
#        if vectorstore is None:
#            return {"retrieved_docs": []}
#
#   2. Call similarity_search_with_relevance_scores inside a try/except:
#        results = vectorstore.similarity_search_with_relevance_scores(
#            state["customer_message"], k=RETRIEVAL_K
#        )
#      This returns a list of (Document, score) tuples.
#      - Document.page_content : the raw chunk text
#      - Document.metadata     : dict, e.g. {"source": "appointment_guide.md"}
#      - score                 : cosine similarity 0-1 (higher = more relevant)
#
#   3. Filter by score threshold and format each passing chunk:
#        retrieved = []
#        for doc, score in results:
#            if score >= RETRIEVAL_SCORE_THRESHOLD:
#                retrieved.append(
#                    f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
#                )
#      Chunks below the threshold are noise -- skip them.
#
#   4. Return the list:
#        return {"retrieved_docs": retrieved}
#
# WHY filter by score?
# Without a threshold, every query retrieves something -- even "how are you?".
# The threshold (0.3) sits between noise (< 0.20) and real matches (> 0.40),
# so fragments and off-topic queries return no chunks and respond() escalates
# them gracefully rather than hallucinating an answer.
#
# ---------------------------------------------------------------------------

def retrieve_docs(state: ClinicalIQState) -> dict:
    """Retrieve relevant policy chunks from ChromaDB."""
    # TODO 2: initialise vectorstore, call similarity_search_with_relevance_scores,
    #         filter by RETRIEVAL_SCORE_THRESHOLD, return {"retrieved_docs": [...]}
    raise NotImplementedError("TODO 2: implement retrieve_docs() in clinicaliq/nodes.py")


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- update respond() to use retrieved_docs
# ---------------------------------------------------------------------------
# respond() now receives retrieved_docs from the state (populated by retrieve_docs()).
#
# Two changes from Session 3:
#
#   1. Early-exit if no docs were retrieved:
#        history   = state.get("history", [])
#        retrieved = state.get("retrieved_docs", [])
#        if not retrieved:
#            new_history = history + [
#                {"role": "user",      "content": state["customer_message"]},
#                {"role": "assistant", "content": ESCALATE_RESPONSE},
#            ]
#            return {"response": ESCALATE_RESPONSE, "history": new_history}
#      WHY: If no chunks passed the threshold, the knowledge base cannot support
#      an answer. Refer the patient to clinical staff without an LLM call.
#
#   2. Build a context-augmented system prompt using XML tags:
#        context_block  = "\n\n".join(
#            f"<policy_chunk>\n{chunk}\n</policy_chunk>" for chunk in retrieved
#        )
#        system_content = (
#            SYSTEM_PROMPT
#            + f"\n\n<retrieved_documents>\n{context_block}\n</retrieved_documents>"
#        )
#      Then use system_content (not SYSTEM_PROMPT) when building the messages list.
#
# The rest of respond() is identical to Session 3: replay history, append the
# current user message, call llm.invoke(), update history, return.
#
# ---------------------------------------------------------------------------

def respond(state: ClinicalIQState) -> dict:
    """Handle IN_SCOPE queries using retrieved policy chunks as context."""
    # TODO 3: read retrieved_docs from state, early-exit if empty,
    #         build XML-tagged context block, build messages list with history,
    #         call llm.invoke(), update and return history + response
    raise NotImplementedError("TODO 3: implement respond() with RAG context in clinicaliq/nodes.py")


def escalate(state: ClinicalIQState) -> dict:
    """Kept for Option A fallback — not wired into the graph under Option B."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: ClinicalIQState) -> dict:
    """Handle OUT_OF_SCOPE queries with a canned decline. No LLM call."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: ClinicalIQState) -> str:
    """Read query_type and return the next node name.

    Session 4: 2-way routing only — IN_SCOPE -> retrieve_docs, OUT_OF_SCOPE -> decline.
    Clinical-question escalation is handled inside respond() when no docs are retrieved.
    """
    qt = state.get("query_type", "IN_SCOPE")
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"
