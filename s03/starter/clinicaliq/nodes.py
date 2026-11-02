"""
clinicaliq/nodes.py
-----------------
Node functions for the ClinicalIQ graph.

Session 3 adds three new functions:
  classify()    -- node: calls the classifier LLM, returns {"query_type": ...}
  route_query() -- routing function (NOT a node): reads query_type, returns
                   the next node name as a string
  escalate()    -- node: returns canned advisor referral, no LLM call
  decline()     -- node: returns canned decline message, no LLM call

respond() is carried forward from Session 2 unchanged.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE, SYSTEM_PROMPT
from .state import ClinicalIQState
from .tools import classifier_llm, llm

# Phrases that should never reach the LLM -- short-circuit before spending an API token.
BLOCKLIST = [
    "ignore all previous",
    "forget everything",
    "you are now",
    "disregard your system",
    "act as",
    "jailbreak",
]


# ---------------------------------------------------------------------------
# TODO 2 of 4 -- implement the LLM call inside classify()
# ---------------------------------------------------------------------------
# The input validation and BLOCKLIST pre-filter are already done above your
# code. You only need to implement the LLM classification step:
#
#   1. Build a messages list with two items:
#        messages = [
#            SystemMessage(content=CLASSIFY_SYSTEM),
#            HumanMessage(content=msg),
#        ]
#      Note: do NOT include history here. The classifier reads the current
#      question only. A previous COMPLEX exchange should not affect whether
#      "What is the Unlimited plan price?" is classified as SIMPLE.
#
#   2. Call classifier_llm (not llm) inside a try/except block:
#        result     = classifier_llm.invoke(messages)
#        query_type = result.content.strip().upper()
#
#   3. Validate the output -- LLMs can return unexpected text:
#        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
#            query_type = "SIMPLE"
#      Default to SIMPLE on unexpected output: respond() has the system
#      prompt rules as a second guard, so failing open is safer than
#      failing to escalate/decline.
#
#   4. In the except block, print the error and default to SIMPLE:
#        except Exception as e:
#            print(f"[ClinicalIQ] Classification error: {e}")
#            query_type = "SIMPLE"
#
#   5. Return the query_type in a dict:
#        return {"query_type": query_type}
#
# ---------------------------------------------------------------------------

def classify(state: ClinicalIQState) -> dict:
    """Classify the customer's question into SIMPLE, COMPLEX, or OUT_OF_SCOPE."""
    msg = state["customer_message"].strip()

    # Input validation -- catch bad input before spending an API token
    if not msg or len(msg) < 3 or len(msg) > 500:
        return {"query_type": "OUT_OF_SCOPE"}

    # Rules-based pre-filter -- catch obvious injection/jailbreak attempts
    if any(phrase in msg.lower() for phrase in BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE"}

    # TODO 2: build messages list, call classifier_llm inside try/except,
    #         validate output, return {"query_type": query_type}
    raise NotImplementedError("TODO 2: implement LLM classification in clinicaliq/nodes.py")


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- implement route_query()
# ---------------------------------------------------------------------------
# route_query() is a routing function, NOT a node.
# It is never added with builder.add_node(). It is passed as the second
# argument to builder.add_conditional_edges().
#
# LangGraph calls it after classify() runs. It receives the current state
# and must return a string that matches a key in the path map dict you pass
# to add_conditional_edges(). In our case the keys equal the node names:
#   "respond", "escalate", or "decline"
#
# Implementation:
#   1. Read query_type from state (default to "SIMPLE" if missing):
#        qt = state.get("query_type", "SIMPLE")
#   2. Return "escalate" if qt == "COMPLEX"
#   3. Return "decline"  if qt == "OUT_OF_SCOPE"
#   4. Return "respond"  otherwise (covers "SIMPLE" and any unexpected value)
#
# ---------------------------------------------------------------------------

def route_query(state: ClinicalIQState) -> str:
    """Read query_type and return the name of the next node."""
    # TODO 3: implement the three if branches and return the correct node name
    raise NotImplementedError("TODO 3: implement route_query() in clinicaliq/nodes.py")


def respond(state: ClinicalIQState) -> dict:
    """Handle SIMPLE queries. Call the LLM with full conversation history."""
    history  = state.get("history", [])
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
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
    """Handle COMPLEX queries with a canned advisor referral. No LLM call."""
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
