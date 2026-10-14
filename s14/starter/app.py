"""
app.py — Session 14 Starter
-----------------------------
TODO: Update the Streamlit UI to handle guard-blocked responses.

Changes from S13:
  1. Update build_input_state to include:
       "blocked_reason": ""
       "llamaguard_score": -1.0
  2. Add guard_badge(blocked_reason, llamaguard_score) helper
  3. Add needs_human_review(result) helper
  4. Update format_route_label to show guard badge when blocked_reason is set
  5. In main(), use placeholder.error(response) when result has a blocked_reason
"""
import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from clinicaliq.agent import build_graph  # noqa: E402
import clinicaliq.nodes as _nodes        # noqa: E402


class _StreamingState:
    def __init__(self, placeholder, token_delay: float = 0.0) -> None:
        self._placeholder = placeholder
        self._text = ""
        self._delay = token_delay

    def __call__(self, token: str) -> None:
        self._text += token
        self._placeholder.markdown(self._text + "\u25ae")
        if self._delay > 0:
            time.sleep(self._delay)

    @property
    def text(self) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_input_state(message: str) -> dict:
    """Return the initial state dict for graph.invoke()."""
    return {
        "customer_message":  message,
        "response":          "",
        "specialist":        "",
        "retrieved_docs":    [],
        "compliance_status": "",
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
    }


def get_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def compliance_badge(status: str) -> str:
    if status == "PASS":
        return "\u2705 Compliant"
    if status == "REVISED":
        return "\u26a0\ufe0f Revised"
    if status.startswith("FAIL"):
        return "\u274c Violation"
    return ""


def guard_badge(blocked_reason: str, llamaguard_score: float = -1.0) -> str:
    if blocked_reason == "pii":
        return "\U0001f512 Blocked (PII)"
    if blocked_reason == "llamaguard":
        score_str = f"{llamaguard_score:.4f}" if llamaguard_score >= 0 else "N/A"
        return f"\U0001f916 Blocked (jailbreak \u2014 Prompt Guard \u00b7 score {score_str})"
    if blocked_reason:
        return "\U0001f6e1\ufe0f Blocked (injection \u2014 regex)"
    return ""


def needs_human_review(result: dict) -> bool:
    return result.get("compliance_status") == "REVISED"


def format_route_label(result: dict) -> str:
    blocked_reason   = result.get("blocked_reason", "")
    llamaguard_score = result.get("llamaguard_score", -1.0)
    if blocked_reason:
        return guard_badge(blocked_reason, llamaguard_score)
    qt    = result.get("query_type", "\u2014")
    sp    = result.get("specialist", "\u2014")
    cs    = result.get("compliance_status", "")
    badge = compliance_badge(cs)
    label = f"Route: {qt} \u2192 {sp}"
    if badge:
        label += f" | {badge}"
    return label


def is_escalated(result: dict) -> bool:
    return result.get("specialist", "") == "escalated"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        st.session_state.graph     = build_graph(checkpointer=MemorySaver())
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages  = []
        st.session_state.routes    = []


def _sidebar() -> None:
    with st.sidebar:
        st.header("\U0001f3e5 ClinicalIQ")
        st.caption("Apollo Health Clinic, Bengaluru")
        st.divider()
        if st.button("\U0001f504 New Conversation", use_container_width=True):
            for key in ["graph", "thread_id", "messages", "routes"]:
                st.session_state.pop(key, None)
            st.rerun()
        if "thread_id" in st.session_state:
            st.caption(f"Session: {st.session_state.thread_id[:8]}\u2026")
        st.divider()
        st.subheader("Demo settings")
        st.session_state["token_delay"] = st.slider(
            "Token delay (ms)", min_value=0, max_value=100,
            value=st.session_state.get("token_delay", 0), step=5,
        )


def _render_history() -> None:
    messages = st.session_state.get("messages", [])
    routes   = st.session_state.get("routes",   [])
    assistant_idx = 0
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if assistant_idx < len(routes):
                st.caption(routes[assistant_idx])
            assistant_idx += 1


def main() -> None:
    st.set_page_config(
        page_title="ClinicalIQ | Apollo Health Clinic",
        page_icon="\U0001f3e5",
        layout="wide",
    )
    st.title("\U0001f3e5 ClinicalIQ | Apollo Health Clinic")
    st.caption("AI patient guidance assistant \u2014 Session 14: Security and Guardrails")

    _init_session()
    _sidebar()
    _render_history()

    prompt = st.chat_input("How can we help you today?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

        delay_ms = st.session_state.get("token_delay", 0)
        streamer  = _StreamingState(placeholder, token_delay=delay_ms / 1000)
        _nodes._stream_callback = streamer
        try:
            result = st.session_state.graph.invoke(
                build_input_state(prompt),
                config=get_thread_config(st.session_state.thread_id),
            )
        finally:
            _nodes._stream_callback = None

        response    = result["response"]
        route_label = format_route_label(result)

        if result.get("blocked_reason"):
            placeholder.error(response)
        elif is_escalated(result):
            placeholder.warning(response)
        else:
            placeholder.markdown(response)
        st.caption(route_label)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.routes.append(route_label)


if __name__ == "__main__":
    main()
