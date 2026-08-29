"""
app.py
------
Streamlit chat UI for ClinicalIQ — Apollo Health Clinic's AI patient assistant.

Session 14b: LlamaGuard 3 8B — no llamaguard_score in state or guard badge.
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
        self._placeholder.markdown(self._text + "▌")
        if self._delay > 0:
            time.sleep(self._delay)

    @property
    def text(self) -> str:
        return self._text


def build_input_state(message: str) -> dict:
    return {
        "customer_message":  message,
        "response":          "",
        "specialist":        "",
        "retrieved_docs":    [],
        "compliance_status": "",
        "blocked_reason":    "",
    }


def get_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def compliance_badge(status: str) -> str:
    if status == "PASS":
        return "✅ Compliant"
    if status == "REVISED":
        return "⚠️ Revised"
    if status.startswith("FAIL"):
        return "❌ Violation"
    return ""


def guard_badge(blocked_reason: str) -> str:
    """Return a guard status badge. No score — LlamaGuard 3 8B returns safe/unsafe, not a float."""
    if blocked_reason == "pii":
        return "🔒 Blocked (PII)"
    if blocked_reason == "llamaguard":
        return "🤖 Blocked (LlamaGuard 3 8B)"
    if blocked_reason:
        return "🛡️ Blocked (injection — regex)"
    return ""


def needs_human_review(result: dict) -> bool:
    return result.get("compliance_status", "") == "REVISED"


def format_route_label(result: dict) -> str:
    blocked_r = result.get("blocked_reason", "")
    if blocked_r:
        return f"Guard: {guard_badge(blocked_r)}"

    sp    = result.get("specialist", "—")
    cs    = result.get("compliance_status", "")
    badge = compliance_badge(cs)
    label = f"Route: {sp}"
    if badge:
        label += f" | {badge}"
    return label


def is_escalated(result: dict) -> bool:
    return result.get("specialist", "") == "escalated"


def _init_session() -> None:
    if "graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        st.session_state.graph     = build_graph(checkpointer=MemorySaver())
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages  = []
        st.session_state.routes    = []


def _sidebar() -> None:
    with st.sidebar:
        st.header("🏥 ClinicalIQ")
        st.caption("Apollo Health Clinic, Bengaluru")
        st.divider()
        if st.button("🔄 New Conversation", use_container_width=True):
            for key in ["graph", "thread_id", "messages", "routes"]:
                st.session_state.pop(key, None)
            st.rerun()
        if "thread_id" in st.session_state:
            st.caption(f"Session: {st.session_state.thread_id[:8]}…")
        st.divider()
        st.subheader("Agents")
        st.markdown(
            "- **Guard** — Layer 1 (regex) + Layer 2 (LlamaGuard 3 8B)\n"
            "- **Supervisor** — classifies query\n"
            "- **Doctors Agent** — finds right specialist\n"
            "- **Services Agent** — fees, booking, prep\n"
            "- **Compliance Agent** — medical ethics check\n"
            "- **Escalate** → nurse / **Decline** → out-of-scope"
        )
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
        page_icon="🏥",
        layout="wide",
    )
    st.title("🏥 ClinicalIQ | Apollo Health Clinic")
    st.caption("AI patient guidance assistant — Session 14b: LlamaGuard 3 8B")

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

        if is_escalated(result):
            placeholder.warning(response)
        elif result.get("blocked_reason"):
            placeholder.error(response)
        else:
            placeholder.markdown(response)
        st.caption(route_label)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.routes.append(route_label)


if __name__ == "__main__":
    main()
