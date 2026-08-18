"""
s13/tests/test_s13.py
---------------------
Tests for Session 13: Streamlit UI (ClinicalIQ).

Run with:
    pytest s13/tests/ -v

All tests are pure Python — no Streamlit context needed.
The app helper functions are imported directly from app.py.

Test groups:
  TestBuildInputState    -- build_input_state() returns correct graph input dict
  TestGetThreadConfig    -- get_thread_config() returns correct LangGraph config
  TestComplianceBadge    -- compliance_badge() returns correct display text
  TestIsEscalated        -- is_escalated() detects escalated specialist
  TestFormatRouteLabel   -- format_route_label() formats routing info correctly
  TestAgentGraph         -- build_graph() compiles; S12 nodes are present
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "clinicaliq" or _k.startswith("clinicaliq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

_spec = importlib.util.spec_from_file_location("app", SOLUTION_DIR / "app.py")
_app  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app)

build_input_state = _app.build_input_state
get_thread_config = _app.get_thread_config
compliance_badge  = _app.compliance_badge
is_escalated      = _app.is_escalated
format_route_label = _app.format_route_label

from clinicaliq.agent import build_graph  # noqa: E402
import clinicaliq.nodes as _nodes         # noqa: E402


# ---------------------------------------------------------------------------
# TestBuildInputState
# ---------------------------------------------------------------------------

class TestBuildInputState:
    def test_has_customer_message(self):
        state = build_input_state("Which doctor handles knee problems?")
        assert state["customer_message"] == "Which doctor handles knee problems?"

    def test_has_empty_response(self):
        assert build_input_state("test")["response"] == ""

    def test_has_empty_specialist(self):
        assert build_input_state("test")["specialist"] == ""

    def test_has_empty_retrieved_docs(self):
        assert build_input_state("test")["retrieved_docs"] == []

    def test_has_empty_compliance_status(self):
        assert build_input_state("test")["compliance_status"] == ""

    def test_all_required_keys_present(self):
        state    = build_input_state("test")
        required = {"customer_message", "response", "specialist", "retrieved_docs", "compliance_status"}
        assert required.issubset(set(state.keys()))

    def test_different_messages_differ(self):
        s1 = build_input_state("doctors question")
        s2 = build_input_state("services question")
        assert s1["customer_message"] != s2["customer_message"]

    def test_empty_message_accepted(self):
        assert build_input_state("")["customer_message"] == ""


# ---------------------------------------------------------------------------
# TestGetThreadConfig
# ---------------------------------------------------------------------------

class TestGetThreadConfig:
    def test_returns_dict(self):
        assert isinstance(get_thread_config("abc"), dict)

    def test_has_configurable_key(self):
        assert "configurable" in get_thread_config("abc")

    def test_configurable_has_thread_id(self):
        assert get_thread_config("my-session")["configurable"]["thread_id"] == "my-session"

    def test_different_ids_differ(self):
        c1 = get_thread_config("s1")
        c2 = get_thread_config("s2")
        assert c1["configurable"]["thread_id"] != c2["configurable"]["thread_id"]


# ---------------------------------------------------------------------------
# TestComplianceBadge
# ---------------------------------------------------------------------------

class TestComplianceBadge:
    def test_pass_returns_checkmark(self):
        assert "✅" in compliance_badge("PASS")

    def test_revised_returns_warning(self):
        assert "⚠️" in compliance_badge("REVISED")

    def test_fail_returns_cross(self):
        assert "❌" in compliance_badge("FAIL: banned phrase")

    def test_empty_returns_empty_string(self):
        assert compliance_badge("") == ""

    def test_unknown_returns_empty_string(self):
        assert compliance_badge("UNKNOWN") == ""

    def test_compliant_text_in_pass(self):
        assert "Compliant" in compliance_badge("PASS")


# ---------------------------------------------------------------------------
# TestIsEscalated
# ---------------------------------------------------------------------------

class TestIsEscalated:
    def test_escalated_specialist_returns_true(self):
        assert is_escalated({"specialist": "escalated"}) is True

    def test_doctors_agent_returns_false(self):
        assert is_escalated({"specialist": "doctors_agent"}) is False

    def test_services_agent_returns_false(self):
        assert is_escalated({"specialist": "services_agent"}) is False

    def test_empty_specialist_returns_false(self):
        assert is_escalated({"specialist": ""}) is False

    def test_missing_specialist_returns_false(self):
        assert is_escalated({}) is False

    def test_declined_is_not_escalated(self):
        assert is_escalated({"specialist": "declined"}) is False


# ---------------------------------------------------------------------------
# TestFormatRouteLabel
# ---------------------------------------------------------------------------

class TestFormatRouteLabel:
    def test_includes_query_type(self):
        result = {"query_type": "DOCTORS", "specialist": "doctors_agent", "compliance_status": "PASS"}
        assert "DOCTORS" in format_route_label(result)

    def test_includes_specialist(self):
        result = {"query_type": "SERVICES", "specialist": "services_agent", "compliance_status": "PASS"}
        assert "services_agent" in format_route_label(result)

    def test_includes_badge_for_pass(self):
        result = {"query_type": "DOCTORS", "specialist": "doctors_agent", "compliance_status": "PASS"}
        assert "✅" in format_route_label(result)

    def test_no_badge_for_empty_status(self):
        result = {"query_type": "COMPLEX", "specialist": "escalated", "compliance_status": ""}
        label  = format_route_label(result)
        assert "✅" not in label and "⚠️" not in label

    def test_dash_for_missing_keys(self):
        assert "—" in format_route_label({})

    def test_revised_badge_shown(self):
        result = {"query_type": "SERVICES", "specialist": "services_agent", "compliance_status": "REVISED"}
        assert "⚠️" in format_route_label(result)


# ---------------------------------------------------------------------------
# TestAgentGraph
# ---------------------------------------------------------------------------

class TestAgentGraph:
    def test_build_graph_compiles(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert build_graph(checkpointer=MemorySaver()) is not None

    def test_graph_has_classify_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "classify" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_has_compliance_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "call_compliance_agent" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_has_doctors_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "call_doctors_agent" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_has_services_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "call_services_agent" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_invocable(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_doctors_agent") as mock_da, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="DOCTORS")
            mock_da.invoke.return_value  = {"response": "Dr. Sharma is available.", "history": []}
            mock_ca.invoke.return_value  = {"response": "Dr. Sharma is available.", "compliance_status": "PASS"}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                build_input_state("Do you have a cardiologist?"),
                config=get_thread_config("test-s13-clinicaliq"),
            )
        assert "response" in result
        assert "compliance_status" in result

    def test_compliance_pass_not_escalated(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_services_agent") as mock_sa, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="SERVICES")
            mock_sa.invoke.return_value  = {"response": "Fee is Rs. 600.", "history": []}
            mock_ca.invoke.return_value  = {"response": "Fee is Rs. 600.", "compliance_status": "PASS"}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                build_input_state("What is the consultation fee?"),
                config=get_thread_config("test-s13-fee"),
            )
        assert is_escalated(result) is False
