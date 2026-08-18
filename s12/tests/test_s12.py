"""
s12/tests/test_s12.py
---------------------
Tests for Session 12: Multi-Agent Architecture Part 2 (Compliance Agent).

Run with:
    pytest s12/tests/ -v

All tests mock the LLM and MCP tools -- no real Groq or MCP calls required.

Test groups:
  TestState                 -- ClinicalIQState has BOTH specialist and compliance_status
  TestComplianceLogic       -- _check_compliance_logic passes/fails on banned phrases
  TestCheckMedicalNode      -- check_medical node sets compliance_status correctly
  TestReviseResponseNode    -- revise_response calls LLM and updates state
  TestRouteComplianceFunc   -- route_compliance returns "revise" for FAIL, END for PASS
  TestComplianceAgentCreate -- create_compliance_agent returns valid compiled graph
  TestCallComplianceAgent   -- call_compliance_agent invokes _compliance_agent
  TestRouting               -- route_supervisor maps all 4 categories correctly (inherited)
  TestSupervisorGraph       -- graph has compliance node; routes through it for specialists
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from langgraph.graph import END

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "clinicaliq" or _k.startswith("clinicaliq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from clinicaliq.state import ClinicalIQState        # noqa: E402
import clinicaliq.nodes as _nodes                   # noqa: E402
from clinicaliq.nodes import (                      # noqa: E402
    _check_compliance_logic,
    call_compliance_agent,
    call_doctors_agent,
    call_services_agent,
    check_medical,
    classify,
    create_compliance_agent,
    create_doctors_agent,
    create_services_agent,
    decline,
    escalate,
    revise_response,
    route_compliance,
    route_supervisor,
)
from clinicaliq.agent import build_graph            # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(message: str = "test", qt: str = "SERVICES") -> ClinicalIQState:
    return {
        "customer_message":  message,
        "response":          "",
        "history":           [],
        "query_type":        qt,
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
    }


def _state_with_response(response: str, qt: str = "SERVICES") -> ClinicalIQState:
    return {**_base_state(qt=qt), "response": response}


# ---------------------------------------------------------------------------
# TestState
# ---------------------------------------------------------------------------

class TestState:
    def test_state_has_specialist_field(self):
        state: ClinicalIQState = {**_base_state(), "specialist": "doctors_agent"}
        assert "specialist" in state

    def test_state_has_compliance_status_field(self):
        state: ClinicalIQState = {**_base_state(), "compliance_status": "PASS"}
        assert "compliance_status" in state

    def test_compliance_status_annotation_exists(self):
        assert "compliance_status" in ClinicalIQState.__annotations__

    def test_specialist_annotation_exists(self):
        assert "specialist" in ClinicalIQState.__annotations__

    def test_compliance_status_accepts_pass(self):
        state: ClinicalIQState = {**_base_state(), "compliance_status": "PASS"}
        assert state["compliance_status"] == "PASS"

    def test_compliance_status_accepts_fail(self):
        state: ClinicalIQState = {**_base_state(), "compliance_status": "FAIL: banned phrase"}
        assert state["compliance_status"].startswith("FAIL")

    def test_compliance_status_accepts_revised(self):
        state: ClinicalIQState = {**_base_state(), "compliance_status": "REVISED"}
        assert state["compliance_status"] == "REVISED"


# ---------------------------------------------------------------------------
# TestComplianceLogic
# ---------------------------------------------------------------------------

class TestComplianceLogic:
    def test_clean_response_passes(self):
        ok, reason = _check_compliance_logic("Please consult a doctor for more details.")
        assert ok is True
        assert reason == "PASS"

    def test_banned_phrase_you_have_fails(self):
        ok, reason = _check_compliance_logic("Based on your symptoms, you have diabetes.")
        assert ok is False
        assert "you have" in reason

    def test_banned_phrase_prescribe_fails(self):
        ok, reason = _check_compliance_logic("I would prescribe metformin for this.")
        assert ok is False
        assert "prescribe" in reason

    def test_banned_phrase_diagnose_fails(self):
        ok, reason = _check_compliance_logic("I diagnose you with hypertension.")
        assert ok is False

    def test_banned_phrase_take_this_medication_fails(self):
        ok, reason = _check_compliance_logic("You should take this medication twice daily.")
        assert ok is False

    def test_banned_phrase_case_insensitive(self):
        ok, reason = _check_compliance_logic("YOU HAVE an infection.")
        assert ok is False

    def test_reason_contains_banned_phrase(self):
        ok, reason = _check_compliance_logic("Your diagnosis is anxiety disorder.")
        assert ok is False
        assert "your diagnosis" in reason.lower()

    def test_partial_match_in_sentence(self):
        ok, reason = _check_compliance_logic(
            "Please see a doctor. You have several options for treatment."
        )
        assert ok is False

    def test_your_condition_is_fails(self):
        ok, reason = _check_compliance_logic("Your condition is serious and needs attention.")
        assert ok is False

    def test_sign_of_fails(self):
        ok, reason = _check_compliance_logic("This is a sign of infection.")
        assert ok is False

    def test_empty_string_passes(self):
        ok, reason = _check_compliance_logic("")
        assert ok is True

    def test_clinic_info_passes(self):
        ok, reason = _check_compliance_logic(
            "Apollo Health Clinic is open Monday to Saturday, 8 AM to 8 PM."
        )
        assert ok is True


# ---------------------------------------------------------------------------
# TestCheckMedicalNode
# ---------------------------------------------------------------------------

class TestCheckMedicalNode:
    def test_clean_response_sets_pass(self):
        state = _state_with_response("Our cardiologist is available on Monday.")
        result = check_medical(state)
        assert result["compliance_status"] == "PASS"

    def test_banned_phrase_sets_fail(self):
        state = _state_with_response("You have a bacterial infection.")
        result = check_medical(state)
        assert result["compliance_status"].startswith("FAIL")

    def test_fail_contains_reason(self):
        state = _state_with_response("I diagnose you with arthritis.")
        result = check_medical(state)
        assert "banned phrase" in result["compliance_status"]

    def test_escalate_response_passes(self):
        state = _state_with_response(
            "Please speak with one of our nurses for clinical questions."
        )
        result = check_medical(state)
        assert result["compliance_status"] == "PASS"


# ---------------------------------------------------------------------------
# TestReviseResponseNode
# ---------------------------------------------------------------------------

class TestReviseResponseNode:
    def _fail_state(self, response: str) -> ClinicalIQState:
        return {**_state_with_response(response), "compliance_status": "FAIL: banned phrase: 'you have'"}

    def test_revise_calls_llm(self):
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Please consult a doctor.\n\nClinicalIQ | Apollo Health Clinic")
            result = revise_response(self._fail_state("You have diabetes."))
        mock_llm.invoke.assert_called_once()

    def test_revise_updates_response(self):
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Revised response.\n\nClinicalIQ | Apollo Health Clinic")
            result = revise_response(self._fail_state("You have diabetes."))
        assert result["response"] == "Revised response.\n\nClinicalIQ | Apollo Health Clinic"

    def test_revise_sets_status_revised(self):
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Safe response.")
            result = revise_response(self._fail_state("You have diabetes."))
        assert result["compliance_status"] == "REVISED"

    def test_revise_fallback_on_llm_error(self):
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.side_effect = Exception("API error")
            result = revise_response(self._fail_state("You have diabetes."))
        assert "ClinicalIQ" in result["response"]
        assert result["compliance_status"] == "REVISED"

    def test_revise_empty_llm_response_uses_fallback(self):
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="")
            result = revise_response(self._fail_state("You have diabetes."))
        assert result["response"] != ""


# ---------------------------------------------------------------------------
# TestRouteComplianceFunc
# ---------------------------------------------------------------------------

class TestRouteComplianceFunc:
    def test_fail_status_returns_revise(self):
        state = {**_base_state(), "compliance_status": "FAIL: banned phrase: 'you have'"}
        assert route_compliance(state) == "revise"

    def test_pass_status_returns_end(self):
        state = {**_base_state(), "compliance_status": "PASS"}
        assert route_compliance(state) == END

    def test_revised_status_returns_end(self):
        state = {**_base_state(), "compliance_status": "REVISED"}
        assert route_compliance(state) == END

    def test_empty_status_returns_end(self):
        state = {**_base_state(), "compliance_status": ""}
        assert route_compliance(state) == END

    def test_fail_prefix_check_is_exact(self):
        state = {**_base_state(), "compliance_status": "FAIL: something"}
        assert route_compliance(state) == "revise"


# ---------------------------------------------------------------------------
# TestComplianceAgentCreate
# ---------------------------------------------------------------------------

class TestComplianceAgentCreate:
    def test_factory_returns_compiled_graph(self):
        agent = create_compliance_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        a1 = create_compliance_agent()
        a2 = create_compliance_agent()
        assert a1 is not a2

    def test_agent_has_check_medical_node(self):
        agent = create_compliance_agent()
        assert "check_medical" in agent.get_graph().nodes

    def test_agent_has_revise_node(self):
        agent = create_compliance_agent()
        assert "revise" in agent.get_graph().nodes

    def test_agent_invocable_clean_response(self):
        agent = create_compliance_agent()
        result = agent.invoke(_state_with_response("Our cardiologist is available Monday."))
        assert result["compliance_status"] == "PASS"

    def test_agent_invocable_banned_phrase_triggers_revision(self):
        agent = create_compliance_agent()
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Please consult a doctor.\n\nClinicalIQ | Apollo Health Clinic")
            result = agent.invoke(_state_with_response("You have a serious infection."))
        assert result["compliance_status"] == "REVISED"

    def test_module_level_compliance_agent_exists(self):
        assert _nodes._compliance_agent is not None


# ---------------------------------------------------------------------------
# TestCallComplianceAgent
# ---------------------------------------------------------------------------

class TestCallComplianceAgent:
    def test_pass_through_for_clean_response(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response":          "Dr. Sharma is available Monday.",
                "compliance_status": "PASS",
            }
            state  = _state_with_response("Dr. Sharma is available Monday.", qt="DOCTORS")
            result = call_compliance_agent(state)
        assert result["compliance_status"] == "PASS"
        assert result["response"] == "Dr. Sharma is available Monday."

    def test_revised_response_for_banned_phrase(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response":          "Revised safe response.",
                "compliance_status": "REVISED",
            }
            state  = _state_with_response("You have diabetes.", qt="SERVICES")
            result = call_compliance_agent(state)
        assert result["compliance_status"] == "REVISED"
        assert result["response"] == "Revised safe response."

    def test_compliance_agent_called_with_reset_status(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Safe.", "compliance_status": "PASS"}
            state = {**_state_with_response("Safe response."), "compliance_status": "PASS"}
            call_compliance_agent(state)
        call_args = mock_agent.invoke.call_args[0][0]
        assert call_args["compliance_status"] == ""

    def test_missing_compliance_status_defaults_pass(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Safe."}
            result = call_compliance_agent(_state_with_response("Safe."))
        assert result["compliance_status"] == "PASS"


# ---------------------------------------------------------------------------
# TestRouting (inherited from S10)
# ---------------------------------------------------------------------------

class TestRouting:
    def _state(self, qt: str) -> ClinicalIQState:
        return {**_base_state(), "query_type": qt}

    def test_doctors_routes_to_doctors_agent(self):
        assert route_supervisor(self._state("DOCTORS")) == "call_doctors_agent"

    def test_services_routes_to_services_agent(self):
        assert route_supervisor(self._state("SERVICES")) == "call_services_agent"

    def test_complex_routes_to_escalate(self):
        assert route_supervisor(self._state("COMPLEX")) == "escalate"

    def test_oos_routes_to_decline(self):
        assert route_supervisor(self._state("OUT_OF_SCOPE")) == "decline"

    def test_unknown_defaults_to_services_agent(self):
        assert route_supervisor(self._state("UNKNOWN")) == "call_services_agent"


# ---------------------------------------------------------------------------
# TestSupervisorGraph
# ---------------------------------------------------------------------------

class TestSupervisorGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_has_classify_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "classify" in graph.get_graph().nodes

    def test_graph_has_call_compliance_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_compliance_agent" in graph.get_graph().nodes

    def test_graph_has_call_doctors_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_doctors_agent" in graph.get_graph().nodes

    def test_graph_has_call_services_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_services_agent" in graph.get_graph().nodes

    def test_doctors_query_routes_through_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_doctors_agent") as mock_doc_agent, \
             patch.object(_nodes, "_compliance_agent") as mock_comp_agent:
            mock_clf.invoke.return_value        = MagicMock(content="DOCTORS")
            mock_doc_agent.invoke.return_value  = {"response": "Dr. Sharma.", "history": []}
            mock_comp_agent.invoke.return_value = {"response": "Dr. Sharma.", "compliance_status": "PASS"}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Do you have a cardiologist?",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-doctors-compliance"}},
            )
        mock_comp_agent.invoke.assert_called_once()
        assert result["specialist"] == "doctors_agent"

    def test_services_query_routes_through_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_services_agent") as mock_svc_agent, \
             patch.object(_nodes, "_compliance_agent") as mock_comp_agent:
            mock_clf.invoke.return_value        = MagicMock(content="SERVICES")
            mock_svc_agent.invoke.return_value  = {"response": "Fee is Rs. 500.", "history": []}
            mock_comp_agent.invoke.return_value = {"response": "Fee is Rs. 500.", "compliance_status": "PASS"}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What is the consultation fee?",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-services-compliance"}},
            )
        mock_comp_agent.invoke.assert_called_once()
        assert result["specialist"] == "services_agent"

    def test_complex_query_bypasses_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_compliance_agent") as mock_comp_agent:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "I have chest pain.",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-complex-bypass"}},
            )
        mock_comp_agent.invoke.assert_not_called()
        assert result["specialist"] == "escalated"

    def test_oos_query_bypasses_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_compliance_agent") as mock_comp_agent:
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Write me a poem.",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-oos-bypass"}},
            )
        mock_comp_agent.invoke.assert_not_called()
        assert result["specialist"] == "declined"

    def test_compliance_pass_preserved_in_result(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_doctors_agent") as mock_doc_agent, \
             patch.object(_nodes, "_compliance_agent") as mock_comp_agent:
            mock_clf.invoke.return_value        = MagicMock(content="DOCTORS")
            mock_doc_agent.invoke.return_value  = {"response": "Dr. Sharma is available.", "history": []}
            mock_comp_agent.invoke.return_value = {"response": "Dr. Sharma is available.", "compliance_status": "PASS"}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Do you have a cardiologist?",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-compliance-pass"}},
            )
        assert result["compliance_status"] == "PASS"

    def test_non_compliant_response_revised(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_services_agent") as mock_svc_agent, \
             patch.object(_nodes, "_compliance_agent") as mock_comp_agent:
            mock_clf.invoke.return_value        = MagicMock(content="SERVICES")
            mock_svc_agent.invoke.return_value  = {"response": "You have an appointment.", "history": []}
            mock_comp_agent.invoke.return_value = {
                "response":          "Please speak with a nurse.\n\nClinicalIQ | Apollo Health Clinic",
                "compliance_status": "REVISED",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Do I have an appointment?",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-revised"}},
            )
        assert result["compliance_status"] == "REVISED"
        assert "ClinicalIQ" in result["response"]
