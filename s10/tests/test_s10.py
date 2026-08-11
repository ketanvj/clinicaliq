"""
s10/tests/test_s10.py
---------------------
Tests for Session 10: Multi-Agent Architecture Part 1.

Run with:
    pytest s10/tests/ -v

All tests mock the LLM and MCP tools -- no real Groq or MCP calls required.

Test groups:
  TestState              -- ClinicalIQState has specialist field; no compliance_status
  TestClassifyNode       -- returns DOCTORS/SERVICES/COMPLEX/OUT_OF_SCOPE; safe default
  TestDoctorsAgent       -- factory returns compiled graph; invocable; updates history
  TestServicesAgent      -- factory returns compiled graph; invocable; updates history
  TestSupervisorNodes    -- call_doctors_agent/call_services_agent return correct state
  TestRouting            -- route_supervisor maps all 4 categories correctly
  TestSupervisorGraph    -- graph compiles; routes correctly; specialist field present
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "clinicaliq" or _k.startswith("clinicaliq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from clinicaliq.state import ClinicalIQState    # noqa: E402
import clinicaliq.nodes as _nodes               # noqa: E402
from clinicaliq.nodes import (                  # noqa: E402
    call_doctors_agent, call_services_agent, classify,
    create_doctors_agent, create_services_agent,
    decline, escalate, route_supervisor,
)
from clinicaliq.agent import build_graph        # noqa: E402


# ---------------------------------------------------------------------------
# TestState
# ---------------------------------------------------------------------------

class TestState:
    def test_state_has_specialist_field(self):
        state: ClinicalIQState = {
            "customer_message": "test",
            "response":         "",
            "history":          [],
            "query_type":       "SERVICES",
            "retrieved_docs":   [],
            "specialist":       "",
        }
        assert "specialist" in state

    def test_specialist_accepts_doctors_agent(self):
        state: ClinicalIQState = {
            "customer_message": "test", "response": "", "history": [],
            "query_type": "DOCTORS", "retrieved_docs": [], "specialist": "doctors_agent",
        }
        assert state["specialist"] == "doctors_agent"

    def test_specialist_accepts_services_agent(self):
        state: ClinicalIQState = {
            "customer_message": "test", "response": "", "history": [],
            "query_type": "SERVICES", "retrieved_docs": [], "specialist": "services_agent",
        }
        assert state["specialist"] == "services_agent"

    def test_state_has_no_compliance_status(self):
        assert "compliance_status" not in ClinicalIQState.__annotations__


# ---------------------------------------------------------------------------
# TestClassifyNode
# ---------------------------------------------------------------------------

class TestClassifyNode:
    def _state(self, message: str = "test") -> ClinicalIQState:
        return {
            "customer_message": message, "response": "", "history": [],
            "query_type": "", "retrieved_docs": [], "specialist": "",
        }

    def test_doctors_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="DOCTORS")
            result = classify(self._state("Which doctor handles knee problems?"))
        assert result["query_type"] == "DOCTORS"

    def test_services_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="SERVICES")
            result = classify(self._state("What is the consultation fee?"))
        assert result["query_type"] == "SERVICES"

    def test_complex_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="COMPLEX")
            result = classify(self._state("I have chest pain, what should I do?"))
        assert result["query_type"] == "COMPLEX"

    def test_oos_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            result = classify(self._state("Write me a poem"))
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_invalid_response_defaults_to_services(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="BANANA")
            result = classify(self._state("test"))
        assert result["query_type"] == "SERVICES"

    def test_classify_error_defaults_to_services(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.side_effect = Exception("API error")
            result = classify(self._state("test"))
        assert result["query_type"] == "SERVICES"

    def test_classify_strips_whitespace(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="  DOCTORS  ")
            result = classify(self._state("test"))
        assert result["query_type"] == "DOCTORS"


# ---------------------------------------------------------------------------
# TestDoctorsAgent
# ---------------------------------------------------------------------------

class TestDoctorsAgent:
    def test_factory_returns_compiled_graph(self):
        agent = create_doctors_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        a1 = create_doctors_agent()
        a2 = create_doctors_agent()
        assert a1 is not a2

    def test_agent_has_respond_node(self):
        agent = create_doctors_agent()
        assert "respond" in agent.get_graph().nodes

    def test_agent_is_invocable(self):
        agent = create_doctors_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Dr. Smith is available.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "Do you have a cardiologist?",
                "history": [], "response": "",
                "query_type": "DOCTORS", "retrieved_docs": [], "specialist": "",
            })
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_agent_updates_history(self):
        agent = create_doctors_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Answer.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "test", "history": [],
                "response": "", "query_type": "DOCTORS",
                "retrieved_docs": [], "specialist": "",
            })
        assert len(result.get("history", [])) == 2


# ---------------------------------------------------------------------------
# TestServicesAgent
# ---------------------------------------------------------------------------

class TestServicesAgent:
    def test_factory_returns_compiled_graph(self):
        agent = create_services_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        a1 = create_services_agent()
        a2 = create_services_agent()
        assert a1 is not a2

    def test_agent_has_respond_node(self):
        agent = create_services_agent()
        assert "respond" in agent.get_graph().nodes

    def test_agent_is_invocable(self):
        agent = create_services_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Consultation fee is Rs. 500.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "What is the consultation fee?",
                "history": [], "response": "",
                "query_type": "SERVICES", "retrieved_docs": [], "specialist": "",
            })
        assert "response" in result

    def test_agent_updates_history(self):
        agent = create_services_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Answer.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "test", "history": [],
                "response": "", "query_type": "SERVICES",
                "retrieved_docs": [], "specialist": "",
            })
        assert len(result.get("history", [])) == 2


# ---------------------------------------------------------------------------
# TestSupervisorNodes
# ---------------------------------------------------------------------------

class TestSupervisorNodes:
    def _state(self, message: str = "test", qt: str = "DOCTORS") -> ClinicalIQState:
        return {
            "customer_message": message, "response": "", "history": [],
            "query_type": qt, "retrieved_docs": [], "specialist": "",
        }

    def test_call_doctors_agent_sets_specialist(self):
        with patch.object(_nodes, "_doctors_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Dr. Sharma is our cardiologist.", "history": []
            }
            result = call_doctors_agent(self._state())
        assert result["specialist"] == "doctors_agent"

    def test_call_doctors_agent_returns_response(self):
        with patch.object(_nodes, "_doctors_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Dr. Sharma is our cardiologist.", "history": []
            }
            result = call_doctors_agent(self._state())
        assert result["response"] == "Dr. Sharma is our cardiologist."

    def test_call_services_agent_sets_specialist(self):
        with patch.object(_nodes, "_services_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Consultation fee is Rs. 500.", "history": []
            }
            result = call_services_agent(self._state(qt="SERVICES"))
        assert result["specialist"] == "services_agent"

    def test_call_services_agent_returns_response(self):
        with patch.object(_nodes, "_services_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Consultation fee is Rs. 500.", "history": []
            }
            result = call_services_agent(self._state(qt="SERVICES"))
        assert result["response"] == "Consultation fee is Rs. 500."

    def test_escalate_sets_specialist(self):
        result = escalate(self._state())
        assert result["specialist"] == "escalated"

    def test_escalate_response_mentions_nurse(self):
        result = escalate(self._state())
        assert "nurse" in result["response"]

    def test_decline_sets_specialist(self):
        result = decline(self._state())
        assert result["specialist"] == "declined"


# ---------------------------------------------------------------------------
# TestRouting
# ---------------------------------------------------------------------------

class TestRouting:
    def _state(self, qt: str) -> ClinicalIQState:
        return {
            "customer_message": "test", "response": "", "history": [],
            "query_type": qt, "retrieved_docs": [], "specialist": "",
        }

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

    def test_graph_has_call_doctors_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_doctors_agent" in graph.get_graph().nodes

    def test_graph_has_call_services_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_services_agent" in graph.get_graph().nodes

    def test_doctors_query_routes_to_doctors_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_doctors_agent") as mock_doc_agent:
            mock_clf.invoke.return_value = MagicMock(content="DOCTORS")
            mock_doc_agent.invoke.return_value = {
                "response": "Dr. Sharma available.", "history": []
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Do you have a cardiologist?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-doctors"}},
            )
        assert result["specialist"] == "doctors_agent"

    def test_services_query_routes_to_services_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_services_agent") as mock_svc_agent:
            mock_clf.invoke.return_value = MagicMock(content="SERVICES")
            mock_svc_agent.invoke.return_value = {
                "response": "Fee is Rs. 500.", "history": []
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What is the consultation fee?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-services"}},
            )
        assert result["specialist"] == "services_agent"

    def test_complex_query_escalates(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "I have chest pain.",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-complex"}},
            )
        assert result["specialist"] == "escalated"
        assert "nurse" in result["response"]

    def test_oos_query_declines(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Write me a poem.",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-oos"}},
            )
        assert result["specialist"] == "declined"

    def test_graph_result_has_specialist_field(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "test", "response": "",
                 "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-field"}},
            )
        assert "specialist" in result
