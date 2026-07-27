"""
s05/tests/test_s05.py
---------------------
Tests for Session 5: SQLite tool calls.

Run with:
    pytest s05/tests/ -v

Test groups:
  TestClinicalIQState     -- state TypedDict has all five fields (unchanged from S04)
  TestQueryDoctorsTool    -- query_doctors() SQL correctness, filtering, output format
  TestQueryServicesTool   -- query_services() SQL correctness, filtering, output format
  TestToolSQLSafety       -- SQL injection protection and parameterised-query enforcement
  TestToolsBinding        -- llm_with_tools exists; tools are @tool decorated; prompt updated
  TestRunToolDispatch     -- _run_tool dispatches correctly; handles unknown names
  TestRespondWithTools    -- respond() calls llm_with_tools; executes tool calls; calls llm again
  TestGraphRouting        -- SIMPLE goes through retrieve_docs -> respond; COMPLEX/OOS skip tools
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "clinicaliq" or _k.startswith("clinicaliq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

import clinicaliq  # noqa: E402
import clinicaliq.nodes as _nodes  # noqa: E402
import clinicaliq.tools as _tools  # noqa: E402
from clinicaliq.config import SYSTEM_PROMPT  # noqa: E402
from clinicaliq.state import ClinicalIQState  # noqa: E402
from clinicaliq.tools import _run_tool, query_doctors, query_services  # noqa: E402
from clinicaliq.nodes import respond  # noqa: E402
from clinicaliq.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# TestClinicalIQState
# ---------------------------------------------------------------------------

class TestClinicalIQState:
    def test_state_has_customer_message_field(self):
        assert "customer_message" in ClinicalIQState.__annotations__

    def test_state_has_response_field(self):
        assert "response" in ClinicalIQState.__annotations__

    def test_state_has_history_field(self):
        assert "history" in ClinicalIQState.__annotations__

    def test_state_has_query_type_field(self):
        assert "query_type" in ClinicalIQState.__annotations__

    def test_state_has_retrieved_docs_field(self):
        assert "retrieved_docs" in ClinicalIQState.__annotations__

    def test_state_has_exactly_five_fields(self):
        assert len(ClinicalIQState.__annotations__) == 5


# ---------------------------------------------------------------------------
# TestQueryDoctorsTool
# ---------------------------------------------------------------------------

class TestQueryDoctorsTool:
    def test_query_doctors_all_returns_multiple(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({"specialty": "all"})
        assert "Dr. Meera Nair" in result
        assert "Dr. Rajesh Kumar" in result

    def test_query_doctors_specialty_filter_returns_matching(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({"specialty": "Cardiology"})
        assert "Dr. Rajesh Kumar" in result

    def test_query_doctors_specialty_filter_excludes_others(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({"specialty": "Cardiology"})
        assert "Dr. Meera Nair" not in result

    def test_query_doctors_includes_fee(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({"specialty": "all"})
        assert "500" in result or "800" in result

    def test_query_doctors_includes_available_days(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({"specialty": "all"})
        assert "Mon" in result

    def test_query_doctors_no_match_returns_message(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({"specialty": "Atlantis"})
        assert "No doctors found" in result

    def test_query_doctors_default_is_all(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({})
        assert "Dr. Meera Nair" in result
        assert "Dr. Rajesh Kumar" in result

    def test_query_doctors_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        assert isinstance(query_doctors.invoke({"specialty": "all"}), str)

    def test_query_doctors_general_medicine_filter(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({"specialty": "General Medicine"})
        assert "Dr. Meera Nair" in result
        assert "Dr. Rajesh Kumar" not in result

    def test_query_doctors_includes_specialty_label(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({"specialty": "Cardiology"})
        assert "Cardiology" in result


# ---------------------------------------------------------------------------
# TestQueryServicesTool
# ---------------------------------------------------------------------------

class TestQueryServicesTool:
    def test_query_services_all_returns_multiple(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_services.invoke({"department": "all"})
        assert "ECG" in result
        assert "Blood" in result

    def test_query_services_department_filter_returns_matching(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_services.invoke({"department": "Cardiology"})
        assert "ECG" in result

    def test_query_services_department_filter_excludes_others(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_services.invoke({"department": "Cardiology"})
        assert "Basic Blood Panel" not in result

    def test_query_services_includes_price(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_services.invoke({"department": "all"})
        assert "350" in result or "450" in result

    def test_query_services_no_match_returns_message(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_services.invoke({"department": "Atlantis"})
        assert "No services found" in result

    def test_query_services_default_is_all(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_services.invoke({})
        assert "ECG" in result

    def test_query_services_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        assert isinstance(query_services.invoke({"department": "all"}), str)


# ---------------------------------------------------------------------------
# TestToolSQLSafety
# ---------------------------------------------------------------------------

class TestToolSQLSafety:
    def test_query_doctors_sql_injection_safe(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_doctors.invoke({"specialty": "'; DROP TABLE doctors; --"})
        assert isinstance(result, str)
        normal = query_doctors.invoke({"specialty": "Cardiology"})
        assert "Dr. Rajesh Kumar" in normal

    def test_query_doctors_uses_question_mark_placeholder(self):
        import inspect
        source = inspect.getsource(query_doctors.func)
        assert "LIKE ?" in source, (
            "query_doctors must use a ? placeholder for the specialty parameter. "
            "Never interpolate user input directly into a SQL string."
        )


# ---------------------------------------------------------------------------
# TestToolsBinding
# ---------------------------------------------------------------------------

class TestToolsBinding:
    def test_llm_with_tools_exists(self):
        assert hasattr(_tools, "llm_with_tools"), (
            "llm_with_tools not found. Create it with llm.bind_tools([query_doctors, query_services])."
        )

    def test_query_doctors_is_tool_decorated(self):
        assert hasattr(query_doctors, "name"), (
            "query_doctors does not appear to be decorated with @tool."
        )

    def test_query_services_is_tool_decorated(self):
        assert hasattr(query_services, "name"), (
            "query_services does not appear to be decorated with @tool."
        )

    def test_query_doctors_tool_name(self):
        assert query_doctors.name == "query_doctors"

    def test_query_services_tool_name(self):
        assert query_services.name == "query_services"

    def test_system_prompt_has_no_hardcoded_fees(self):
        assert "Rs. 500" not in SYSTEM_PROMPT and "Rs. 800" not in SYSTEM_PROMPT, (
            "Session 5 removes hardcoded fees from SYSTEM_PROMPT. "
            "Fees now come from query_doctors(). Remove any hardcoded fee references."
        )

    def test_system_prompt_mentions_tools_or_database(self):
        assert "tool" in SYSTEM_PROMPT.lower() or "database" in SYSTEM_PROMPT.lower(), (
            "SYSTEM_PROMPT should instruct the LLM to use database tools for fees."
        )


# ---------------------------------------------------------------------------
# TestRunToolDispatch
# ---------------------------------------------------------------------------

class TestRunToolDispatch:
    def test_run_tool_dispatches_query_doctors(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_doctors", {"specialty": "Cardiology"})
        assert "Dr. Rajesh Kumar" in result

    def test_run_tool_dispatches_query_services(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_services", {"department": "all"})
        assert "ECG" in result

    def test_run_tool_unknown_name_returns_error_string(self):
        result = _run_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    def test_run_tool_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_doctors", {"specialty": "all"})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestRespondWithTools
# ---------------------------------------------------------------------------

class TestRespondWithTools:
    def _make_tool_call_result(self, tool_name, args, call_id="call_abc123"):
        result = MagicMock()
        result.content = ""
        result.tool_calls = [{"id": call_id, "name": tool_name, "args": args}]
        return result

    def _make_text_result(self, content):
        result = MagicMock()
        result.content = content
        result.tool_calls = []
        return result

    def _base_state(self):
        return {
            "customer_message": "What is the consultation fee for Cardiology?",
            "response": "", "history": [], "query_type": "SIMPLE", "retrieved_docs": [],
        }

    def test_respond_calls_llm_with_tools_first(self):
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm:
            mock_wt.invoke.return_value = self._make_text_result(
                "The Cardiology consultation fee is Rs. 800."
            )
            respond(self._base_state())
        mock_wt.invoke.assert_called_once()
        mock_llm.invoke.assert_not_called()

    def test_respond_no_tool_calls_returns_first_result(self):
        expected = "Apollo Health Clinic has departments for Cardiology and General Medicine."
        state    = {**self._base_state(), "customer_message": "What departments do you have?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm"):
            mock_wt.invoke.return_value = self._make_text_result(expected)
            result = respond(state)
        assert result["response"] == expected

    def test_respond_makes_second_call_when_tool_requested(self):
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Dr. Rajesh Kumar (Cardiology) Fee: Rs. 800"):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_doctors", {"specialty": "Cardiology"}
            )
            mock_llm.invoke.return_value = self._make_text_result(
                "Dr. Rajesh Kumar in Cardiology charges Rs. 800. ClinicalIQ | Apollo Health Clinic"
            )
            respond(self._base_state())
        mock_llm.invoke.assert_called_once()

    def test_respond_executes_tool_via_run_tool(self):
        state = {**self._base_state(), "customer_message": "How much does an ECG cost?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="ECG (Cardiology): Rs. 350") as mock_rt:
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_services", {"department": "Cardiology"}
            )
            mock_llm.invoke.return_value = self._make_text_result("An ECG costs Rs. 350.")
            respond(state)
        mock_rt.assert_called_once_with("query_services", {"department": "Cardiology"})

    def test_respond_uses_second_call_content_as_response(self):
        final_answer = "An ECG at Apollo Health Clinic costs Rs. 350. ClinicalIQ | Apollo Health Clinic"
        state = {**self._base_state(), "customer_message": "How much does an ECG cost?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="ECG (Cardiology): Rs. 350"):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_services", {"department": "Cardiology"}
            )
            mock_llm.invoke.return_value = self._make_text_result(final_answer)
            result = respond(state)
        assert result["response"] == final_answer

    def test_respond_history_grows_by_two(self):
        state = {**self._base_state(), "customer_message": "What services do you offer?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm"):
            mock_wt.invoke.return_value = self._make_text_result("We offer ECG, blood tests, and more.")
            result = respond(state)
        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["role"] == "assistant"

    def test_respond_appends_tool_message_to_conversation(self):
        captured_messages = []

        def capture_invoke(msgs):
            captured_messages.extend(msgs)
            return MagicMock(content="ECG costs Rs. 350. ClinicalIQ | Apollo Health Clinic", tool_calls=[])

        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="ECG (Cardiology): Rs. 350"):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_services", {"department": "Cardiology"}
            )
            mock_llm.invoke.side_effect = capture_invoke
            respond(self._base_state())

        from langchain_core.messages import ToolMessage as TM
        tool_messages = [m for m in captured_messages if isinstance(m, TM)]
        assert len(tool_messages) == 1
        assert "ECG" in tool_messages[0].content or "350" in tool_messages[0].content


# ---------------------------------------------------------------------------
# TestGraphRouting
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def _mock_vectorstore(self):
        vs = MagicMock()
        vs.similarity_search.return_value = []
        return vs

    def test_simple_path_calls_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "llm"), \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="SIMPLE")
            mock_wt.invoke.return_value = MagicMock(
                content="The Cardiology consultation fee is Rs. 800.", tool_calls=[]
            )
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-simple"}}
            graph.invoke(
                {"customer_message": "What is the consultation fee for Cardiology?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_called_once()

    def test_complex_path_skips_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-complex"}}
            result = graph.invoke(
                {"customer_message": "I have chest pain, what should I do?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "nurse" in result["response"] or "+91-80-2222-1111" in result["response"]

    def test_out_of_scope_path_skips_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-oos"}}
            result = graph.invoke(
                {"customer_message": "Recommend a hospital in Delhi.", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "only help with Apollo" in result["response"]
