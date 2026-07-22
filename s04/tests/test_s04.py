"""
s04/tests/test_s04.py
---------------------
Unit tests for Session 4: ChromaDB RAG.

Run from the clinicaliq/ directory:
    pytest s04/tests/ -v

All tests run without a live Groq API key, ChromaDB, or HuggingFace model.
The vectorstore and LLMs are mocked throughout.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "clinicaliq" or _k.startswith("clinicaliq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from clinicaliq.config import DECLINE_RESPONSE, ESCALATE_RESPONSE, RETRIEVAL_K, SYSTEM_PROMPT  # noqa: E402
from clinicaliq.state import ClinicalIQState  # noqa: E402
import clinicaliq.nodes as _nodes  # noqa: E402
from clinicaliq.nodes import classify, decline, escalate, respond, retrieve_docs, route_query  # noqa: E402
from clinicaliq.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_doc(page_content: str, source: str = "faq.md") -> MagicMock:
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = {"source": source}
    return doc


def _mock_vectorstore_with(docs: list) -> MagicMock:
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = docs
    return mock_vs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_checkpointer():
    return MemorySaver()


@pytest.fixture
def mock_llm_simple():
    with patch.object(_nodes, "llm") as mock_main, \
         patch.object(_nodes, "classifier_llm") as mock_clf:
        mock_clf.invoke.return_value = MagicMock(content="SIMPLE")
        mock_main.invoke.return_value = MagicMock(
            content="For a blood test, you must fast for 8-12 hours. ClinicalIQ | Apollo Health Clinic"
        )
        yield mock_main, mock_clf


@pytest.fixture
def mock_vectorstore():
    doc = _make_mock_doc(
        "For blood tests, patients must fast for 8-12 hours before the appointment. "
        "Water is permitted. Please inform reception if you have diabetes.",
        source="test_preparation.md",
    )
    mock_vs = _mock_vectorstore_with([doc])
    with patch.object(_nodes, "vectorstore", mock_vs), \
         patch.object(_nodes, "_init_vectorstore"):
        yield mock_vs


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------

class TestClinicalIQState:
    def test_state_has_customer_message(self):
        assert "customer_message" in ClinicalIQState.__annotations__

    def test_state_has_response(self):
        assert "response" in ClinicalIQState.__annotations__

    def test_state_has_history(self):
        assert "history" in ClinicalIQState.__annotations__

    def test_state_has_query_type(self):
        assert "query_type" in ClinicalIQState.__annotations__

    def test_state_has_retrieved_docs(self):
        assert "retrieved_docs" in ClinicalIQState.__annotations__, (
            "ClinicalIQState must have a 'retrieved_docs' field (added in Session 4). "
            "Add it after 'query_type' with type hint list[str]."
        )

    def test_retrieved_docs_is_list_type(self):
        annotation = ClinicalIQState.__annotations__["retrieved_docs"]
        origin = getattr(annotation, "__origin__", annotation)
        assert origin is list

    def test_state_instantiable_with_all_fields(self):
        state: ClinicalIQState = {
            "customer_message": "Do I need to fast before a blood test?",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   [],
        }
        assert state["retrieved_docs"] == []

    def test_state_retrieved_docs_accepts_strings(self):
        state: ClinicalIQState = {
            "customer_message": "test",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   ["[test_preparation.md]\nFast for 8-12 hours before blood tests."],
        }
        assert len(state["retrieved_docs"]) == 1


# ---------------------------------------------------------------------------
# retrieve_docs() node tests
# ---------------------------------------------------------------------------

class TestRetrieveDocsNode:
    def _state(self, question: str = "Do I need to fast before a blood test?") -> ClinicalIQState:
        return {
            "customer_message": question,
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   [],
        }

    def test_retrieve_docs_returns_dict(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert isinstance(result, dict)

    def test_retrieve_docs_returns_retrieved_docs_key(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert "retrieved_docs" in result

    def test_retrieve_docs_returns_list(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert isinstance(result["retrieved_docs"], list)

    def test_retrieve_docs_calls_similarity_search(self, mock_vectorstore):
        retrieve_docs(self._state("How do I book an appointment?"))
        mock_vectorstore.similarity_search.assert_called_once()

    def test_retrieve_docs_passes_question_to_search(self, mock_vectorstore):
        question = "Do I need to fast before a blood test?"
        retrieve_docs(self._state(question))
        call_args = mock_vectorstore.similarity_search.call_args
        assert call_args[0][0] == question or call_args[1].get("query") == question

    def test_retrieve_docs_passes_k_parameter(self, mock_vectorstore):
        retrieve_docs(self._state())
        call_args = mock_vectorstore.similarity_search.call_args
        k_value   = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("k")
        assert k_value == RETRIEVAL_K

    def test_retrieve_docs_formats_source_in_output(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert len(result["retrieved_docs"]) > 0
        first = result["retrieved_docs"][0]
        assert "test_preparation.md" in first

    def test_retrieve_docs_includes_page_content(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        combined = " ".join(result["retrieved_docs"])
        assert "fast" in combined.lower() or "blood" in combined.lower()

    def test_retrieve_docs_returns_empty_when_vectorstore_is_none(self):
        with patch.object(_nodes, "vectorstore", None), \
             patch.object(_nodes, "_init_vectorstore"):
            result = retrieve_docs(self._state())
        assert result == {"retrieved_docs": []}

    def test_retrieve_docs_returns_empty_on_exception(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search.side_effect = Exception("ChromaDB connection error")
        with patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            result = retrieve_docs(self._state())
        assert result == {"retrieved_docs": []}

    def test_retrieve_docs_does_not_call_llm(self, mock_vectorstore):
        with patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "classifier_llm") as mock_clf:
            retrieve_docs(self._state())
            mock_llm.invoke.assert_not_called()
            mock_clf.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# respond() node tests -- context injection
# ---------------------------------------------------------------------------

class TestRespondWithContext:
    def _state_with_docs(self, docs: list[str]) -> ClinicalIQState:
        return {
            "customer_message": "Do I need to fast before a blood test?",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   docs,
        }

    def test_respond_includes_retrieved_docs_in_system_message(self):
        chunk = "[test_preparation.md]\nFast for 8-12 hours before blood tests."
        state = self._state_with_docs([chunk])
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Yes, fasting is required.")
            respond(state)
        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "fast" in system_text.lower() or "test_preparation.md" in system_text

    def test_respond_without_docs_escalates_directly(self):
        state = self._state_with_docs([])
        with patch.object(_nodes, "llm") as mock_llm:
            result = respond(state)
        mock_llm.invoke.assert_not_called()
        assert result["response"] == ESCALATE_RESPONSE

    def test_respond_context_contains_policy_keyword(self):
        chunk = "[test_preparation.md]\nPatients must fast for 8 to 12 hours before a blood test."
        state = self._state_with_docs([chunk])
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Fast for 8 hours.")
            respond(state)
        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "8" in system_text or "fast" in system_text.lower()

    def test_respond_with_multiple_docs(self):
        chunks = [
            "[test_preparation.md]\nFast for 8-12 hours before blood tests.",
            "[departments_overview.md]\nCardiology department handles heart-related conditions.",
        ]
        state = self._state_with_docs(chunks)
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Both details noted.")
            respond(state)
        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "fast" in system_text.lower() or "8" in system_text
        assert "Cardiology" in system_text or "heart" in system_text.lower()

    def test_respond_updates_history_with_docs(self):
        chunk = "[appointment_guide.md]\nAppointments can be booked online or by calling reception."
        state = self._state_with_docs([chunk])
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="You can book online.")
            result = respond(state)
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# route_query() tests
# ---------------------------------------------------------------------------

class TestRouteQuery:
    def _state(self, query_type: str) -> dict:
        return {"customer_message": "test", "response": "",
                "history": [], "query_type": query_type, "retrieved_docs": []}

    def test_simple_routes_to_retrieve_docs(self):
        assert route_query(self._state("SIMPLE")) == "retrieve_docs"

    def test_complex_routes_to_retrieve_docs(self):
        # Under Option B, COMPLEX is not a valid type — coerced to IN_SCOPE → retrieve_docs.
        assert route_query(self._state("COMPLEX")) == "retrieve_docs"

    def test_out_of_scope_routes_to_decline(self):
        assert route_query(self._state("OUT_OF_SCOPE")) == "decline"

    def test_default_routes_to_retrieve_docs(self):
        state = {"customer_message": "test", "response": "", "history": [], "retrieved_docs": []}
        assert route_query(state) == "retrieve_docs"


# ---------------------------------------------------------------------------
# Graph routing tests
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def test_simple_path_calls_vectorstore(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "route-simple-rag"}}
        graph.invoke(
            {"customer_message": "Do I need to fast before a blood test?", "response": ""},
            config=config,
        )
        mock_vectorstore.similarity_search.assert_called_once()

    def test_simple_path_sets_retrieved_docs_in_result(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "route-simple-docs"}}
        result = graph.invoke(
            {"customer_message": "How do I book an appointment?", "response": ""},
            config=config,
        )
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)
        assert len(result["retrieved_docs"]) > 0

    def test_no_docs_path_escalates(self, memory_checkpointer):
        # Option B: IN_SCOPE → retrieve_docs → no docs returned → respond() escalates directly.
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        with patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "llm"):
            mock_clf.invoke.return_value = MagicMock(content="IN_SCOPE")
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-no-docs-escalate"}}
            result = graph.invoke(
                {"customer_message": "Which plan suits heavy users?", "response": ""},
                config=config,
            )
        assert result["response"] == ESCALATE_RESPONSE

    def test_out_of_scope_path_skips_vectorstore(self, memory_checkpointer):
        mock_vs = MagicMock()
        with patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "llm"):
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-oos-no-rag"}}
            graph.invoke(
                {"customer_message": "Write me a poem.", "response": ""},
                config=config,
            )
        mock_vs.similarity_search.assert_not_called()

    def test_all_paths_produce_non_empty_response(self, mock_vectorstore, memory_checkpointer):
        for query_type, question in [
            ("SIMPLE",       "Do I need to fast before a blood test?"),
            ("COMPLEX",      "I have chest pain, what should I do?"),
            ("OUT_OF_SCOPE", "Write a poem."),
        ]:
            with patch.object(_nodes, "classifier_llm") as mock_clf, \
                 patch.object(_nodes, "llm") as mock_llm:
                mock_clf.invoke.return_value = MagicMock(content=query_type)
                mock_llm.invoke.return_value = MagicMock(content="Some answer.")
                graph  = build_graph(checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": f"all-paths-{query_type}"}}
                result = graph.invoke(
                    {"customer_message": question, "response": ""},
                    config=config,
                )
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

class TestMemoryWithRAG:
    def test_history_accumulates_across_simple_turns(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        graph     = build_graph(checkpointer=memory_checkpointer)
        thread_id = "mem-rag-simple"
        config    = {"configurable": {"thread_id": thread_id}}

        graph.invoke(
            {"customer_message": "Do I need to fast before a blood test?", "response": ""},
            config=config,
        )
        result = graph.invoke(
            {"customer_message": "What departments do you have?", "response": ""},
            config=config,
        )
        assert len(result["history"]) == 4

    def test_different_threads_isolated(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        graph = build_graph(checkpointer=memory_checkpointer)

        graph.invoke(
            {"customer_message": "What are your clinic hours?", "response": ""},
            config={"configurable": {"thread_id": "rag-thread-X"}},
        )
        result = graph.invoke(
            {"customer_message": "How do I book an appointment?", "response": ""},
            config={"configurable": {"thread_id": "rag-thread-Y"}},
        )
        assert len(result["history"]) == 2
