"""
s09/tests/test_s09.py
---------------------
Tests for Session 9: Compliance filter + LangSmith observability.

Run with:
    pytest s09/tests/ -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "clinicaliq" or _k.startswith("clinicaliq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from clinicaliq.config import (         # noqa: E402
    CLINICALIQ_BANNED_PHRASES,
    SAFE_COMPLIANCE_RESPONSE,
)
from clinicaliq.nodes import (          # noqa: E402
    _BANNED_PATTERN,
    _check_compliance,
    _normalize_for_check,
    check_compliance,
    classify,
    decline,
    escalate,
)
from clinicaliq.state import ClinicalIQState   # noqa: E402
import clinicaliq.nodes as _nodes              # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(message: str = "test", response: str = "") -> ClinicalIQState:
    return {
        "customer_message": message,
        "response":         response,
        "history":          [],
        "query_type":       "SIMPLE",
        "retrieved_docs":   [],
        "compliance_status": "",
    }


# ---------------------------------------------------------------------------
# TestBannedPattern
# ---------------------------------------------------------------------------

class TestBannedPattern:
    def test_detects_you_have(self):
        assert _BANNED_PATTERN.search("you have diabetes") is not None

    def test_detects_i_diagnose(self):
        assert _BANNED_PATTERN.search("i diagnose you with a sinus infection") is not None

    def test_detects_prescribe(self):
        assert _BANNED_PATTERN.search("I would prescribe ibuprofen for this") is not None

    def test_no_match_for_safe_text(self):
        assert _BANNED_PATTERN.search("please visit our dermatology department") is None

    def test_case_insensitive(self):
        assert _BANNED_PATTERN.search("YOU HAVE HYPERTENSION") is not None


# ---------------------------------------------------------------------------
# TestNormalizeForCheck
# ---------------------------------------------------------------------------

class TestNormalizeForCheck:
    def test_lowercases_text(self):
        result = _normalize_for_check("You Have Diabetes")
        assert result == "you have diabetes"

    def test_replaces_unicode_en_dash(self):
        text   = "well–being"
        result = _normalize_for_check(text)
        assert "-" in result
        assert "–" not in result


# ---------------------------------------------------------------------------
# TestCheckCompliance
# ---------------------------------------------------------------------------

class TestCheckCompliance:
    def test_banned_phrase_you_have(self):
        passed, reason = _check_compliance("Based on your symptoms, you have a fever.")
        assert passed is False
        assert "banned phrase" in reason

    def test_banned_phrase_i_diagnose(self):
        passed, reason = _check_compliance("I diagnose you with mild anaemia.")
        assert passed is False
        assert "banned phrase" in reason

    def test_banned_phrase_take_this_medication(self):
        passed, reason = _check_compliance("Take this medication twice a day.")
        assert passed is False
        assert "banned phrase" in reason

    def test_safe_response_passes(self):
        passed, reason = _check_compliance(
            "Please book an appointment with our Cardiology department. "
            "You can call us at +91-80-2222-1111. ClinicalIQ | Apollo Health Clinic"
        )
        assert passed is True
        assert reason == "PASS"

    def test_banned_phrase_you_should_take(self):
        passed, reason = _check_compliance("you should take paracetamol for pain relief.")
        assert passed is False
        assert "banned phrase" in reason


# ---------------------------------------------------------------------------
# TestCheckComplianceNode
# ---------------------------------------------------------------------------

class TestCheckComplianceNode:
    def test_compliant_response_passes(self):
        state = _make_state(response="Our Cardiology department is open Monday to Saturday.")
        result = check_compliance(state)
        assert result["compliance_status"] == "PASS"
        assert "response" not in result or result.get("response") != SAFE_COMPLIANCE_RESPONSE

    def test_banned_phrase_replaces_response(self):
        state = _make_state(response="you have diabetes and need insulin immediately.")
        result = check_compliance(state)
        assert result["response"] == SAFE_COMPLIANCE_RESPONSE
        assert result["compliance_status"].startswith("FAIL")

    def test_compliance_fail_contains_reason(self):
        state = _make_state(response="I diagnose you with hypertension.")
        result = check_compliance(state)
        assert "FAIL" in result["compliance_status"]
        assert "banned phrase" in result["compliance_status"]

    def test_safe_compliance_response_imported(self):
        assert SAFE_COMPLIANCE_RESPONSE is not None
        assert "Apollo Health Clinic" in SAFE_COMPLIANCE_RESPONSE


# ---------------------------------------------------------------------------
# TestGraphNodes
# ---------------------------------------------------------------------------

class TestGraphNodes:
    def test_escalate_mentions_nurse(self):
        result = escalate(_make_state("I have chest pain"))
        assert "nurse" in result["response"].lower()

    def test_escalate_includes_phone(self):
        result = escalate(_make_state("what medicine should I take?"))
        assert "+91-80-2222-1111" in result["response"]

    def test_escalate_updates_history(self):
        result = escalate(_make_state("Is this rash serious?"))
        assert len(result["history"]) == 2

    def test_decline_mentions_apollo(self):
        result = decline(_make_state("Recommend a hospital in Delhi"))
        assert "Apollo Health Clinic" in result["response"]

    def test_decline_updates_history(self):
        result = decline(_make_state("Write me a poem"))
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# TestBuildGraph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        from clinicaliq.agent import build_graph
        assert build_graph(checkpointer=MemorySaver()) is not None

    def test_graph_invoke_complex_escalates(self):
        from langgraph.checkpoint.memory import MemorySaver
        from clinicaliq.agent import build_graph
        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What medicine for fever?", "response": "",
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-complex"}},
            )
        assert "nurse" in result["response"].lower()
        assert result["query_type"] == "COMPLEX"

    def test_graph_invoke_oos_declines(self):
        from langgraph.checkpoint.memory import MemorySaver
        from clinicaliq.agent import build_graph
        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What is the stock market doing?", "response": "",
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-oos"}},
            )
        assert "Apollo Health Clinic" in result["response"]
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_graph_result_includes_compliance_status(self):
        from langgraph.checkpoint.memory import MemorySaver
        from clinicaliq.agent import build_graph
        with patch.object(_nodes, "classifier_llm") as mock_llm, \
             patch.object(_nodes, "llm_with_tools") as mock_llm_tools:
            mock_llm.invoke.return_value = MagicMock(content="SIMPLE")
            mock_resp = MagicMock()
            mock_resp.tool_calls = []
            mock_resp.content = "Our Cardiology department is available Monday to Saturday."
            mock_llm_tools.invoke.return_value = mock_resp
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What departments do you have?", "response": "",
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-simple"}},
            )
        assert "compliance_status" in result
