"""
s06/tests/test_s06.py
---------------------
Tests for Session 6: Baseline Evaluation.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

import evaluate  # noqa: E402
from evaluate import (  # noqa: E402
    DATASET_PATH,
    PASS_SCORE,
    REQUIRED_FIELDS,
    evaluate_response,
    generate_report,
    load_dataset,
    parse_judge_response,
    run_evaluation,
)


@pytest.fixture
def dataset():
    return load_dataset(DATASET_PATH)


@pytest.fixture
def simple_item():
    return {
        "id": "S01",
        "query": "How do I book an appointment at Apollo Health Clinic?",
        "expected_route": "SIMPLE",
        "category": "appointments",
        "criteria": ["080-33001100", "appointment"],
        "must_not_contain": ["Manipal"],
    }


@pytest.fixture
def complex_item():
    return {
        "id": "C01",
        "query": "I have chest pain. What should I do?",
        "expected_route": "COMPLEX",
        "category": "complex",
        "criteria": ["nurses"],
        "must_not_contain": [],
    }


@pytest.fixture
def oos_item():
    return {
        "id": "O01",
        "query": "Tell me a joke.",
        "expected_route": "OUT_OF_SCOPE",
        "category": "oos",
        "criteria": ["Apollo Health Clinic services"],
        "must_not_contain": [],
    }


class TestGoldenDataset:
    def test_dataset_loads(self, dataset):
        assert isinstance(dataset, list)

    def test_dataset_has_40_items(self, dataset):
        assert len(dataset) == 40, f"Expected 40 items, got {len(dataset)}"

    def test_dataset_has_20_simple(self, dataset):
        simple = [d for d in dataset if d["expected_route"] == "SIMPLE"]
        assert len(simple) == 20

    def test_dataset_has_10_complex(self, dataset):
        complex_ = [d for d in dataset if d["expected_route"] == "COMPLEX"]
        assert len(complex_) == 10

    def test_dataset_has_10_oos(self, dataset):
        oos = [d for d in dataset if d["expected_route"] == "OUT_OF_SCOPE"]
        assert len(oos) == 10

    def test_all_items_have_required_fields(self, dataset):
        for item in dataset:
            missing = REQUIRED_FIELDS - set(item.keys())
            assert not missing, f"Item {item.get('id')} missing: {missing}"

    def test_all_ids_are_unique(self, dataset):
        ids = [d["id"] for d in dataset]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in dataset"

    def test_expected_route_values_are_valid(self, dataset):
        valid = {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}
        for item in dataset:
            assert item["expected_route"] in valid

    def test_criteria_is_a_list(self, dataset):
        for item in dataset:
            assert isinstance(item["criteria"], list)

    def test_simple_items_have_non_empty_criteria(self, dataset):
        for item in dataset:
            if item["expected_route"] == "SIMPLE":
                assert len(item["criteria"]) > 0


class TestLoadDataset:
    def test_load_returns_list(self):
        result = load_dataset(DATASET_PATH)
        assert isinstance(result, list)

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.json")

    def test_load_raises_on_missing_fields(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([{"id": "X01", "query": "hello"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="missing fields"):
            load_dataset(bad)

    def test_load_raises_with_item_id_in_message(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([{"id": "MISSING_FIELDS", "query": "q"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="MISSING_FIELDS"):
            load_dataset(bad)

    def test_load_valid_file_returns_correct_count(self, tmp_path):
        items = [
            {"id": f"T{i}", "query": "q", "expected_route": "SIMPLE", "category": "cat", "criteria": ["x"]}
            for i in range(5)
        ]
        path = tmp_path / "valid.json"
        path.write_text(json.dumps(items), encoding="utf-8")
        result = load_dataset(path)
        assert len(result) == 5


class TestParseJudgeResponse:
    def test_parses_well_formed_output(self):
        score, reason = parse_judge_response("SCORE: 4\nREASON: The response is accurate.")
        assert score == 4
        assert "accurate" in reason

    def test_parses_score_1(self):
        score, _ = parse_judge_response("SCORE: 1\nREASON: Wrong department given.")
        assert score == 1

    def test_parses_score_5(self):
        score, _ = parse_judge_response("SCORE: 5\nREASON: Perfect answer.")
        assert score == 5

    def test_clamps_score_above_5(self):
        score, _ = parse_judge_response("SCORE: 9\nREASON: impossible.")
        assert score == 5

    def test_clamps_score_below_1(self):
        score, _ = parse_judge_response("SCORE: -3\nREASON: bad.")
        assert score == 1

    def test_returns_zero_on_missing_score(self):
        score, _ = parse_judge_response("REASON: No score line.")
        assert score == 0

    def test_returns_default_reason_on_missing_reason(self):
        score, reason = parse_judge_response("SCORE: 3")
        assert score == 3
        assert "parse" in reason.lower() or reason

    def test_handles_empty_output(self):
        score, reason = parse_judge_response("")
        assert score == 0
        assert isinstance(reason, str)

    def test_handles_extra_whitespace(self):
        score, reason = parse_judge_response("  SCORE:  4  \n  REASON:  Good response.  ")
        assert score == 4
        assert "Good response" in reason

    def test_handles_non_integer_score(self):
        score, _ = parse_judge_response("SCORE: four\nREASON: Spelled out.")
        assert score == 0

    def test_reason_with_colon_in_text(self):
        _, reason = parse_judge_response("SCORE: 3\nREASON: Hours are correct: Monday to Saturday.")
        assert "Monday" in reason


class TestEvaluateResponse:
    def test_complex_passes_when_criteria_met(self, complex_item):
        result = {"query_type": "COMPLEX", "response": "Our nurses are available to assist."}
        with patch("evaluate.llm_judge") as mock_judge:
            out = evaluate_response(complex_item, result)
        mock_judge.assert_not_called()
        assert out["passed"] is True
        assert out["score"] == 5

    def test_complex_fails_when_criteria_missing(self, complex_item):
        result = {"query_type": "COMPLEX", "response": "Please call us for help."}
        out = evaluate_response(complex_item, result)
        assert out["passed"] is False
        assert out["score"] == 1

    def test_complex_fails_on_wrong_route(self, complex_item):
        result = {"query_type": "SIMPLE", "response": "Please speak with our nurses."}
        out = evaluate_response(complex_item, result)
        assert out["route_correct"] is False
        assert out["passed"] is False

    def test_oos_passes_when_criteria_met(self, oos_item):
        result = {"query_type": "OUT_OF_SCOPE", "response": "I can only assist with Apollo Health Clinic services."}
        out = evaluate_response(oos_item, result)
        assert out["passed"] is True
        assert out["score"] == 5

    def test_oos_fails_when_criteria_missing(self, oos_item):
        result = {"query_type": "OUT_OF_SCOPE", "response": "That is outside my expertise."}
        out = evaluate_response(oos_item, result)
        assert out["passed"] is False

    def test_simple_passes_with_high_score(self, simple_item):
        result = {"query_type": "SIMPLE", "response": "Call 080-33001100 to book an appointment."}
        with patch("evaluate.llm_judge", return_value=(5, "Perfect.")):
            out = evaluate_response(simple_item, result)
        assert out["passed"] is True
        assert out["score"] == 5

    def test_simple_fails_with_low_score(self, simple_item):
        result = {"query_type": "SIMPLE", "response": "I am not sure how to book."}
        with patch("evaluate.llm_judge", return_value=(2, "Incomplete.")):
            out = evaluate_response(simple_item, result)
        assert out["passed"] is False

    def test_simple_fails_on_wrong_route(self, simple_item):
        result = {"query_type": "COMPLEX", "response": "Call 080-33001100 to book an appointment."}
        with patch("evaluate.llm_judge", return_value=(5, "Good.")):
            out = evaluate_response(simple_item, result)
        assert out["route_correct"] is False
        assert out["passed"] is False

    def test_simple_fails_on_forbidden_content(self, simple_item):
        result = {"query_type": "SIMPLE", "response": "Book via 080-33001100. Manipal is also good."}
        with patch("evaluate.llm_judge", return_value=(4, "Good but mentions Manipal.")):
            out = evaluate_response(simple_item, result)
        assert out["forbidden_found"] == ["Manipal"]
        assert out["passed"] is False

    def test_result_has_all_required_keys(self, simple_item):
        result = {"query_type": "SIMPLE", "response": "Call 080-33001100."}
        with patch("evaluate.llm_judge", return_value=(4, "Good.")):
            out = evaluate_response(simple_item, result)
        for key in ("id", "query", "category", "expected_route", "actual_route",
                    "route_correct", "score", "reason", "forbidden_found", "passed", "response"):
            assert key in out

    def test_actual_route_is_from_result(self, simple_item):
        result = {"query_type": "SIMPLE", "response": "Call 080-33001100."}
        with patch("evaluate.llm_judge", return_value=(4, "Good.")):
            out = evaluate_response(simple_item, result)
        assert out["actual_route"] == "SIMPLE"

    def test_handles_missing_query_type_in_result(self, simple_item):
        result = {"response": "Call 080-33001100 for appointment."}
        with patch("evaluate.llm_judge", return_value=(4, "Good.")):
            out = evaluate_response(simple_item, result)
        assert out["actual_route"] == "UNKNOWN"


class TestGenerateReport:
    def _make_results(self, specs):
        out = []
        for i, (cat, passed, score) in enumerate(specs):
            route = "COMPLEX" if cat == "complex" else ("oos" if cat == "oos" else "SIMPLE")
            out.append({
                "id": f"T{i:02d}", "query": f"Question {i}", "category": cat,
                "expected_route": route, "actual_route": route, "route_correct": True,
                "score": score, "reason": "test reason", "forbidden_found": [],
                "passed": passed, "response": "some response",
            })
        return out

    def test_total_count(self):
        assert generate_report(self._make_results([("appointments", True, 5)] * 10))["total"] == 10

    def test_passed_count(self):
        results = self._make_results([("appointments", True, 5)] * 7 + [("appointments", False, 1)] * 3)
        assert generate_report(results)["passed"] == 7

    def test_failed_count(self):
        results = self._make_results([("appointments", True, 5)] * 7 + [("appointments", False, 1)] * 3)
        assert generate_report(results)["failed"] == 3

    def test_pass_rate_calculation(self):
        results = self._make_results([("appointments", True, 5)] * 3 + [("appointments", False, 1)] * 1)
        assert abs(generate_report(results)["pass_rate"] - 0.75) < 0.01

    def test_average_score_excludes_complex_and_oos(self):
        results = self._make_results([
            ("appointments", True, 4), ("departments", True, 4),
            ("complex", True, 5), ("oos", True, 5),
        ])
        assert generate_report(results)["average_score"] == 4.0

    def test_by_category_contains_all_categories(self):
        results = self._make_results([
            ("appointments", True, 5), ("departments", False, 2), ("test_prep", True, 4),
        ])
        report = generate_report(results)
        assert "appointments" in report["by_category"]
        assert "departments" in report["by_category"]
        assert "test_prep" in report["by_category"]

    def test_by_category_pass_rate(self):
        results = self._make_results([
            ("appointments", True, 5), ("appointments", True, 4), ("appointments", False, 1),
        ])
        cat = generate_report(results)["by_category"]["appointments"]
        assert abs(cat["pass_rate"] - 2/3) < 0.01

    def test_failures_list_contains_failed_items(self):
        results = self._make_results([("appointments", True, 5), ("appointments", False, 2)])
        report = generate_report(results)
        assert len(report["failures"]) == 1
        assert report["failures"][0]["id"] == "T01"

    def test_failures_list_has_required_keys(self):
        results = self._make_results([("appointments", False, 1)])
        for key in ("id", "query", "reason", "score", "actual_route"):
            assert key in generate_report(results)["failures"][0]

    def test_empty_results_does_not_crash(self):
        report = generate_report([])
        assert report["total"] == 0
        assert report["pass_rate"] == 0.0


class TestRunEvaluation:
    def _make_dataset(self, n=3):
        return [
            {"id": f"T{i:02d}", "query": f"Question {i}", "expected_route": "SIMPLE",
             "category": "appointments", "criteria": ["test"], "must_not_contain": []}
            for i in range(n)
        ]

    def _mock_eval_return(self):
        return {
            "id": "x", "query": "q", "category": "c", "expected_route": "SIMPLE",
            "actual_route": "SIMPLE", "route_correct": True, "score": 4,
            "reason": "r", "forbidden_found": [], "passed": True, "response": "r",
        }

    def test_run_evaluation_returns_list(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"query_type": "SIMPLE", "response": "test"}
        with patch("evaluate.evaluate_response", return_value=self._mock_eval_return()):
            results = run_evaluation(mock_graph, self._make_dataset(2))
        assert isinstance(results, list)
        assert len(results) == 2

    def test_run_evaluation_calls_graph_for_each_item(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"query_type": "SIMPLE", "response": "ok"}
        with patch("evaluate.evaluate_response", return_value=self._mock_eval_return()):
            run_evaluation(mock_graph, self._make_dataset(3))
        assert mock_graph.invoke.call_count == 3

    def test_run_evaluation_uses_unique_thread_ids(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"query_type": "SIMPLE", "response": "ok"}
        with patch("evaluate.evaluate_response", return_value=self._mock_eval_return()):
            run_evaluation(mock_graph, self._make_dataset(2))
        thread_ids = [
            call.kwargs["config"]["configurable"]["thread_id"]
            for call in mock_graph.invoke.call_args_list
        ]
        assert len(thread_ids) == len(set(thread_ids))

    def test_run_evaluation_handles_graph_exception(self):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("Graph crashed")
        err_return = {**self._mock_eval_return(), "actual_route": "ERROR", "passed": False}
        with patch("evaluate.evaluate_response", return_value=err_return):
            results = run_evaluation(mock_graph, self._make_dataset(1))
        assert len(results) == 1
