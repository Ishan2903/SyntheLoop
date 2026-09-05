"""Unit tests for LLM Evaluator module (backend/evaluator/prompts.py and backend/evaluator/llm_evaluator.py).

Verifies prompt building, JSON parsing/stripping, retry on schema/hallucination failure,
rule-based fallback on consecutive errors, rate-limit backoff, and Section 5.6 Definition of Done.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from backend.evaluator.prompts import EVALUATOR_SYSTEM_PROMPT, build_evaluator_prompt
from backend.evaluator.llm_evaluator import LLMEvaluator


@pytest.fixture
def sample_metrics_contract_4_3():
    """Hand-written metrics fixture matching Data Contract 4.3 with correlation: false."""
    return {
        "iteration": 2,
        "per_column_ks": {
            "age": 0.08,
            "salary": 0.22,
            "hours_per_week": 0.12,
        },
        "correlation_diff_frobenius": 0.28,
        "class_balance_js_divergence": 0.05,
        "privacy_dcr_5th_percentile": 0.31,
        "utility": {
            "trtr_auc": 0.87,
            "tstr_auc": 0.81,
            "auc_drop": 0.06,
        },
        "thresholds": {
            "ks_stat_max": 0.15,
            "corr_diff_max": 0.20,
            "js_divergence_max": 0.10,
            "dcr_min_percentile": 5.0,
            "utility_auc_drop_max": 0.10,
        },
        "passed": {
            "ks": False,
            "correlation": False,
            "balance": True,
            "privacy": True,
            "utility": True,
        },
        "overall_passed": False,
    }


@pytest.fixture
def valid_feedback_dict():
    """Valid feedback dict matching Data Contract 4.4."""
    return {
        "weak_areas": ["correlation", "ks"],
        "weak_columns": ["salary"],
        "diagnosis": "Correlation difference exceeds threshold and salary distribution diverged.",
        "config_adjustments": {
            "epochs": 200,
            "generator_dim": [512, 256],
            "discriminator_dim": [512, 256],
        },
        "stop_recommended": False,
    }


def make_mock_client(content_responses):
    """Creates a mock Groq client returning a sequence of text responses or exceptions."""
    client = MagicMock()
    side_effects = []
    for resp in content_responses:
        if isinstance(resp, Exception):
            side_effects.append(resp)
        else:
            mock_choice = MagicMock()
            mock_choice.message.content = resp
            mock_res = MagicMock()
            mock_res.choices = [mock_choice]
            side_effects.append(mock_res)
    client.chat.completions.create.side_effect = side_effects
    return client


def test_build_evaluator_prompt_without_prior_config(sample_metrics_contract_4_3):
    """Verifies evaluator prompt serializes metrics safely without raw data."""
    prompt = build_evaluator_prompt(sample_metrics_contract_4_3)
    assert "Metrics Input:" in prompt
    assert "Data Contract 4.4" in prompt
    assert "prior_config" not in prompt

    data = json.loads(prompt.split("Metrics Input:\n")[1])
    assert "metrics" in data
    assert data["metrics"]["iteration"] == 2
    assert "salary" in data["metrics"]["per_column_ks"]


def test_build_evaluator_prompt_with_prior_config(sample_metrics_contract_4_3):
    """Verifies evaluator prompt includes prior_config when provided."""
    prior_config = {"epochs": 150, "batch_size": 200}
    prompt = build_evaluator_prompt(sample_metrics_contract_4_3, prior_config=prior_config)
    data = json.loads(prompt.split("Metrics Input:\n")[1])
    assert "prior_config" in data
    assert data["prior_config"]["epochs"] == 150


def test_evaluate_feedback_valid_response(sample_metrics_contract_4_3, valid_feedback_dict):
    """Verifies valid LLM response is parsed and meets Data Contract 4.4."""
    client = make_mock_client([json.dumps(valid_feedback_dict)])
    evaluator = LLMEvaluator(client=client, model="llama-3.3-70b-versatile")
    result = evaluator.evaluate_feedback(sample_metrics_contract_4_3)

    assert result == valid_feedback_dict
    assert client.chat.completions.create.call_count == 1
    for key in ["weak_areas", "weak_columns", "diagnosis", "config_adjustments", "stop_recommended"]:
        assert key in result


def test_evaluate_feedback_definition_of_done(sample_metrics_contract_4_3, valid_feedback_dict):
    """Methodology Section 5.6 Definition of Done:

    Given a hand-written metrics fixture (Data Contract 4.3) with correlation: false,
    evaluate_feedback() returns feedback whose weak_areas includes 'correlation'.
    """
    client = make_mock_client([json.dumps(valid_feedback_dict)])
    evaluator = LLMEvaluator(client=client, model="llama-3.3-70b-versatile")
    feedback = evaluator.evaluate_feedback(sample_metrics_contract_4_3)

    assert "correlation" in feedback["weak_areas"]
    assert isinstance(feedback["stop_recommended"], bool)
    assert isinstance(feedback["diagnosis"], str)


def test_evaluate_feedback_strips_markdown_fences(sample_metrics_contract_4_3, valid_feedback_dict):
    """Verifies markdown fences (```json ... ```) are cleanly stripped."""
    fenced_response = f"```json\n{json.dumps(valid_feedback_dict, indent=2)}\n```"
    client = make_mock_client([fenced_response])
    evaluator = LLMEvaluator(client=client, model="llama-3.3-70b-versatile")
    result = evaluator.evaluate_feedback(sample_metrics_contract_4_3)

    assert result == valid_feedback_dict
    assert result["weak_areas"] == ["correlation", "ks"]


def test_evaluate_feedback_rejects_hallucinated_columns_and_retries(sample_metrics_contract_4_3, valid_feedback_dict):
    """Verifies weak_columns with hallucinated names are rejected and retried with corrective instruction."""
    invalid_feedback = dict(valid_feedback_dict)
    invalid_feedback["weak_columns"] = ["salary", "hallucinated_feature_x"]

    client = make_mock_client([json.dumps(invalid_feedback), json.dumps(valid_feedback_dict)])
    evaluator = LLMEvaluator(client=client, model="llama-3.3-70b-versatile")
    result = evaluator.evaluate_feedback(sample_metrics_contract_4_3)

    assert client.chat.completions.create.call_count == 2
    assert result == valid_feedback_dict
    assert "hallucinated_feature_x" not in result["weak_columns"]


def test_evaluate_feedback_retries_on_invalid_json(sample_metrics_contract_4_3, valid_feedback_dict):
    """Verifies that malformed JSON triggers a retry and parses successfully on attempt 2."""
    client = make_mock_client(["Here is your feedback: {not valid json...}", json.dumps(valid_feedback_dict)])
    evaluator = LLMEvaluator(client=client, model="llama-3.3-70b-versatile")
    result = evaluator.evaluate_feedback(sample_metrics_contract_4_3)

    assert client.chat.completions.create.call_count == 2
    assert result == valid_feedback_dict


def test_evaluate_feedback_fallback_on_consecutive_failures(sample_metrics_contract_4_3):
    """Verifies consecutive failures trigger the deterministic rule-based fallback without crashing."""
    client = make_mock_client(["Broken response 1", "Broken response 2"])
    evaluator = LLMEvaluator(client=client, model="llama-3.3-70b-versatile")
    prior_config = {"epochs": 150}
    result = evaluator.evaluate_feedback(sample_metrics_contract_4_3, prior_config=prior_config)

    assert client.chat.completions.create.call_count == 2
    # Check rule-based fallback properties
    assert "Rule-based fallback" in result["diagnosis"]
    assert "correlation" in result["weak_areas"]
    # Salary has KS 0.22 > threshold 0.15
    assert "salary" in result["weak_columns"]
    # +50 epochs recommended
    assert result["config_adjustments"]["epochs"] == 200
    assert result["stop_recommended"] is False


def test_evaluate_feedback_fallback_when_all_passed():
    """Verifies fallback when all thresholds passed recommends stopping."""
    passed_metrics = {
        "iteration": 3,
        "per_column_ks": {"age": 0.05},
        "correlation_diff_frobenius": 0.10,
        "class_balance_js_divergence": 0.02,
        "privacy_dcr_5th_percentile": 0.35,
        "utility": {"trtr_auc": 0.85, "tstr_auc": 0.83, "auc_drop": 0.02},
        "thresholds": {"ks_stat_max": 0.15, "corr_diff_max": 0.20},
        "passed": {"ks": True, "correlation": True, "balance": True, "privacy": True, "utility": True},
        "overall_passed": True,
    }
    client = make_mock_client(["Broken 1", "Broken 2"])
    evaluator = LLMEvaluator(client=client, model="llama-3.3-70b-versatile")
    result = evaluator.evaluate_feedback(passed_metrics)

    assert result["stop_recommended"] is True
    assert result["weak_areas"] == []
    assert result["config_adjustments"] == {}


def test_evaluate_feedback_handles_rate_limit_backoff(sample_metrics_contract_4_3, valid_feedback_dict):
    """Verifies rate-limit errors (HTTP 429) back off and retry within the attempt."""
    class MockRateLimitError(Exception):
        status_code = 429

    client = make_mock_client([MockRateLimitError("Rate limit exceeded"), json.dumps(valid_feedback_dict)])
    evaluator = LLMEvaluator(client=client, model="llama-3.3-70b-versatile")

    with patch("time.sleep", return_value=None) as mock_sleep:
        result = evaluator.evaluate_feedback(sample_metrics_contract_4_3)

    assert result == valid_feedback_dict
    assert mock_sleep.call_count == 1
    assert client.chat.completions.create.call_count == 2
