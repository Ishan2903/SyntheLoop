"""Unit tests for the LLM Planner module (backend/planner/prompts.py and backend/planner/llm_planner.py)."""

import json
from unittest.mock import MagicMock, patch
import pytest

from backend.planner.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from backend.planner.llm_planner import LLMPlanner


@pytest.fixture
def sample_eda_summary():
    return {
        "n_rows": 1000,
        "n_columns": 4,
        "target_column": "income",
        "columns": {
            "age": {
                "type": "continuous",
                "missing_pct": 0.0,
                "n_unique": 70,
                "mean": 38.5,
                "std": 13.2,
                "skew": 0.5,
                "kurtosis": -0.2,
                "min": 17.0,
                "max": 90.0,
            },
            "workclass": {
                "type": "categorical",
                "missing_pct": 0.05,
                "n_unique": 5,
                "top_categories": {"Private": 0.7, "Self-Emp": 0.2},
                "balance_ratio": 0.28,
            },
            "education": {
                "type": "categorical",
                "missing_pct": 0.0,
                "n_unique": 8,
                "top_categories": {"Bachelors": 0.4, "HS-grad": 0.3},
                "balance_ratio": 0.45,
            },
            "income": {
                "type": "categorical",
                "missing_pct": 0.0,
                "n_unique": 2,
                "top_categories": {"<=50K": 0.75, ">50K": 0.25},
                "balance_ratio": 0.33,
            },
        },
        "correlation_matrix": {
            "age": {"age": 1.0}
        },
        "class_balance": {"<=50K": 0.75, ">50K": 0.25},
        "categorical_columns": ["workclass", "education", "income"],
    }


@pytest.fixture
def valid_plan_dict():
    return {
        "categorical_columns": ["workclass", "education", "income"],
        "epochs": 150,
        "batch_size": 200,
        "generator_dim": [256, 256],
        "discriminator_dim": [256, 256],
        "pac": 10,
        "reasoning": "Standard dimensions and epochs suitable for 1000 rows.",
    }


def make_mock_client(content_responses):
    """Creates a mock Groq client returning a sequence of text responses."""
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


def test_build_planner_prompt_without_feedback(sample_eda_summary):
    prompt = build_planner_prompt(sample_eda_summary)
    assert "Dataset Input:" in prompt
    assert "initial CTGAN configuration" in prompt
    assert "prior_feedback" not in prompt
    # Ensure raw summary JSON is contained
    data = json.loads(prompt.split("Dataset Input:\n")[1])
    assert "eda_summary" in data
    assert data["eda_summary"]["n_rows"] == 1000


def test_build_planner_prompt_with_feedback(sample_eda_summary):
    feedback = {
        "diagnosis": "Correlation difference is high and age distribution is slightly skewed.",
        "weak_areas": ["correlation"],
        "weak_columns": ["age"],
        "config_adjustments": {"epochs": 200, "discriminator_dim": [256, 256]},
        "stop_recommended": False,
        "reasoning": "Increase epochs to improve correlation modeling.",
    }
    prompt = build_planner_prompt(sample_eda_summary, prior_feedback=feedback)
    assert "refinement iteration" in prompt
    data = json.loads(prompt.split("Dataset Input:\n")[1])
    assert "prior_feedback" in data
    assert data["prior_feedback"]["weak_areas"] == ["correlation"]


def test_plan_valid_response(sample_eda_summary, valid_plan_dict):
    client = make_mock_client([json.dumps(valid_plan_dict)])
    planner = LLMPlanner(client=client, model="llama-3.3-70b-versatile")
    result = planner.plan(sample_eda_summary)

    assert result == valid_plan_dict
    assert client.chat.completions.create.call_count == 1
    # Verify Data Contract 4.2 keys
    for key in ["categorical_columns", "epochs", "batch_size", "generator_dim", "discriminator_dim", "pac", "reasoning"]:
        assert key in result


def test_plan_strips_markdown_fences(sample_eda_summary, valid_plan_dict):
    fenced_response = f"```json\n{json.dumps(valid_plan_dict, indent=2)}\n```"
    client = make_mock_client([fenced_response])
    planner = LLMPlanner(client=client, model="llama-3.3-70b-versatile")
    result = planner.plan(sample_eda_summary)

    assert result == valid_plan_dict
    assert result["epochs"] == 150


def test_plan_rejects_hallucinated_columns_and_retries(sample_eda_summary, valid_plan_dict):
    # First response includes a column not in sample_eda_summary["categorical_columns"]
    invalid_dict = dict(valid_plan_dict)
    invalid_dict["categorical_columns"] = ["workclass", "hallucinated_column"]

    client = make_mock_client([json.dumps(invalid_dict), json.dumps(valid_plan_dict)])
    planner = LLMPlanner(client=client, model="llama-3.3-70b-versatile")
    result = planner.plan(sample_eda_summary)

    # First attempt rejected, retried once, succeeded
    assert client.chat.completions.create.call_count == 2
    assert result == valid_plan_dict
    assert "hallucinated_column" not in result["categorical_columns"]


def test_plan_retries_on_invalid_json(sample_eda_summary, valid_plan_dict):
    client = make_mock_client(["Here is your config: {not valid json...}", json.dumps(valid_plan_dict)])
    planner = LLMPlanner(client=client, model="llama-3.3-70b-versatile")
    result = planner.plan(sample_eda_summary)

    assert client.chat.completions.create.call_count == 2
    assert result == valid_plan_dict


def test_plan_fallback_on_consecutive_failures(sample_eda_summary):
    client = make_mock_client(["Invalid JSON 1", "Invalid JSON 2"])
    planner = LLMPlanner(client=client, model="llama-3.3-70b-versatile")
    result = planner.plan(sample_eda_summary)

    # Both attempts failed -> fallback returned without crashing
    assert client.chat.completions.create.call_count == 2
    assert result["categorical_columns"] == sample_eda_summary["categorical_columns"]
    assert result["epochs"] == 150
    assert result["batch_size"] == 500
    assert result["pac"] == 10
    assert "Fallback" in result["reasoning"]


def test_plan_handles_rate_limit_backoff(sample_eda_summary, valid_plan_dict):
    class MockRateLimitError(Exception):
        status_code = 429

    client = make_mock_client([MockRateLimitError("Rate limit reached"), json.dumps(valid_plan_dict)])
    planner = LLMPlanner(client=client, model="llama-3.3-70b-versatile")

    with patch("time.sleep", return_value=None) as mock_sleep:
        result = planner.plan(sample_eda_summary)

    assert result == valid_plan_dict
    assert mock_sleep.call_count == 1
    # 2 calls within attempt 1 due to 429 retry
    assert client.chat.completions.create.call_count == 2


def test_definition_of_done(sample_eda_summary, valid_plan_dict):
    """Methodology Section 5.5 Definition of Done:

    Given a hand-written EDA summary fixture, plan() returns a dict matching
    Data Contract 4.2, with categorical_columns validated as a subset of the input's
    categorical columns.
    """
    client = make_mock_client([json.dumps(valid_plan_dict)])
    planner = LLMPlanner(client=client, model="llama-3.3-70b-versatile")
    config = planner.plan(sample_eda_summary)

    # 1. Matching Data Contract 4.2
    assert isinstance(config["categorical_columns"], list)
    assert isinstance(config["epochs"], int) and 50 <= config["epochs"] <= 500
    assert isinstance(config["batch_size"], int) and 50 <= config["batch_size"] <= 1000 and config["batch_size"] % 10 == 0
    assert isinstance(config["generator_dim"], (list, tuple))
    assert isinstance(config["discriminator_dim"], (list, tuple))
    assert isinstance(config["pac"], int) and 1 <= config["pac"] <= 20
    assert isinstance(config["reasoning"], str) and len(config["reasoning"]) > 0

    # 2. categorical_columns validated as a subset of the input's categorical columns
    allowed = set(sample_eda_summary["categorical_columns"])
    for col in config["categorical_columns"]:
        assert col in allowed
