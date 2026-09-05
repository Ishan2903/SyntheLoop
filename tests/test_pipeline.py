"""Unit and integration tests for the SyntheLoop FeedbackLoop pipeline."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from backend.pipeline.audit_trail import AuditTrail
from backend.pipeline.feedback_loop import FeedbackLoop


@pytest.fixture
def small_df():
    """Small mixed-type dataset for fast testing."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "age": np.random.randint(20, 65, size=n),
            "salary": np.random.uniform(30000, 120000, size=n),
            "education": np.random.choice(["HighSchool", "Bachelors", "Masters"], size=n),
            "target": np.random.choice([0, 1], size=n),
        }
    )


@pytest.fixture
def fast_planner():
    """Mock planner returning a minimal CTGAN configuration for speed."""
    planner = MagicMock()
    planner.plan.return_value = {
        "categorical_columns": ["education"],
        "epochs": 1,
        "batch_size": 20,
        "generator_dim": [64, 64],
        "discriminator_dim": [64, 64],
        "pac": 1,
        "reasoning": "Fast test plan.",
    }
    return planner


@pytest.fixture
def mock_evaluator_metrics():
    """Mock evaluator metrics function."""
    def _metrics(real, synth, eda_summary, thresholds, iteration):
        return {
            "iteration": iteration,
            "per_column_ks": {"age": 0.05, "salary": 0.08},
            "correlation_diff_frobenius": 0.10,
            "class_balance_js_divergence": 0.02,
            "privacy_dcr_5th_percentile": 0.40,
            "utility": {"trtr_auc": 0.85, "tstr_auc": 0.83, "auc_drop": 0.02},
            "thresholds": thresholds or {},
            "passed": {
                "ks": True,
                "correlation": True,
                "balance": True,
                "privacy": True,
                "utility": True,
            },
            "overall_passed": True,
        }

    return _metrics


@pytest.fixture
def mock_llm_evaluator():
    """Mock LLM evaluator returning successful feedback."""
    evaluator = MagicMock()
    evaluator.evaluate_feedback.return_value = {
        "weak_areas": [],
        "weak_columns": [],
        "diagnosis": "Quality criteria met.",
        "config_adjustments": {},
        "stop_recommended": True,
    }
    return evaluator


def test_input_validations(tmp_path):
    """Verifies validation on invalid input arguments."""
    trail = AuditTrail(run_id="valid-run", output_dir=str(tmp_path))

    # None DataFrame
    with pytest.raises(ValueError, match="must be a valid pandas DataFrame"):
        FeedbackLoop(real_df=None, audit_trail=trail)

    # Empty DataFrame
    with pytest.raises(ValueError, match="cannot be empty"):
        FeedbackLoop(real_df=pd.DataFrame(), audit_trail=trail)

    # max_iterations < 1
    sample = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="must be at least 1"):
        FeedbackLoop(real_df=sample, max_iterations=0, audit_trail=trail)


def test_feedback_loop_definition_of_done(
    small_df, fast_planner, mock_evaluator_metrics, mock_llm_evaluator, tmp_path
):
    """Methodology Section 5.8 Definition of Done:

    Runs FeedbackLoop.run() end-to-end on a small sample dataset with max_iterations=2,
    asserts it returns a dict with status in expected set and final_synth is a non-empty DataFrame.
    """
    run_id = "dod-run-001"
    trail = AuditTrail(run_id=run_id, output_dir=str(tmp_path))

    loop = FeedbackLoop(
        real_df=small_df,
        target_col="target",
        max_iterations=2,
        planner=fast_planner,
        evaluator_metrics_fn=mock_evaluator_metrics,
        llm_evaluator=mock_llm_evaluator,
        audit_trail=trail,
        save_artifacts=True,
    )

    result = loop.run()

    # Assert status in expected set
    assert result["status"] in {"completed_threshold_met", "completed_max_iterations"}
    assert isinstance(result["final_synth"], pd.DataFrame)
    assert not result["final_synth"].empty
    assert len(result["final_synth"]) == len(small_df)
    assert result["iterations_used"] >= 1
    assert "final_metrics" in result
    assert result["final_metrics"]["overall_passed"] is True

    # Check synthetic.csv was saved
    artifact_csv = trail.run_dir / "synthetic.csv"
    assert artifact_csv.exists()
    saved_df = pd.read_csv(artifact_csv)
    assert len(saved_df) == len(small_df)


def test_feedback_loop_early_stopping(
    small_df, fast_planner, mock_evaluator_metrics, mock_llm_evaluator, tmp_path
):
    """Verifies loop terminates after 1 iteration when threshold_met or stop_recommended."""
    run_id = "early-stop-run"
    trail = AuditTrail(run_id=run_id, output_dir=str(tmp_path))

    loop = FeedbackLoop(
        real_df=small_df,
        target_col="target",
        max_iterations=5,
        planner=fast_planner,
        evaluator_metrics_fn=mock_evaluator_metrics,
        llm_evaluator=mock_llm_evaluator,
        audit_trail=trail,
    )

    result = loop.run()
    assert result["status"] == "completed_threshold_met"
    assert result["iterations_used"] == 1

    entries = trail.get_full_trail()
    assert len(entries) == 1
    assert entries[0]["action_taken"] == "stopped_threshold_met"


def test_feedback_loop_runs_to_max_iterations(
    small_df, fast_planner, tmp_path
):
    """Verifies loop runs until max_iterations when quality threshold is not satisfied."""
    run_id = "max-iter-run"
    trail = AuditTrail(run_id=run_id, output_dir=str(tmp_path))

    # Metrics that fail
    def failing_metrics(real, synth, eda_summary, thresholds, iteration):
        return {
            "iteration": iteration,
            "per_column_ks": {"age": 0.35},
            "overall_passed": False,
            "passed": {"ks": False},
        }

    # Evaluator that does not stop early
    continuing_evaluator = MagicMock()
    continuing_evaluator.evaluate_feedback.return_value = {
        "weak_areas": ["ks"],
        "weak_columns": ["age"],
        "diagnosis": "KS failed.",
        "config_adjustments": {"epochs": 150},
        "stop_recommended": False,
    }

    loop = FeedbackLoop(
        real_df=small_df,
        target_col="target",
        max_iterations=2,
        planner=fast_planner,
        evaluator_metrics_fn=failing_metrics,
        llm_evaluator=continuing_evaluator,
        audit_trail=trail,
    )

    result = loop.run()
    assert result["status"] == "completed_max_iterations"
    assert result["iterations_used"] == 2

    entries = trail.get_full_trail()
    assert len(entries) == 2
    assert entries[0]["action_taken"] == "continued"
    assert entries[1]["action_taken"] == "stopped_max_iterations"

    # Verify prior_feedback was passed on iteration 2
    assert fast_planner.plan.call_count == 2
    assert fast_planner.plan.call_args_list[1].kwargs["prior_feedback"] is not None


def test_feedback_loop_handles_exception_and_logs_failure(
    small_df, tmp_path
):
    """Verifies errors inside the loop log action_taken: 'failed' and return status 'failed'."""
    run_id = "fail-test-run"
    trail = AuditTrail(run_id=run_id, output_dir=str(tmp_path))

    # Planner that throws an error
    failing_planner = MagicMock()
    failing_planner.plan.side_effect = RuntimeError("Groq API quota exhausted")

    mock_eval = MagicMock()

    loop = FeedbackLoop(
        real_df=small_df,
        max_iterations=3,
        planner=failing_planner,
        llm_evaluator=mock_eval,
        audit_trail=trail,
    )

    result = loop.run()
    assert result["status"] == "failed"
    assert "Groq API quota exhausted" in result["error"]
    assert result["iterations_used"] == 1

    entries = trail.get_full_trail()
    assert len(entries) == 1
    assert entries[0]["action_taken"] == "failed"


def test_config_immutability_in_audit_trail(
    small_df, fast_planner, mock_evaluator_metrics, mock_llm_evaluator, tmp_path
):
    """Verifies that configs recorded in the audit trail are decoupled snapshots."""
    run_id = "immutable-config-run"
    trail = AuditTrail(run_id=run_id, output_dir=str(tmp_path))

    loop = FeedbackLoop(
        real_df=small_df,
        target_col="target",
        max_iterations=1,
        planner=fast_planner,
        evaluator_metrics_fn=mock_evaluator_metrics,
        llm_evaluator=mock_llm_evaluator,
        audit_trail=trail,
    )

    loop.run()
    entry = trail.get_latest_entry()
    assert entry is not None

    # Mutate the planner's return value after the fact
    fast_planner.plan.return_value["epochs"] = 9999

    # Verify the audit trail snapshot was not mutated
    assert entry["config_used"]["epochs"] == 1
