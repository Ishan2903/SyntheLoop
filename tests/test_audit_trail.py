"""Unit tests for the SyntheLoop AuditTrail module."""

import json
from pathlib import Path
import pytest

from backend.pipeline.audit_trail import AuditTrail, VALID_ACTIONS


@pytest.fixture
def sample_entry():
    return {
        "run_id": "test-run-123",
        "iteration": 1,
        "timestamp": "2026-09-05T10:00:00Z",
        "config_used": {
            "categorical_columns": ["workclass", "education"],
            "epochs": 100,
            "batch_size": 200,
            "generator_dim": [256, 256],
            "discriminator_dim": [256, 256],
            "pac": 10,
            "reasoning": "Baseline run for initial evaluation.",
        },
        "metrics": {
            "iteration": 1,
            "per_column_ks": {"age": 0.08, "capital_gain": 0.12},
            "correlation_diff_frobenius": 0.15,
            "class_balance_js_divergence": 0.04,
            "privacy_dcr_5th_percentile": 0.35,
            "utility": {"trtr_auc": 0.85, "tstr_auc": 0.82, "auc_drop": 0.03},
            "thresholds": {},
            "passed": {
                "ks": True,
                "correlation": True,
                "balance": True,
                "privacy": True,
                "utility": True,
            },
            "overall_passed": True,
        },
        "feedback": {
            "weak_areas": [],
            "weak_columns": [],
            "diagnosis": "All quality criteria met.",
            "config_adjustments": {},
            "stop_recommended": True,
        },
        "action_taken": "stopped_threshold_met",
    }


def test_audit_trail_init(tmp_path):
    """Verifies that AuditTrail creates the run directory and sets correct paths."""
    run_id = "run-abc-1"
    trail = AuditTrail(run_id=run_id, output_dir=str(tmp_path))

    assert trail.run_id == run_id
    assert trail.output_dir == tmp_path
    assert trail.run_dir == tmp_path / run_id
    assert trail.run_dir.exists()
    assert trail.trail_file == tmp_path / run_id / "audit_trail.json"
    assert trail.get_full_trail() == []
    assert trail.get_latest_entry() is None


def test_audit_trail_empty_run_id(tmp_path):
    """Verifies that initializing with an empty run_id raises ValueError."""
    with pytest.raises(ValueError, match="run_id cannot be empty"):
        AuditTrail(run_id="", output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="run_id cannot be empty"):
        AuditTrail(run_id="   ", output_dir=str(tmp_path))


def test_log_single_iteration(tmp_path, sample_entry):
    """Verifies logging a valid iteration saves to disk matching Data Contract 4.5."""
    trail = AuditTrail(run_id=sample_entry["run_id"], output_dir=str(tmp_path))
    trail.log_iteration(sample_entry)

    assert trail.trail_file.exists()
    saved_entries = trail.get_full_trail()
    assert len(saved_entries) == 1
    assert saved_entries[0] == sample_entry
    assert trail.get_latest_entry() == sample_entry

    # Verify JSON format directly from disk
    raw_data = json.loads(trail.trail_file.read_text(encoding="utf-8"))
    assert isinstance(raw_data, list)
    assert len(raw_data) == 1
    assert raw_data[0]["run_id"] == "test-run-123"
    assert raw_data[0]["iteration"] == 1
    assert raw_data[0]["action_taken"] == "stopped_threshold_met"


def test_log_iteration_auto_populates_missing_metadata(tmp_path):
    """Verifies run_id and timestamp are auto-populated if omitted."""
    run_id = "auto-meta-run"
    trail = AuditTrail(run_id=run_id, output_dir=str(tmp_path))

    minimal_entry = {
        "iteration": 1,
        "config_used": {"epochs": 100},
        "metrics": {"overall_passed": False},
        "action_taken": "continued",
    }

    trail.log_iteration(minimal_entry)
    saved = trail.get_full_trail()
    assert len(saved) == 1
    record = saved[0]

    assert record["run_id"] == run_id
    assert "timestamp" in record and len(record["timestamp"]) > 0
    assert record["feedback"] is None
    assert record["action_taken"] == "continued"


def test_log_multiple_iterations_sequential(tmp_path, sample_entry):
    """Verifies sequential appending across multiple iterations."""
    trail = AuditTrail(run_id=sample_entry["run_id"], output_dir=str(tmp_path))

    # Iteration 1
    entry_1 = dict(sample_entry)
    entry_1["iteration"] = 1
    entry_1["action_taken"] = "continued"
    trail.log_iteration(entry_1)

    # Iteration 2
    entry_2 = dict(sample_entry)
    entry_2["iteration"] = 2
    entry_2["action_taken"] = "stopped_max_iterations"
    trail.log_iteration(entry_2)

    history = trail.get_full_trail()
    assert len(history) == 2
    assert history[0]["iteration"] == 1
    assert history[0]["action_taken"] == "continued"
    assert history[1]["iteration"] == 2
    assert history[1]["action_taken"] == "stopped_max_iterations"
    assert trail.get_latest_entry() == history[1]


def test_persistence_across_instances(tmp_path, sample_entry):
    """Verifies that a newly initialized AuditTrail reads previously written logs."""
    run_id = "persistent-run"
    entry = dict(sample_entry)
    entry["run_id"] = run_id

    trail_instance_1 = AuditTrail(run_id=run_id, output_dir=str(tmp_path))
    trail_instance_1.log_iteration(entry)

    # Create separate instance with the same run_id
    trail_instance_2 = AuditTrail(run_id=run_id, output_dir=str(tmp_path))
    loaded = trail_instance_2.get_full_trail()
    assert len(loaded) == 1
    assert loaded[0]["iteration"] == entry["iteration"]


def test_invalid_action_rejected(tmp_path, sample_entry):
    """Verifies that an unknown action_taken raises a ValueError."""
    trail = AuditTrail(run_id=sample_entry["run_id"], output_dir=str(tmp_path))
    invalid_entry = dict(sample_entry)
    invalid_entry["action_taken"] = "invalid_status_action"

    with pytest.raises(ValueError, match="Invalid action_taken"):
        trail.log_iteration(invalid_entry)


def test_all_valid_actions_accepted(tmp_path, sample_entry):
    """Verifies that all actions defined in VALID_ACTIONS are accepted."""
    trail = AuditTrail(run_id=sample_entry["run_id"], output_dir=str(tmp_path))

    for idx, action in enumerate(VALID_ACTIONS, start=1):
        entry = dict(sample_entry)
        entry["iteration"] = idx
        entry["action_taken"] = action
        trail.log_iteration(entry)

    full = trail.get_full_trail()
    assert len(full) == len(VALID_ACTIONS)
    actions_logged = {e["action_taken"] for e in full}
    assert actions_logged == VALID_ACTIONS


def test_missing_required_fields_raises_error(tmp_path):
    """Verifies ValueError when required fields are missing."""
    trail = AuditTrail(run_id="req-fields-run", output_dir=str(tmp_path))

    # Missing iteration
    with pytest.raises(ValueError, match="missing required field: 'iteration'"):
        trail.log_iteration({"config_used": {}, "metrics": {}, "action_taken": "continued"})

    # Missing config_used
    with pytest.raises(ValueError, match="missing required field: 'config_used'"):
        trail.log_iteration({"iteration": 1, "metrics": {}, "action_taken": "continued"})

    # Missing metrics
    with pytest.raises(ValueError, match="missing required field: 'metrics'"):
        trail.log_iteration({"iteration": 1, "config_used": {}, "action_taken": "continued"})

    # Missing action_taken
    with pytest.raises(ValueError, match="missing required field: 'action_taken'"):
        trail.log_iteration({"iteration": 1, "config_used": {}, "metrics": {}})


def test_invalid_iteration_number(tmp_path, sample_entry):
    """Verifies non-positive or non-int iterations are rejected."""
    trail = AuditTrail(run_id=sample_entry["run_id"], output_dir=str(tmp_path))

    entry_zero = dict(sample_entry)
    entry_zero["iteration"] = 0
    with pytest.raises(ValueError, match="Iteration must be a positive integer"):
        trail.log_iteration(entry_zero)

    entry_str = dict(sample_entry)
    entry_str["iteration"] = "1"
    with pytest.raises(ValueError, match="Iteration must be a positive integer"):
        trail.log_iteration(entry_str)


def test_corrupted_file_recovery(tmp_path, sample_entry):
    """Verifies graceful handling if audit_trail.json is empty or corrupted."""
    trail = AuditTrail(run_id=sample_entry["run_id"], output_dir=str(tmp_path))
    # Write garbage to the trail file
    trail.trail_file.write_text("invalid json content {{{", encoding="utf-8")

    # get_full_trail should return empty list rather than crashing
    assert trail.get_full_trail() == []

    # Logging an iteration should overwrite/repair the corrupted file
    trail.log_iteration(sample_entry)
    assert len(trail.get_full_trail()) == 1
    assert trail.get_full_trail()[0]["iteration"] == 1
