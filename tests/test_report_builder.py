"""Unit tests for the SyntheLoop HTML Report Builder module."""

from pathlib import Path
import pytest

from backend.reports.report_builder import build_report, save_report


@pytest.fixture
def sample_eda_summary():
    return {
        "n_rows": 1000,
        "n_cols": 3,
        "target_column": "income",
        "columns": {
            "age": {
                "dtype": "int64",
                "type": "continuous",
                "missing_pct": 0.0,
                "n_unique": 50,
                "mean": 38.5,
                "std": 12.3,
            },
            "education": {
                "dtype": "object",
                "type": "categorical",
                "missing_pct": 1.2,
                "n_unique": 4,
                "top_categories": {"Bachelors": 0.5, "Masters": 0.3},
            },
            "income": {
                "dtype": "int64",
                "type": "categorical",
                "missing_pct": 0.0,
                "n_unique": 2,
                "top_categories": {"<=50K": 0.75, ">50K": 0.25},
            },
        },
    }


@pytest.fixture
def sample_audit_trail():
    return [
        {
            "run_id": "run-test-99",
            "iteration": 1,
            "timestamp": "2026-09-05T12:00:00Z",
            "config_used": {
                "epochs": 100,
                "batch_size": 200,
                "generator_dim": [128, 128],
                "discriminator_dim": [128, 128],
                "pac": 10,
                "reasoning": "Baseline training attempt.",
            },
            "metrics": {
                "iteration": 1,
                "per_column_ks": {"age": 0.18},
                "correlation_diff_frobenius": 0.25,
                "class_balance_js_divergence": 0.08,
                "privacy_dcr_5th_percentile": 0.25,
                "utility": {"trtr_auc": 0.85, "tstr_auc": 0.70, "auc_drop": 0.15},
                "thresholds": {},
                "passed": {
                    "ks": False,
                    "correlation": False,
                    "balance": True,
                    "privacy": True,
                    "utility": False,
                },
                "overall_passed": False,
            },
            "feedback": {
                "weak_areas": ["ks", "correlation", "utility"],
                "weak_columns": ["age"],
                "diagnosis": "Severe underfitting observed in correlation and KS.",
                "config_adjustments": {"epochs": 200},
                "stop_recommended": False,
            },
            "action_taken": "continued",
        },
        {
            "run_id": "run-test-99",
            "iteration": 2,
            "timestamp": "2026-09-05T12:05:00Z",
            "config_used": {
                "epochs": 200,
                "batch_size": 200,
                "generator_dim": [256, 256],
                "discriminator_dim": [256, 256],
                "pac": 10,
                "reasoning": "Refined capacity to address weak correlation.",
            },
            "metrics": {
                "iteration": 2,
                "per_column_ks": {"age": 0.09},
                "correlation_diff_frobenius": 0.14,
                "class_balance_js_divergence": 0.03,
                "privacy_dcr_5th_percentile": 0.38,
                "utility": {"trtr_auc": 0.85, "tstr_auc": 0.82, "auc_drop": 0.03},
                "thresholds": {
                    "ks_stat_max": 0.15,
                    "corr_diff_max": 0.20,
                    "js_divergence_max": 0.10,
                    "dcr_min_percentile": 5,
                    "utility_auc_drop_max": 0.10,
                },
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
                "diagnosis": "All thresholds satisfied.",
                "config_adjustments": {},
                "stop_recommended": True,
            },
            "action_taken": "stopped_threshold_met",
        },
    ]


@pytest.fixture
def sample_final_metrics(sample_audit_trail):
    return sample_audit_trail[-1]["metrics"]


def test_build_report_structure(sample_eda_summary, sample_audit_trail, sample_final_metrics):
    """Verifies that generated HTML contains all expected core sections and values."""
    run_id = "run-test-99"
    html_content = build_report(
        run_id=run_id,
        eda_summary=sample_eda_summary,
        audit_trail=sample_audit_trail,
        final_metrics=sample_final_metrics,
    )

    # Basic HTML document verification
    assert "<!DOCTYPE html>" in html_content
    assert f"SyntheLoop Synthesis Report — {run_id}" in html_content
    assert run_id in html_content

    # Sections exist
    assert "Executive Quality Scorecard" in html_content
    assert "Metric Trends Across Iterations" in html_content
    assert "Dataset Overview" in html_content
    assert "Full Audit Trail &amp; LLM Refinement History" in html_content

    # Dataset details
    assert "Rows: <strong>1000</strong>" in html_content
    assert "Columns: <strong>3</strong>" in html_content
    assert "income" in html_content
    assert "age" in html_content
    assert "education" in html_content

    # Iteration trend table content
    assert "#1" in html_content
    assert "#2" in html_content
    assert "continued" in html_content
    assert "stopped_threshold_met" in html_content

    # Audit trail details
    assert "Baseline training attempt." in html_content
    assert "Refined capacity to address weak correlation." in html_content
    assert "Severe underfitting observed in correlation and KS." in html_content


def test_build_report_passed_status(sample_eda_summary, sample_audit_trail, sample_final_metrics):
    """Verifies that passing run produces the PASSED badge."""
    html_content = build_report(
        run_id="run-pass-1",
        eda_summary=sample_eda_summary,
        audit_trail=sample_audit_trail,
        final_metrics=sample_final_metrics,
    )
    assert '<span class="badge badge-pass">PASSED</span>' in html_content


def test_build_report_failed_status(sample_eda_summary, sample_audit_trail):
    """Verifies that failing run produces THRESHOLD NOT MET badge."""
    failed_metrics = dict(sample_audit_trail[0]["metrics"])
    failed_metrics["overall_passed"] = False

    html_content = build_report(
        run_id="run-fail-1",
        eda_summary=sample_eda_summary,
        audit_trail=[sample_audit_trail[0]],
        final_metrics=failed_metrics,
    )
    assert '<span class="badge badge-fail">THRESHOLD NOT MET</span>' in html_content


def test_build_report_handles_sparse_and_none_inputs():
    """Verifies report builder is resilient against None or empty arguments."""
    html_content = build_report(
        run_id="sparse-run",
        eda_summary=None,
        audit_trail=None,
        final_metrics=None,
    )
    assert "<!DOCTYPE html>" in html_content
    assert "sparse-run" in html_content
    assert "Executive Quality Scorecard" in html_content
    assert "No column data available" in html_content
    assert "No iteration history available" in html_content
    assert "No audit trail logged" in html_content


def test_save_report(tmp_path, sample_eda_summary, sample_audit_trail, sample_final_metrics):
    """Verifies save_report writes report.html into {output_dir}/{run_id}/."""
    run_id = "save-run-test"
    html_content = build_report(
        run_id=run_id,
        eda_summary=sample_eda_summary,
        audit_trail=sample_audit_trail,
        final_metrics=sample_final_metrics,
    )

    report_path = save_report(html_content, run_id=run_id, output_dir=str(tmp_path))

    assert report_path.exists()
    assert report_path.name == "report.html"
    assert report_path.parent == tmp_path / run_id
    saved_text = report_path.read_text(encoding="utf-8")
    assert saved_text == html_content
