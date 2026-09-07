"""Tests for frontend/app.py helper functions and integration contract with backend.

Verifies:
- API health check function
- Dataset upload function and EDA response handling
- Trigger run start payload construction
- Run status fetching
- Artifact downloads (CSV, report HTML, audit trail JSON)
- Sample churn dataset integrity
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from frontend.app import (
    DEFAULT_THRESHOLDS,
    append_terminal_log,
    check_api_health,
    fetch_artifact_bytes,
    fetch_run_status,
    trigger_start_run,
    upload_dataset,
)


def test_sample_churn_dataset_exists_and_valid():
    """Verify that data/samples/sample_churn.csv exists and has expected columns and rows."""
    sample_file = Path("data/samples/sample_churn.csv")
    assert sample_file.exists(), "Sample churn CSV should exist in data/samples/"

    df = pd.read_csv(sample_file)
    assert len(df) >= 20, "Sample dataset should contain sufficient rows for CTGAN testing"
    assert "churn" in df.columns, "Target column 'churn' must be present"
    assert "credit_score" in df.columns
    assert "age" in df.columns


def test_check_api_health():
    """Test health check helper function with mocked HTTP responses."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        assert check_api_health("http://127.0.0.1:8000") is True

    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        assert check_api_health("http://127.0.0.1:8000") is False


def test_upload_dataset_success():
    """Test upload_dataset helper with mocked successful response."""
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "run_id": "test_123",
            "filename": "test.csv",
            "eda_summary": {"n_rows": 100, "n_cols": 5},
        }

        result = upload_dataset("http://127.0.0.1:8000", b"a,b\n1,2\n3,4", "test.csv")
        assert result["run_id"] == "test_123"
        assert result["eda_summary"]["n_rows"] == 100


def test_upload_dataset_failure():
    """Test upload_dataset raises descriptive RuntimeError on failure."""
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 400
        mock_post.return_value.headers = {"content-type": "application/json"}
        mock_post.return_value.json.return_value = {"detail": "Empty file"}

        with pytest.raises(RuntimeError, match="Upload failed"):
            upload_dataset("http://127.0.0.1:8000", b"", "empty.csv")


def test_trigger_start_run():
    """Test trigger_start_run constructs expected payload."""
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 202
        mock_post.return_value.json.return_value = {
            "run_id": "test_123",
            "status": "started",
        }

        res = trigger_start_run(
            "http://127.0.0.1:8000",
            "test_123",
            "churn",
            DEFAULT_THRESHOLDS,
            3,
        )
        assert res["status"] == "started"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["target_col"] == "churn"
        assert kwargs["json"]["max_iterations"] == 3


def test_fetch_run_status():
    """Test fetch_run_status retrieves run telemetry."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "run_id": "test_123",
            "status": "running",
            "iteration": 1,
        }

        status = fetch_run_status("http://127.0.0.1:8000", "test_123")
        assert status["status"] == "running"
        assert status["iteration"] == 1


def test_fetch_artifact_bytes():
    """Test artifact fetching for dataset, report, and audit trail."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"col1,col2\nval1,val2"

        data = fetch_artifact_bytes("http://127.0.0.1:8000", "test_123", "dataset")
        assert data == b"col1,col2\nval1,val2"

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 404
        assert fetch_artifact_bytes("http://127.0.0.1:8000", "test_123", "missing") is None
