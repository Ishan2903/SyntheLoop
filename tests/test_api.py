"""Tests for SyntheLoop FastAPI REST API (backend/main.py).

Verifies:
- Health check endpoint
- CSV dataset upload & validation (empty, single-row, unparseable, valid)
- Run start & background feedback loop execution
- Run status query & disk audit trail recovery
- Artifact downloading (dataset, report, audit_trail, invalid, missing)
"""

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pandas as pd
import pytest

import backend.main as main_module
from backend.main import RUN_STATES, app
from backend.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Provides a TestClient with isolated upload and output directories."""
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "output_dir", str(output_dir))

    # Clear in-memory run states
    RUN_STATES.clear()

    return TestClient(app)


@pytest.fixture
def sample_csv_bytes():
    """Generates valid CSV file bytes for testing."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 45, 50, 35, 40],
            "income": [50000, 60000, 80000, 95000, 70000, 75000],
            "city": ["NYC", "LA", "NYC", "Chicago", "LA", "Chicago"],
            "target": [0, 1, 1, 1, 0, 1],
        }
    )
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def test_health_endpoint(client):
    """GET /health should return 200 OK with app info."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app": "SyntheLoop"}


def test_upload_valid_csv(client, sample_csv_bytes):
    """POST /upload with valid CSV should return 200, run_id, and EDA summary."""
    files = {"file": ("test_data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["filename"] == "test_data.csv"
    assert "eda_summary" in data
    assert data["eda_summary"]["n_rows"] == 6
    assert data["eda_summary"]["n_cols"] == 4

    # Verify file saved on disk
    saved_file = Path(settings.upload_dir) / f"{data['run_id']}.csv"
    assert saved_file.exists()


def test_upload_empty_file(client):
    """POST /upload with empty file should return 400 Bad Request."""
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    response = client.post("/upload", files=files)

    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_upload_single_row(client):
    """POST /upload with single row CSV should return 400 Bad Request."""
    single_row_csv = b"col1,col2\nval1,val2\n"
    files = {"file": ("single.csv", io.BytesIO(single_row_csv), "text/csv")}
    response = client.post("/upload", files=files)

    assert response.status_code == 400
    assert "Dataset must contain more than 1 row" in response.json()["detail"]


def test_upload_corrupt_csv(client):
    """POST /upload with non-csv data should return 400 Bad Request."""
    corrupt_bytes = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    files = {"file": ("corrupt.csv", io.BytesIO(corrupt_bytes), "application/octet-stream")}
    response = client.post("/upload", files=files)

    assert response.status_code == 400
    assert "Invalid CSV file format" in response.json()["detail"]


def test_upload_no_usable_columns(client):
    """POST /upload with only unparseable/empty columns should return 400."""
    all_null_csv = b"col1,col2\n,\n,\n"
    files = {"file": ("nulls.csv", io.BytesIO(all_null_csv), "text/csv")}
    response = client.post("/upload", files=files)

    assert response.status_code == 400
    assert "usable numeric or categorical column" in response.json()["detail"]


def test_start_run_not_found(client):
    """POST /runs/{run_id}/start for nonexistent file should return 404."""
    response = client.post("/runs/unknown_123/start", json={})
    assert response.status_code == 404
    assert "Uploaded dataset for run_id 'unknown_123' not found" in response.json()["detail"]


def test_start_run_and_status_flow(client, sample_csv_bytes, monkeypatch):
    """Full lifecycle: upload, start run with mock loop, and check status."""
    # 1. Upload CSV
    files = {"file": ("sample.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    upload_res = client.post("/upload", files=files)
    run_id = upload_res.json()["run_id"]

    # 2. Mock FeedbackLoop factory
    mock_loop = MagicMock()
    mock_metrics = {
        "overall_passed": True,
        "iteration": 1,
        "ks_statistic": {"overall_score": 0.05, "passed": True},
        "correlation": {"overall_score": 0.08, "passed": True},
        "js_divergence": {"overall_score": 0.02, "passed": True},
        "dcr": {"overall_score": 12.0, "passed": True},
        "utility": {"overall_score": 0.03, "passed": True},
    }
    mock_synth_df = pd.DataFrame({"age": [26, 31], "income": [51000, 61000]})

    def fake_run():
        # Write synthetic.csv and audit_trail.json to simulate real FeedbackLoop artifact saving
        run_dir = Path(settings.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        mock_synth_df.to_csv(run_dir / "synthetic.csv", index=False)
        trail_data = [
            {
                "run_id": run_id,
                "iteration": 1,
                "timestamp": "2026-09-05T12:00:00Z",
                "config_used": {"epochs": 100},
                "metrics": mock_metrics,
                "feedback": {"weak_areas": [], "stop_recommended": True},
                "action_taken": "stop",
            }
        ]
        (run_dir / "audit_trail.json").write_text(json.dumps(trail_data), encoding="utf-8")
        return {
            "status": "completed_threshold_met",
            "final_synth": mock_synth_df,
            "final_metrics": mock_metrics,
            "iterations_used": 1,
            "audit_trail": trail_data,
        }

    mock_loop.run.side_effect = fake_run
    monkeypatch.setattr(main_module, "feedback_loop_factory", lambda **kwargs: mock_loop)

    # 3. Start run
    payload = {"target_col": "target", "max_iterations": 2}
    start_res = client.post(f"/runs/{run_id}/start", json=payload)
    assert start_res.status_code == 202
    assert start_res.json()["status"] == "started"

    # Test conflict if starting already running run
    RUN_STATES[run_id]["is_running"] = True
    conflict_res = client.post(f"/runs/{run_id}/start", json=payload)
    assert conflict_res.status_code == 409
    RUN_STATES[run_id]["is_running"] = False

    # 4. Status query
    status_res = client.get(f"/runs/{run_id}/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["run_id"] == run_id
    assert status_data["status"] == "completed_threshold_met"
    assert status_data["iteration"] == 1
    assert status_data["latest_metrics"]["overall_passed"] is True
    assert status_data["is_running"] is False

    # Verify report was generated
    report_file = Path(settings.output_dir) / run_id / "report.html"
    assert report_file.exists()


def test_get_status_from_disk(client):
    """GET /runs/{run_id}/status recovers status from disk if in-memory state is empty."""
    run_id = "disk_recovered_run"
    run_dir = Path(settings.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trail_data = [
        {
            "run_id": run_id,
            "iteration": 2,
            "timestamp": "2026-09-05T12:00:00Z",
            "config_used": {},
            "metrics": {"overall_passed": True, "ks_statistic": {"overall_score": 0.05}},
            "feedback": None,
            "action_taken": "continue",
        }
    ]
    (run_dir / "audit_trail.json").write_text(json.dumps(trail_data), encoding="utf-8")

    # In-memory RUN_STATES is empty
    assert run_id not in RUN_STATES

    res = client.get(f"/runs/{run_id}/status")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == run_id
    assert data["status"] == "completed_threshold_met"
    assert data["iteration"] == 2
    assert data["is_running"] is False


def test_get_status_nonexistent_run(client):
    """GET /runs/{run_id}/status returns 404 if run does not exist anywhere."""
    res = client.get("/runs/completely_fake/status")
    assert res.status_code == 404
    assert "Run 'completely_fake' not found" in res.json()["detail"]


def test_download_valid_artifacts(client):
    """GET /runs/{run_id}/download/{artifact} streams dataset, report, and audit_trail."""
    run_id = "download_test_run"
    run_dir = Path(settings.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "synthetic.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (run_dir / "report.html").write_text("<html><body>Report</body></html>", encoding="utf-8")
    (run_dir / "audit_trail.json").write_text("[]", encoding="utf-8")

    # 1. Download dataset
    res_ds = client.get(f"/runs/{run_id}/download/dataset")
    assert res_ds.status_code == 200
    assert "text/csv" in res_ds.headers["content-type"]
    assert "a,b" in res_ds.text

    # 2. Download report
    res_rep = client.get(f"/runs/{run_id}/download/report")
    assert res_rep.status_code == 200
    assert "text/html" in res_rep.headers["content-type"]
    assert "Report" in res_rep.text

    # 3. Download audit trail
    res_at = client.get(f"/runs/{run_id}/download/audit_trail")
    assert res_at.status_code == 200
    assert "application/json" in res_at.headers["content-type"]
    assert res_at.text == "[]"


def test_download_invalid_artifact_type(client):
    """GET /runs/{run_id}/download/{artifact} returns 400 for invalid artifact name."""
    res = client.get("/runs/any_run/download/invalid_artifact")
    assert res.status_code == 400
    assert "Invalid artifact type" in res.json()["detail"]


def test_download_missing_artifact_file(client):
    """GET /runs/{run_id}/download/{artifact} returns 404 if file does not exist on disk."""
    run_id = "missing_artifact_run"
    run_dir = Path(settings.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # File not created

    res = client.get(f"/runs/{run_id}/download/dataset")
    assert res.status_code == 404
    assert "Artifact 'dataset' for run 'missing_artifact_run' not found" in res.json()["detail"]


def test_background_task_failure_handling(client, sample_csv_bytes, monkeypatch):
    """Background task catches exceptions and marks status as 'failed'."""
    files = {"file": ("fail_sample.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    upload_res = client.post("/upload", files=files)
    run_id = upload_res.json()["run_id"]

    mock_loop = MagicMock()
    mock_loop.run.side_effect = RuntimeError("Simulated crash in CTGAN fitting")
    monkeypatch.setattr(main_module, "feedback_loop_factory", lambda **kwargs: mock_loop)

    start_res = client.post(f"/runs/{run_id}/start", json={})
    assert start_res.status_code == 202

    status_res = client.get(f"/runs/{run_id}/status")
    assert status_res.status_code == 200
    data = status_res.json()
    assert data["status"] == "failed"
    assert "Simulated crash in CTGAN fitting" in data["error"]
    assert data["is_running"] is False
