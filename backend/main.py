"""SyntheLoop — FastAPI REST API Service.

Provides endpoints for:
- Health checking (GET /health)
- CSV file upload & EDA analysis (POST /upload)
- Asynchronous feedback loop execution (POST /runs/{run_id}/start)
- Run status & metric progress monitoring (GET /runs/{run_id}/status)
- Artifact download streaming (GET /runs/{run_id}/download/{artifact})
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional
import uuid

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import groq
import pandas as pd
from pydantic import BaseModel, Field

from backend.config import settings
from backend.eda.analyzer import EDAAnalyzer
from backend.evaluator.llm_evaluator import LLMEvaluator
from backend.evaluator.metrics import evaluate_all
from backend.pipeline.audit_trail import AuditTrail
from backend.pipeline.feedback_loop import FeedbackLoop
from backend.planner.llm_planner import LLMPlanner
from backend.reports.report_builder import build_report, save_report

logger = logging.getLogger("syntheloop.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="SyntheLoop API",
    description="Closed-Loop Synthetic Tabular Data Generation Engine",
    version="0.1.0",
)

# Enable CORS for local Streamlit frontend and cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory tracking of active and recent runs
RUN_STATES: dict[str, dict[str, Any]] = {}


class StartRunRequest(BaseModel):
    """Payload to configure and initiate a synthetic data generation run."""

    target_col: Optional[str] = Field(
        default=None,
        description="Target column name for downstream ML utility and class balance evaluation.",
    )
    thresholds: Optional[dict[str, Any]] = Field(
        default=None,
        description="Custom threshold overrides for metrics (defaults to settings.default_thresholds).",
    )
    max_iterations: Optional[int] = Field(
        default=None,
        description="Maximum optimization loop iterations (defaults to settings.max_iterations_default).",
    )


def default_feedback_loop_factory(
    df: pd.DataFrame,
    target_col: Optional[str],
    thresholds: Optional[dict[str, Any]],
    max_iterations: Optional[int],
    audit_trail: AuditTrail,
) -> FeedbackLoop:
    """Instantiates a standard production FeedbackLoop using configured Groq settings."""
    client = groq.Groq(api_key=settings.groq_api_key or "gsk_dummy")
    planner = LLMPlanner(client=client, model=settings.groq_model)
    evaluator = LLMEvaluator(client=client, model=settings.groq_model)

    return FeedbackLoop(
        real_df=df,
        target_col=target_col,
        thresholds=thresholds,
        max_iterations=max_iterations,
        planner=planner,
        evaluator_metrics_fn=evaluate_all,
        llm_evaluator=evaluator,
        audit_trail=audit_trail,
        save_artifacts=True,
    )


# Hook allowing tests to inject mock loop factories
feedback_loop_factory: Callable[..., FeedbackLoop] = default_feedback_loop_factory


def _execute_pipeline_task(
    run_id: str,
    target_col: Optional[str],
    thresholds: Optional[dict[str, Any]],
    max_iterations: Optional[int],
) -> None:
    """Executes the iterative feedback loop as a background task and updates state."""
    logger.info(f"Background task started for run_id '{run_id}'")
    upload_path = Path(settings.upload_dir) / f"{run_id}.csv"

    try:
        real_df = pd.read_csv(upload_path)
        audit_trail = AuditTrail(run_id=run_id, output_dir=settings.output_dir)

        # Pre-compute EDA for final report generation
        eda_summary = EDAAnalyzer(real_df, target_col=target_col).analyze()

        loop = feedback_loop_factory(
            df=real_df,
            target_col=target_col,
            thresholds=thresholds,
            max_iterations=max_iterations,
            audit_trail=audit_trail,
        )

        result = loop.run()

        # Generate HTML report
        trail_entries = audit_trail.get_full_trail()
        final_metrics = result.get("final_metrics")

        try:
            report_html = build_report(
                run_id=run_id,
                eda_summary=eda_summary,
                audit_trail=trail_entries,
                final_metrics=final_metrics,
            )
            save_report(report_html, run_id=run_id, output_dir=settings.output_dir)
        except Exception as report_err:
            logger.warning(f"Could not generate HTML report for run '{run_id}': {report_err}")

        # Update in-memory state
        RUN_STATES[run_id] = {
            "run_id": run_id,
            "status": result.get("status", "completed"),
            "iteration": result.get("iterations_used", len(trail_entries)),
            "latest_metrics": final_metrics,
            "error": result.get("error"),
            "is_running": False,
        }
        logger.info(f"Background task completed for run_id '{run_id}' with status '{result.get('status')}'")

    except Exception as exc:
        logger.error(f"Background task failed for run_id '{run_id}': {exc}", exc_info=True)
        RUN_STATES[run_id] = {
            "run_id": run_id,
            "status": "failed",
            "iteration": 0,
            "latest_metrics": None,
            "error": str(exc),
            "is_running": False,
        }


@app.get("/health", summary="Health check")
def health_check() -> dict[str, str]:
    """Returns application status and service identifier."""
    return {"status": "ok", "app": "SyntheLoop"}


@app.post("/upload", summary="Upload CSV dataset and run EDA")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accepts a CSV upload, validates against methodology constraints, and runs EDA.

    Validation rules:
    - Non-empty file
    - Valid CSV format
    - More than 1 data row
    - At least one usable continuous or categorical column
    """
    if file is None or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided",
        )

    try:
        contents = await file.read()
    except Exception as read_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {read_err}",
        )

    if not contents or len(contents.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid CSV file format: file must be UTF-8 encoded text",
        )

    if "\x00" in text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid CSV file format: binary or null characters detected",
        )

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as parse_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid CSV file format: {parse_err}",
        )

    if len(df) <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset must contain more than 1 row",
        )

    # Check for usable columns (must have at least one column with valid non-empty data)
    usable_cols = [
        c for c in df.columns
        if df[c].notna().sum() > 0 and df[c].dropna().astype(str).str.strip().ne("").sum() > 0
    ]
    if len(usable_cols) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset must contain at least one usable numeric or categorical column",
        )

    # Perform EDA analysis and check column usability
    try:
        analyzer = EDAAnalyzer(df)
        eda_summary = analyzer.analyze()
    except ValueError as eda_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset validation failed: {str(eda_err)}",
        )

    total_usable = len(eda_summary.get("continuous_columns", [])) + len(
        eda_summary.get("categorical_columns", [])
    )
    if total_usable == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset must contain at least one usable numeric or categorical column",
        )

    # Save to disk
    run_id = uuid.uuid4().hex[:8]
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{run_id}.csv"

    try:
        file_path.write_bytes(contents)
    except Exception as write_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist dataset to disk: {write_err}",
        )

    return {
        "run_id": run_id,
        "filename": file.filename,
        "eda_summary": eda_summary,
    }


@app.post("/runs/{run_id}/start", status_code=status.HTTP_202_ACCEPTED, summary="Start optimization run")
def start_run(
    run_id: str,
    request: StartRunRequest = StartRunRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict[str, str]:
    """Starts the closed-loop optimization run in a non-blocking background task."""
    upload_path = Path(settings.upload_dir) / f"{run_id}.csv"
    if not upload_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Uploaded dataset for run_id '{run_id}' not found",
        )

    current_state = RUN_STATES.get(run_id)
    if current_state and current_state.get("is_running", False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Optimization run for run_id '{run_id}' is already running",
        )

    # Initialize in-memory run state
    RUN_STATES[run_id] = {
        "run_id": run_id,
        "status": "running",
        "iteration": 0,
        "latest_metrics": None,
        "error": None,
        "is_running": True,
    }

    background_tasks.add_task(
        _execute_pipeline_task,
        run_id=run_id,
        target_col=request.target_col,
        thresholds=request.thresholds,
        max_iterations=request.max_iterations,
    )

    return {
        "run_id": run_id,
        "status": "started",
        "message": "Optimization run started in background",
    }


@app.get("/runs/{run_id}/status", summary="Get run status & metric progress")
def get_run_status(run_id: str) -> dict[str, Any]:
    """Returns the current execution state, iteration count, and latest metrics."""
    # Check in-memory state first
    state = RUN_STATES.get(run_id)
    if state:
        # If currently running, check live audit trail to return intermediate iteration & metrics
        if state.get("is_running"):
            trail_path = Path(settings.output_dir) / run_id / "audit_trail.json"
            if trail_path.exists():
                try:
                    entries = json.loads(trail_path.read_text(encoding="utf-8"))
                    if entries:
                        latest = entries[-1]
                        state["iteration"] = latest.get("iteration", state["iteration"])
                        state["latest_metrics"] = latest.get("metrics", state["latest_metrics"])
                except Exception:
                    pass
        return state

    # Fallback to persistent disk audit trail
    trail_path = Path(settings.output_dir) / run_id / "audit_trail.json"
    if trail_path.exists():
        try:
            entries = json.loads(trail_path.read_text(encoding="utf-8"))
            if entries:
                latest = entries[-1]
                overall_passed = bool(latest.get("metrics", {}).get("overall_passed", False))
                action = latest.get("action_taken", "")
                final_status = (
                    "failed"
                    if action == "failed"
                    else ("completed_threshold_met" if overall_passed else "completed_max_iterations")
                )
                return {
                    "run_id": run_id,
                    "status": final_status,
                    "iteration": latest.get("iteration", len(entries)),
                    "latest_metrics": latest.get("metrics"),
                    "error": None,
                    "is_running": False,
                }
        except Exception as read_err:
            logger.warning(f"Error reading audit trail for run '{run_id}': {read_err}")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Run '{run_id}' not found",
    )


@app.get("/runs/{run_id}/download/{artifact}", summary="Download generation artifacts")
def download_artifact(run_id: str, artifact: str) -> FileResponse:
    """Streams a generated artifact file for a run.

    Supported artifacts:
    - 'dataset': synthetic.csv
    - 'report': report.html
    - 'audit_trail': audit_trail.json
    """
    valid_artifacts = {
        "dataset": ("synthetic.csv", "text/csv", f"synthetic_{run_id}.csv"),
        "report": ("report.html", "text/html", f"report_{run_id}.html"),
        "audit_trail": ("audit_trail.json", "application/json", f"audit_trail_{run_id}.json"),
    }

    if artifact not in valid_artifacts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid artifact type '{artifact}'. Must be one of: dataset, report, audit_trail",
        )

    file_subpath, media_type, download_name = valid_artifacts[artifact]
    target_file = Path(settings.output_dir) / run_id / file_subpath

    if not target_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{artifact}' for run '{run_id}' not found",
        )

    return FileResponse(
        path=str(target_file),
        media_type=media_type,
        filename=download_name,
    )
