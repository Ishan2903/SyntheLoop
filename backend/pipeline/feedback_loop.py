"""SyntheLoop — Iterative Optimization Feedback Loop.

Coordinates the end-to-end synthetic data generation pipeline:
EDA -> LLM Planning -> CTGAN Training & Sampling -> Metrics Evaluation ->
LLM Feedback & Diagnosis -> Audit Logging -> Convergence Check.
"""

from copy import deepcopy
from datetime import datetime, timezone
import logging
from typing import Any, Callable
import pandas as pd

from backend.config import settings
from backend.eda.analyzer import EDAAnalyzer
from backend.generator.ctgan_wrapper import SyntheticGenerator
from backend.evaluator.metrics import evaluate_all
from backend.evaluator.llm_evaluator import LLMEvaluator
from backend.planner.llm_planner import LLMPlanner
from backend.pipeline.audit_trail import AuditTrail

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Orchestrates iterative CTGAN generation with LLM-guided refinement and quality gates."""

    def __init__(
        self,
        real_df: pd.DataFrame,
        target_col: str | None = None,
        thresholds: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        planner: LLMPlanner | None = None,
        evaluator_metrics_fn: Callable[..., dict[str, Any]] = evaluate_all,
        llm_evaluator: LLMEvaluator | None = None,
        audit_trail: AuditTrail | None = None,
        save_artifacts: bool = True,
    ) -> None:
        """Initializes the FeedbackLoop.

        Args:
            real_df: Real baseline DataFrame.
            target_col: Optional target column for classification utility & balance metrics.
            thresholds: Quality gate thresholds. Defaults to settings.default_thresholds.
            max_iterations: Maximum loop iterations. Defaults to settings.max_iterations_default.
            planner: Configured LLMPlanner instance.
            evaluator_metrics_fn: Function computing Data Contract 4.3 metrics from real & synth DataFrames.
            llm_evaluator: Configured LLMEvaluator instance.
            audit_trail: AuditTrail instance for persisting run iterations.
            save_artifacts: Whether to save synthetic.csv into audit_trail.run_dir upon completion.
        """
        if real_df is None or not isinstance(real_df, pd.DataFrame):
            raise ValueError("real_df must be a valid pandas DataFrame")
        if real_df.empty:
            raise ValueError("real_df cannot be empty")

        self.real_df = real_df
        self.target_col = target_col
        self.thresholds = (
            deepcopy(thresholds) if thresholds is not None else deepcopy(settings.default_thresholds)
        )
        self.max_iterations = (
            int(max_iterations)
            if max_iterations is not None
            else int(settings.max_iterations_default)
        )
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self.planner = planner
        self.evaluator_metrics_fn = evaluator_metrics_fn
        self.llm_evaluator = llm_evaluator
        self.audit_trail = audit_trail
        self.save_artifacts = save_artifacts

    def run(self) -> dict[str, Any]:
        """Runs the iterative optimization loop.

        Returns:
            dict containing:
                status: 'completed_threshold_met' | 'completed_max_iterations' | 'failed'
                final_synth: Generated synthetic DataFrame (or None if failed)
                final_metrics: Latest evaluation metrics dictionary (or None if failed)
                iterations_used: Number of iterations executed
                audit_trail: Full list of logged iteration records (if audit_trail configured)
                error: Error message string (only present if status is 'failed')
        """
        logger.info(
            f"Starting FeedbackLoop (max_iterations={self.max_iterations}, "
            f"target_col={self.target_col})"
        )

        current_iteration = 1
        current_config: dict[str, Any] | None = None
        current_metrics: dict[str, Any] | None = None
        current_feedback: dict[str, Any] | None = None
        synth_df: pd.DataFrame | None = None

        try:
            # 1. Exploratory Data Analysis
            logger.info("Executing initial EDA analysis on real dataset...")
            eda_analyzer = EDAAnalyzer(self.real_df, self.target_col)
            eda_summary = eda_analyzer.analyze()

            prior_feedback: dict[str, Any] | None = None

            # 2. Iteration Loop
            for iteration in range(1, self.max_iterations + 1):
                current_iteration = iteration
                logger.info(f"--- Starting Iteration {iteration}/{self.max_iterations} ---")

                # Plan generation configuration
                if self.planner is None:
                    raise RuntimeError("LLMPlanner is required to run the feedback loop")

                raw_config = self.planner.plan(eda_summary, prior_feedback=prior_feedback)
                # Ensure fresh snapshot to prevent mutation across iterations
                current_config = deepcopy(raw_config)
                logger.info(
                    f"Iteration {iteration} planned config: epochs={current_config.get('epochs')}, "
                    f"batch_size={current_config.get('batch_size')}"
                )

                # Train generator and sample synthetic data
                generator = SyntheticGenerator(current_config)
                generator.fit(self.real_df)
                synth_df = generator.sample(len(self.real_df))

                # Evaluate metrics
                current_metrics = self.evaluator_metrics_fn(
                    real=self.real_df,
                    synth=synth_df,
                    eda_summary=eda_summary,
                    thresholds=self.thresholds,
                    iteration=iteration,
                )

                # LLM Evaluator feedback & diagnosis
                if self.llm_evaluator is None:
                    raise RuntimeError("LLMEvaluator is required to run the feedback loop")

                current_feedback = self.llm_evaluator.evaluate_feedback(
                    metrics=current_metrics, prior_config=current_config
                )

                # Check stopping criteria
                overall_passed = bool(current_metrics.get("overall_passed", False))
                stop_recommended = bool(current_feedback.get("stop_recommended", False))
                threshold_met = overall_passed or stop_recommended

                if threshold_met:
                    action_taken = "stopped_threshold_met"
                elif iteration == self.max_iterations:
                    action_taken = "stopped_max_iterations"
                else:
                    action_taken = "continued"

                # Log iteration to audit trail
                if self.audit_trail:
                    self.audit_trail.log_iteration(
                        {
                            "run_id": self.audit_trail.run_id,
                            "iteration": iteration,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "config_used": deepcopy(current_config),
                            "metrics": deepcopy(current_metrics),
                            "feedback": deepcopy(current_feedback),
                            "action_taken": action_taken,
                        }
                    )

                # Termination check
                if threshold_met:
                    logger.info(
                        f"Iteration {iteration}: Quality threshold met or early stop recommended. "
                        f"Stopping loop."
                    )
                    self._save_final_artifacts(synth_df)
                    return {
                        "status": "completed_threshold_met",
                        "final_synth": synth_df,
                        "final_metrics": current_metrics,
                        "iterations_used": iteration,
                        "audit_trail": (
                            self.audit_trail.get_full_trail() if self.audit_trail else []
                        ),
                    }

                # Prepare prior_feedback for next iteration
                prior_feedback = deepcopy(current_feedback)

            # Max iterations reached without meeting thresholds
            logger.info(f"FeedbackLoop finished: reached max iterations ({self.max_iterations}).")
            self._save_final_artifacts(synth_df)
            return {
                "status": "completed_max_iterations",
                "final_synth": synth_df,
                "final_metrics": current_metrics,
                "iterations_used": self.max_iterations,
                "audit_trail": (self.audit_trail.get_full_trail() if self.audit_trail else []),
            }

        except Exception as exc:
            error_msg = f"FeedbackLoop failed at iteration {current_iteration}: {str(exc)}"
            logger.error(error_msg, exc_info=True)

            # Log failure to audit trail
            if self.audit_trail:
                try:
                    self.audit_trail.log_iteration(
                        {
                            "run_id": self.audit_trail.run_id,
                            "iteration": current_iteration,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "config_used": deepcopy(current_config) if current_config else {},
                            "metrics": deepcopy(current_metrics) if current_metrics else {},
                            "feedback": deepcopy(current_feedback) if current_feedback else None,
                            "action_taken": "failed",
                        }
                    )
                except Exception as log_exc:
                    logger.error(f"Failed to log failure event to audit trail: {log_exc}")

            return {
                "status": "failed",
                "error": str(exc),
                "iterations_used": current_iteration,
                "final_synth": synth_df,
                "final_metrics": current_metrics,
                "audit_trail": (self.audit_trail.get_full_trail() if self.audit_trail else []),
            }

    def _save_final_artifacts(self, synth_df: pd.DataFrame | None) -> None:
        """Saves synthetic DataFrame to CSV in run directory if configured."""
        if not self.save_artifacts or self.audit_trail is None or synth_df is None:
            return

        try:
            csv_path = self.audit_trail.run_dir / "synthetic.csv"
            synth_df.to_csv(csv_path, index=False)
            logger.info(f"Saved synthetic dataset to {csv_path}")
        except Exception as exc:
            logger.warning(f"Could not save synthetic dataset artifact: {exc}")
