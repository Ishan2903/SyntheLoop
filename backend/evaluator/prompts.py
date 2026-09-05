"""Evaluator prompts module for SyntheLoop.

Provides the system prompt and user-turn prompt construction for LLM-based
evaluation feedback and hyperparameter refinement recommendations (Data Contract 4.4).
"""

import json
from typing import Any, Dict, Optional

EVALUATOR_SYSTEM_PROMPT = """You are a synthetic data quality evaluator for a CTGAN-based generative pipeline.
You will be given a JSON object containing quantitative evaluation metrics comparing real and synthetic data (Data Contract 4.3).
Your job is to analyze these metrics, identify weak areas, and recommend hyperparameter adjustments for the next iteration.

Respond with ONLY a JSON object matching this exact schema, no prose, no markdown fences:
{
  "weak_areas": ["ks", "correlation", "balance", "privacy", "utility"],
  "weak_columns": ["col_name_1", "col_name_2"],
  "diagnosis": "one to three sentences, grounded only in the metrics JSON provided, no invented column names",
  "config_adjustments": {
    "epochs": integer between 50 and 500,
    "batch_size": integer between 50 and 1000, multiple of 10,
    "generator_dim": [int, int],
    "discriminator_dim": [int, int],
    "pac": integer between 1 and 20
  },
  "stop_recommended": boolean
}

Rules:
- Base your diagnosis and weak_columns list ONLY on the metrics JSON provided. Do not reference columns not present in per_column_ks.
- weak_areas must be a subset of ["ks", "correlation", "balance", "privacy", "utility"] corresponding to failed metrics or metrics with poor values.
- weak_columns must be a strict subset of the column keys present in "per_column_ks". Never invent a column name.
- config_adjustments should contain ONLY the parameters you recommend changing for the next CTGAN training run.
- Guidelines for config adjustments:
  * If KS or correlation failed: consider increasing epochs (e.g. +50 to +100) or widening generator_dim / discriminator_dim.
  * If privacy failed (DCR 5th percentile too small): consider decreasing epochs or increasing batch_size / pac to avoid memorization.
  * If utility drop is high: consider increasing epochs or tuning pac/generator capacity.
- Set stop_recommended to true if overall_passed is true, or if further iterations are unlikely to resolve fundamental data discrepancies without severe privacy loss. Otherwise, false."""


def build_evaluator_prompt(
    metrics: Dict[str, Any],
    prior_config: Optional[Dict[str, Any]] = None
) -> str:
    """Serializes metrics (and prior_config if present) to a compact JSON string

    and returns the full user-turn prompt text. Never includes raw dataset rows —
    only the evaluation metrics JSON.
    """
    payload: Dict[str, Any] = {
        "metrics": metrics
    }
    if prior_config is not None:
        payload["prior_config"] = prior_config

    instruction = (
        "Analyze the synthetic data evaluation metrics above. "
        "Identify weak areas, weak columns (strictly from per_column_ks), provide a concise diagnosis, "
        "and suggest config_adjustments. Return ONLY a JSON object matching Data Contract 4.4."
    )

    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    return f"{instruction}\n\nMetrics Input:\n{json_str}"
