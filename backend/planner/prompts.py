"""Planner prompts module for SyntheLoop.

Provides the system prompt and user-turn prompt construction for LLM-based
CTGAN hyperparameter planning based on EDA summary and iterative evaluator feedback.
"""

import json
from typing import Any, Dict, Optional

PLANNER_SYSTEM_PROMPT = """You are a configuration planner for a CTGAN-based
synthetic data generator. You will be given a JSON summary of a dataset's
exploratory data analysis. Respond with ONLY a JSON object matching this
exact schema, no prose, no markdown fences:
{
  "categorical_columns": [list of column names from the input, exactly as spelled],
  "epochs": integer between 50 and 500,
  "batch_size": integer, multiple of 10, between 50 and 1000,
  "generator_dim": [int, int],
  "discriminator_dim": [int, int],
  "pac": integer between 1 and 20,
  "reasoning": "one sentence"
}
Rules:
- categorical_columns MUST be a subset of the "categorical_columns" list
  already present in the input JSON. Never invent a column name not present
  in the input.
- Base epochs/batch_size on n_rows: smaller datasets (<5000 rows) need
  fewer epochs to avoid overfitting; larger datasets can use more.
- If this is a refinement call (input includes prior feedback), adjust
  only the parameters the feedback flags as weak."""


def build_planner_prompt(eda_summary: Dict[str, Any], prior_feedback: Optional[Dict[str, Any]] = None) -> str:
    """Serializes eda_summary (and prior_feedback if present) to a compact JSON string

    and returns the full user-turn prompt text. Never includes raw dataset rows —
    only the EDA summary JSON.
    """
    payload: Dict[str, Any] = {
        "eda_summary": eda_summary
    }

    if prior_feedback is not None:
        payload["prior_feedback"] = prior_feedback
        instruction = (
            "This is a refinement iteration. Review the prior evaluator feedback above "
            "and adjust only the CTGAN configuration parameters flagged as weak. "
            "Return ONLY a JSON object matching Data Contract 4.2."
        )
    else:
        instruction = (
            "Analyze the EDA summary above and propose an initial CTGAN configuration. "
            "Return ONLY a JSON object matching Data Contract 4.2."
        )

    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    return f"{instruction}\n\nDataset Input:\n{json_str}"
