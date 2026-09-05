"""LLM Evaluator implementation for SyntheLoop.

Interacts with the Groq API to analyze evaluation metrics (Data Contract 4.3),
diagnose deficiencies without hallucinating column names, and propose hyperparameter
refinements (Data Contract 4.4) with rule-based fallback and rate-limit backoff.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import groq

from backend.evaluator.prompts import EVALUATOR_SYSTEM_PROMPT, build_evaluator_prompt

logger = logging.getLogger(__name__)

ALLOWED_WEAK_AREAS = {"ks", "correlation", "balance", "privacy", "utility"}


class LLMEvaluator:
    """Evaluates synthetic data metrics and produces refinement feedback using Groq LLMs."""

    def __init__(self, client: Any, model: str):
        """Initializes the evaluator with a Groq client and model name.

        Args:
            client: groq.Groq client instance (or compatible mock).
            model: Name of the Groq model (e.g. 'llama-3.3-70b-versatile').
        """
        self.client = client
        self.model = model

    def evaluate_feedback(
        self,
        metrics: Dict[str, Any],
        prior_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calls the Groq API to produce feedback matching Data Contract 4.4.

        Retries once on JSON parse failure or schema validation failure with corrective
        instructions. On repeated failure, falls back to a rule-based default without crashing.

        Args:
            metrics: Output from evaluate_all() (Data Contract 4.3).
            prior_config: Optional CTGAN configuration used in the evaluated iteration.

        Returns:
            Dict matching Data Contract 4.4.
        """
        user_prompt = build_evaluator_prompt(metrics, prior_config)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_content = ""
        last_error = ""

        # Attempt 1
        try:
            raw_content = self._call_llm_with_retry(messages)
            feedback = self._parse_and_validate(raw_content, metrics)
            return feedback
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Attempt 1 of LLM evaluator feedback failed: %s. Retrying once with corrective prompt...",
                exc
            )

        # Attempt 2 (Retry with added corrective instruction)
        allowed_cols = sorted(list(metrics.get("per_column_ks", {}).keys()))
        retry_prompt = (
            "Your previous response failed validation with the following error:\n"
            f"{last_error}\n\n"
            "Please output ONLY a valid JSON object matching Data Contract 4.4 with no markdown "
            "fences or extra prose.\n"
            "Strict constraints:\n"
            f"- 'weak_columns' must be a strict subset of available columns: {allowed_cols}. NEVER invent column names.\n"
            "- 'weak_areas' must only contain items from: ['ks', 'correlation', 'balance', 'privacy', 'utility'].\n"
            "- 'diagnosis' must be 1-3 sentences grounded strictly in the provided metrics JSON.\n"
            "- 'config_adjustments' must be a dict containing only valid CTGAN parameters (e.g. epochs between 50 and 500).\n"
            "- 'stop_recommended' must be a boolean."
        )

        retry_messages: List[Dict[str, str]] = [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_content or "{}"},
            {"role": "user", "content": retry_prompt},
        ]

        try:
            raw_content_2 = self._call_llm_with_retry(retry_messages)
            feedback = self._parse_and_validate(raw_content_2, metrics)
            return feedback
        except Exception as exc_retry:
            logger.warning(
                "Attempt 2 of LLM evaluator feedback failed: %s. Falling back to rule-based feedback.",
                exc_retry
            )
            return self._default_feedback(metrics, prior_config)

    def _call_llm_with_retry(
        self,
        messages: List[Dict[str, str]],
        use_json_mode: bool = True
    ) -> str:
        """Invokes the Groq chat completion API with rate-limit (HTTP 429) backoff.

        Args:
            messages: List of chat message dicts.
            use_json_mode: Whether to pass response_format={'type': 'json_object'}.

        Returns:
            The raw text string response from the model.
        """
        max_rate_limit_retries = 3
        for attempt in range(max_rate_limit_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                }
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return content or ""
            except Exception as exc:
                is_rate_limit = (
                    (hasattr(groq, "RateLimitError") and isinstance(exc, groq.RateLimitError))
                    or getattr(exc, "status_code", None) == 429
                    or "429" in str(exc)
                    or "rate limit" in str(exc).lower()
                )
                if is_rate_limit and attempt < max_rate_limit_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(
                        "Rate limit encountered in LLM call. Backing off for %d seconds (attempt %d/%d)...",
                        delay,
                        attempt + 1,
                        max_rate_limit_retries
                    )
                    time.sleep(delay)
                    continue

                if use_json_mode and (
                    "response_format" in str(exc).lower()
                    or "json_object" in str(exc).lower()
                ):
                    logger.warning(
                        "Model '%s' rejected response_format={'type': 'json_object'}. Retrying without json_mode.",
                        self.model
                    )
                    return self._call_llm_with_retry(messages, use_json_mode=False)

                raise exc

        raise RuntimeError("Exhausted rate limit retries for LLM call.")

    def _parse_and_validate(self, text: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Strips markdown fences, parses JSON, and validates against Data Contract 4.4."""
        stripped = self._strip_markdown_fences(text)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse response as JSON: {err}. Raw response: {text[:200]}") from err

        if not isinstance(parsed, dict):
            raise ValueError(f"Response must be a JSON object, got {type(parsed).__name__}")

        valid, reason = self._validate_schema(parsed, metrics)
        if not valid:
            raise ValueError(f"Schema validation failed: {reason}")

        return parsed

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        """Strips markdown ```json and ``` code block wrappers."""
        text = content.strip()
        fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _validate_schema(feedback: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates adherence to Data Contract 4.4 and anti-hallucination rules."""
        required_keys = ["weak_areas", "weak_columns", "diagnosis", "config_adjustments", "stop_recommended"]
        for key in required_keys:
            if key not in feedback:
                return False, f"Missing required field: '{key}'"

        # 1. weak_areas
        weak_areas = feedback.get("weak_areas")
        if not isinstance(weak_areas, list):
            return False, f"'weak_areas' must be a list, got {type(weak_areas).__name__}"
        for area in weak_areas:
            if not isinstance(area, str) or area.lower() not in ALLOWED_WEAK_AREAS:
                return False, f"Invalid weak area '{area}'. Allowed: {sorted(list(ALLOWED_WEAK_AREAS))}"

        # 2. weak_columns (Anti-hallucination per NFR-7 / Section 5.6)
        weak_cols = feedback.get("weak_columns")
        if not isinstance(weak_cols, list):
            return False, f"'weak_columns' must be a list, got {type(weak_cols).__name__}"

        allowed_cols = set(metrics.get("per_column_ks", {}).keys())
        for col in weak_cols:
            if not isinstance(col, str):
                return False, f"All items in 'weak_columns' must be strings, got {col}"
            if col not in allowed_cols:
                return False, (
                    f"Hallucinated weak column '{col}' not present in per_column_ks columns: {sorted(list(allowed_cols))}"
                )

        # 3. diagnosis
        diagnosis = feedback.get("diagnosis")
        if not isinstance(diagnosis, str) or len(diagnosis.strip()) == 0:
            return False, "'diagnosis' must be a non-empty string"

        # 4. config_adjustments
        adjustments = feedback.get("config_adjustments")
        if not isinstance(adjustments, dict):
            return False, f"'config_adjustments' must be a dict, got {type(adjustments).__name__}"

        if "epochs" in adjustments:
            epochs = adjustments["epochs"]
            if isinstance(epochs, bool) or not isinstance(epochs, int) or not (50 <= epochs <= 500):
                return False, f"'epochs' adjustment must be an integer between 50 and 500, got {epochs}"

        if "batch_size" in adjustments:
            bs = adjustments["batch_size"]
            if isinstance(bs, bool) or not isinstance(bs, int) or not (50 <= bs <= 1000) or bs % 10 != 0:
                return False, f"'batch_size' adjustment must be a multiple of 10 between 50 and 1000, got {bs}"

        if "generator_dim" in adjustments:
            gdim = adjustments["generator_dim"]
            if not isinstance(gdim, (list, tuple)) or len(gdim) < 1 or any(
                isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in gdim
            ):
                return False, f"'generator_dim' adjustment must be a list of positive integers, got {gdim}"

        if "discriminator_dim" in adjustments:
            ddim = adjustments["discriminator_dim"]
            if not isinstance(ddim, (list, tuple)) or len(ddim) < 1 or any(
                isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in ddim
            ):
                return False, f"'discriminator_dim' adjustment must be a list of positive integers, got {ddim}"

        if "pac" in adjustments:
            pac = adjustments["pac"]
            if isinstance(pac, bool) or not isinstance(pac, int) or not (1 <= pac <= 20):
                return False, f"'pac' adjustment must be an integer between 1 and 20, got {pac}"

        # 5. stop_recommended
        stop_rec = feedback.get("stop_recommended")
        if not isinstance(stop_rec, bool):
            return False, f"'stop_recommended' must be a boolean, got {type(stop_rec).__name__}"

        return True, None

    def _default_feedback(
        self,
        metrics: Dict[str, Any],
        prior_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deterministic rule-based fallback feedback per Section 5.6.

        Flags whichever single metric is furthest from its threshold and recommends +50 epochs.
        """
        overall_passed = metrics.get("overall_passed", False)
        if overall_passed:
            return {
                "weak_areas": [],
                "weak_columns": [],
                "diagnosis": "All quality and privacy thresholds met.",
                "config_adjustments": {},
                "stop_recommended": True,
            }

        passed_dict = metrics.get("passed", {})
        thresholds = metrics.get("thresholds", {})
        per_column_ks = metrics.get("per_column_ks", {})

        # Compute shortfall for each metric
        deficits: Dict[str, float] = {}

        # 1. KS
        ks_thresh = thresholds.get("ks_stat_max", 0.15)
        mean_ks = sum(per_column_ks.values()) / max(len(per_column_ks), 1) if per_column_ks else 0.0
        deficits["ks"] = max(0.0, mean_ks - ks_thresh) if not passed_dict.get("ks", True) else 0.0

        # 2. Correlation
        corr_thresh = thresholds.get("corr_diff_max", 0.20)
        corr_val = metrics.get("correlation_diff_frobenius", 0.0)
        deficits["correlation"] = max(0.0, corr_val - corr_thresh) if not passed_dict.get("correlation", True) else 0.0

        # 3. Class balance
        js_thresh = thresholds.get("js_divergence_max", 0.10)
        js_val = metrics.get("class_balance_js_divergence", 0.0)
        deficits["balance"] = max(0.0, js_val - js_thresh) if not passed_dict.get("balance", True) else 0.0

        # 4. Privacy
        dcr_thresh = thresholds.get("dcr_min_percentile", 5.0)
        dcr_val = metrics.get("privacy_dcr_5th_percentile", 0.0)
        deficits["privacy"] = max(0.0, dcr_thresh - dcr_val) if not passed_dict.get("privacy", True) else 0.0

        # 5. Utility
        util_thresh = thresholds.get("utility_auc_drop_max", 0.10)
        auc_drop = metrics.get("utility", {}).get("auc_drop", 0.0)
        deficits["utility"] = max(0.0, auc_drop - util_thresh) if not passed_dict.get("utility", True) else 0.0

        # Determine primary weak area (failed metric with largest shortfall, or any failed)
        failed_areas = [k for k, v in passed_dict.items() if not v and k in ALLOWED_WEAK_AREAS]
        if deficits:
            primary_weak = max(deficits.keys(), key=lambda k: deficits[k])
            if deficits[primary_weak] == 0.0 and failed_areas:
                primary_weak = failed_areas[0]
        elif failed_areas:
            primary_weak = failed_areas[0]
        else:
            primary_weak = "correlation"

        weak_areas = [primary_weak]

        # Identify weak columns from KS values exceeding threshold
        weak_cols = [
            col for col, stat in per_column_ks.items()
            if stat > ks_thresh
        ]
        if not weak_cols and per_column_ks and primary_weak == "ks":
            worst_col = max(per_column_ks.keys(), key=lambda c: per_column_ks[c])
            weak_cols = [worst_col]

        # Config adjustment: recommend +50 epochs bounded by [50, 500]
        current_epochs = 150
        if prior_config and "epochs" in prior_config and isinstance(prior_config["epochs"], int):
            current_epochs = prior_config["epochs"]

        new_epochs = min(500, max(50, current_epochs + 50))

        return {
            "weak_areas": weak_areas,
            "weak_columns": weak_cols,
            "diagnosis": (
                f"Rule-based fallback: metric '{primary_weak}' is furthest from threshold. "
                f"Increasing training epochs to {new_epochs} to improve distribution convergence."
            ),
            "config_adjustments": {
                "epochs": new_epochs
            },
            "stop_recommended": False,
        }
