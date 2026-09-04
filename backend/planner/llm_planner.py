"""LLM Planner implementation for SyntheLoop.

Interacts with the Groq API to plan CTGAN hyperparameter configurations
from EDA summaries and feedback, enforcing Data Contract 4.2 and NFR-7 anti-hallucination.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import groq

from backend.planner.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt

logger = logging.getLogger(__name__)


class LLMPlanner:
    """Plans CTGAN hyperparameters using Groq LLMs based on EDA summary and feedback."""

    def __init__(self, client: Any, model: str):
        """Initializes the planner with a Groq client and model name.

        Args:
            client: groq.Groq client instance (or compatible mock).
            model: Name of the Groq model (e.g. 'llama-3.3-70b-versatile').
        """
        self.client = client
        self.model = model

    def plan(
        self,
        eda_summary: Dict[str, Any],
        prior_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calls the Groq API to generate a CTGAN configuration matching Data Contract 4.2.

        Retries once on JSON parse failure or schema validation failure with corrective
        instructions. On repeated failure, falls back to a safe default config without crashing.

        Args:
            eda_summary: Output from EDAAnalyzer.analyze() (Data Contract 4.1).
            prior_feedback: Optional feedback from previous iteration (Data Contract 4.4).

        Returns:
            Dict matching Data Contract 4.2.
        """
        user_prompt = build_planner_prompt(eda_summary, prior_feedback)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Attempt 1
        raw_content = ""
        last_error = ""
        try:
            raw_content = self._call_llm_with_retry(messages)
            config = self._parse_and_validate(raw_content, eda_summary)
            return config
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Attempt 1 of LLM planning failed: %s. Retrying once with corrective prompt...",
                exc
            )

        # Attempt 2 (Retry with added corrective instruction)
        retry_prompt = (
            "Your previous response failed validation with the following error:\n"
            f"{last_error}\n\n"
            "Please output ONLY a valid JSON object matching Data Contract 4.2 with no markdown "
            "fences or extra prose.\n"
            "Requirements:\n"
            "- categorical_columns MUST be a strict subset of the input's categorical_columns.\n"
            "- epochs must be an integer between 50 and 500.\n"
            "- batch_size must be a multiple of 10 between 50 and 1000.\n"
            "- generator_dim and discriminator_dim must each be a list of 2 integers.\n"
            "- pac must be an integer between 1 and 20.\n"
            "- reasoning must be a one-sentence string."
        )

        retry_messages: List[Dict[str, str]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_content or "{}"},
            {"role": "user", "content": retry_prompt},
        ]

        try:
            raw_content_2 = self._call_llm_with_retry(retry_messages)
            config = self._parse_and_validate(raw_content_2, eda_summary)
            return config
        except Exception as exc_retry:
            logger.warning(
                "Attempt 2 of LLM planning failed: %s. Falling back to default configuration.",
                exc_retry
            )
            return self._default_config(eda_summary)

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

                # Fallback if the specific model rejects json_object response_format
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

    def _parse_and_validate(self, text: str, eda_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Strips markdown fences, parses JSON, and validates against Data Contract 4.2."""
        stripped = self._strip_markdown_fences(text)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse response as JSON: {err}. Raw response: {text[:200]}") from err

        if not isinstance(parsed, dict):
            raise ValueError(f"Response must be a JSON object, got {type(parsed).__name__}")

        valid, reason = self._validate_schema(parsed, eda_summary)
        if not valid:
            raise ValueError(f"Schema validation failed: {reason}")

        return parsed

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        """Strips markdown ```json and ``` code block wrappers."""
        text = content.strip()
        # Strip ```json ... ``` or ``` ... ```
        fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        # Fallback regex strip if fences are uneven
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _validate_schema(config: Dict[str, Any], eda_summary: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates that config adheres strictly to Data Contract 4.2 and anti-hallucination rules."""
        required_keys = [
            "categorical_columns",
            "epochs",
            "batch_size",
            "generator_dim",
            "discriminator_dim",
            "pac",
            "reasoning",
        ]
        for key in required_keys:
            if key not in config:
                return False, f"Missing required field: '{key}'"

        # 1. categorical_columns
        cat_cols = config.get("categorical_columns")
        if not isinstance(cat_cols, list):
            return False, f"'categorical_columns' must be a list, got {type(cat_cols).__name__}"

        allowed_cat = set(eda_summary.get("categorical_columns", []))
        for col in cat_cols:
            if not isinstance(col, str):
                return False, f"All items in 'categorical_columns' must be strings, got {col}"
            if col not in allowed_cat:
                return False, (
                    f"Hallucinated categorical column '{col}' not present in EDA categorical_columns: {sorted(list(allowed_cat))}"
                )

        # 2. epochs
        epochs = config.get("epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int):
            return False, f"'epochs' must be an integer, got {epochs}"
        if not (50 <= epochs <= 500):
            return False, f"'epochs' must be between 50 and 500, got {epochs}"

        # 3. batch_size
        batch_size = config.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            return False, f"'batch_size' must be an integer, got {batch_size}"
        if not (50 <= batch_size <= 1000):
            return False, f"'batch_size' must be between 50 and 1000, got {batch_size}"
        if batch_size % 10 != 0:
            return False, f"'batch_size' must be a multiple of 10, got {batch_size}"

        # 4. generator_dim
        gen_dim = config.get("generator_dim")
        if not isinstance(gen_dim, (list, tuple)) or len(gen_dim) < 1:
            return False, f"'generator_dim' must be a non-empty list of integers, got {gen_dim}"
        if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in gen_dim):
            return False, f"'generator_dim' dimensions must be positive integers, got {gen_dim}"

        # 5. discriminator_dim
        disc_dim = config.get("discriminator_dim")
        if not isinstance(disc_dim, (list, tuple)) or len(disc_dim) < 1:
            return False, f"'discriminator_dim' must be a non-empty list of integers, got {disc_dim}"
        if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in disc_dim):
            return False, f"'discriminator_dim' dimensions must be positive integers, got {disc_dim}"

        # 6. pac
        pac = config.get("pac")
        if isinstance(pac, bool) or not isinstance(pac, int):
            return False, f"'pac' must be an integer, got {pac}"
        if not (1 <= pac <= 20):
            return False, f"'pac' must be between 1 and 20, got {pac}"

        # 7. reasoning
        reasoning = config.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            return False, f"'reasoning' must be a non-empty string, got {reasoning}"

        return True, None

    @staticmethod
    def _default_config(eda_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Provides a safe fallback CTGAN configuration matching Data Contract 4.2."""
        return {
            "categorical_columns": list(eda_summary.get("categorical_columns", [])),
            "epochs": 150,
            "batch_size": 500,
            "generator_dim": [256, 256],
            "discriminator_dim": [256, 256],
            "pac": 10,
            "reasoning": "Fallback default configuration due to LLM response failure.",
        }
