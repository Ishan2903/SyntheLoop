"""SyntheLoop — Pipeline Audit Trail Module.

Manages iteration logging for the optimization loop and persists records to
{output_dir}/{run_id}/audit_trail.json as a JSON array conforming to Data Contract 4.5.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_ACTIONS = {
    "continued",
    "stopped_threshold_met",
    "stopped_max_iterations",
    "failed",
}


class AuditTrail:
    """Manages appending and retrieving audit trail iteration logs for a run."""

    def __init__(self, run_id: str, output_dir: str = "outputs") -> None:
        """Initializes audit trail directory and file path.

        Args:
            run_id: Unique identifier for the pipeline run.
            output_dir: Base outputs directory. Defaults to 'outputs'.
        """
        if not run_id or not str(run_id).strip():
            raise ValueError("run_id cannot be empty")

        self.run_id = str(run_id).strip()
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trail_file = self.run_dir / "audit_trail.json"

    def log_iteration(self, entry: dict[str, Any]) -> None:
        """Appends one Data Contract 4.5 entry to {output_dir}/{run_id}/audit_trail.json.

        Expected schema (Data Contract 4.5):
        {
            "run_id": str,
            "iteration": int,
            "timestamp": "ISO-8601",
            "config_used": dict,
            "metrics": dict,
            "feedback": dict | None,
            "action_taken": "continued | stopped_threshold_met | stopped_max_iterations | failed"
        }

        Args:
            entry: Dictionary containing iteration run data.

        Raises:
            ValueError: If required fields are missing or action_taken is invalid.
        """
        if not isinstance(entry, dict):
            raise ValueError(f"Audit trail entry must be a dictionary, got {type(entry).__name__}")

        record = dict(entry)

        # Auto-populate run_id if missing or empty
        if not record.get("run_id"):
            record["run_id"] = self.run_id

        # Auto-populate timestamp if missing or empty
        if not record.get("timestamp"):
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Ensure feedback key exists even if None
        if "feedback" not in record:
            record["feedback"] = None

        # Validate required fields
        required_keys = ["iteration", "config_used", "metrics", "action_taken"]
        for key in required_keys:
            if key not in record:
                raise ValueError(f"Audit trail entry missing required field: '{key}'")

        if not isinstance(record["iteration"], int) or record["iteration"] < 1:
            raise ValueError(f"Iteration must be a positive integer, got {record['iteration']}")

        action = str(record["action_taken"])
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action_taken '{action}'. Must be one of: {sorted(VALID_ACTIONS)}"
            )

        # Read existing trail
        trail = self._load_trail()
        trail.append(record)

        # Persist updated trail
        self._write_trail(trail)
        logger.info(
            f"Logged iteration {record['iteration']} for run {self.run_id} (action: {action})"
        )

    def get_full_trail(self) -> list[dict[str, Any]]:
        """Returns the full list of audit trail entries for this run.

        Returns:
            list[dict]: Array of iteration log dictionaries conforming to Data Contract 4.5.
        """
        return self._load_trail()

    def get_latest_entry(self) -> dict[str, Any] | None:
        """Returns the most recent iteration entry, or None if no entries logged yet."""
        trail = self.get_full_trail()
        return trail[-1] if trail else None

    def _load_trail(self) -> list[dict[str, Any]]:
        """Loads trail from JSON file if present, handling missing or empty files."""
        if not self.trail_file.exists():
            return []

        try:
            content = self.trail_file.read_text(encoding="utf-8").strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            logger.warning(f"Audit trail file {self.trail_file} was not a list, resetting to empty")
            return []
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read audit trail file {self.trail_file}: {e}")
            return []

    def _write_trail(self, trail: list[dict[str, Any]]) -> None:
        """Writes trail to JSON file safely."""
        temp_file = self.run_dir / f"audit_trail_{os.getpid()}_{datetime.now().timestamp()}.tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(trail, f, indent=2, ensure_ascii=False)
            # Atomic replacement
            temp_file.replace(self.trail_file)
        except Exception:
            # Fallback direct write if atomic replace encounters an OS lock
            try:
                with open(self.trail_file, "w", encoding="utf-8") as f:
                    json.dump(trail, f, indent=2, ensure_ascii=False)
            finally:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass
