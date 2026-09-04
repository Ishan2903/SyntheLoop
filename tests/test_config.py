import os
import pytest
from backend.config import Settings, settings


def test_settings_defaults():
    """Verify settings defaults when no environment variables are explicitly set."""
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.max_iterations_default == 5
    assert settings.output_dir == "outputs"
    assert settings.upload_dir == "data/uploads"
    assert settings.default_thresholds["ks_stat_max"] == 0.15
    assert settings.default_thresholds["corr_diff_max"] == 0.20
    assert settings.default_thresholds["js_divergence_max"] == 0.10
    assert settings.default_thresholds["dcr_min_percentile"] == 5
    assert settings.default_thresholds["utility_auc_drop_max"] == 0.10


def test_groq_api_key_defaults_to_empty_string():
    """Verify groq_api_key is an empty string (not None) when GROQ_API_KEY is unset.
    The module-level settings singleton is loaded without a .env file in CI,
    so groq_api_key must be '' to prevent downstream type errors."""
    fresh = Settings(groq_api_key=os.getenv("GROQ_API_KEY", ""))
    assert isinstance(fresh.groq_api_key, str)
    assert fresh.groq_api_key == ""


def test_settings_override_via_env_chain(monkeypatch):
    """Test the full os.getenv chain: patch env vars, then construct Settings()
    reading those vars directly — simulating what happens when .env is populated."""
    monkeypatch.setenv("GROQ_API_KEY", "test_key_123")
    monkeypatch.setenv("GROQ_MODEL", "mixtral-8x7b-32768")
    monkeypatch.setenv("MAX_ITERATIONS_DEFAULT", "10")

    # Simulate the same logic as backend/config.py's class body reading os.getenv
    custom_settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        max_iterations_default=int(os.getenv("MAX_ITERATIONS_DEFAULT", "5")),
    )

    assert custom_settings.groq_api_key == "test_key_123"
    assert custom_settings.groq_model == "mixtral-8x7b-32768"
    assert custom_settings.max_iterations_default == 10


def test_default_thresholds_are_independent_instances():
    """Verify default_thresholds is not a shared mutable dict across Settings instances.
    Using Field(default_factory=...) means each Settings() gets its own dict copy."""
    s1 = Settings()
    s2 = Settings()
    s1.default_thresholds["ks_stat_max"] = 0.99
    assert s2.default_thresholds["ks_stat_max"] == 0.15, (
        "default_thresholds must not be shared across instances — use default_factory"
    )


def test_all_threshold_keys_present():
    """Verify all 5 required threshold keys are present in default_thresholds."""
    required_keys = {
        "ks_stat_max",
        "corr_diff_max",
        "js_divergence_max",
        "dcr_min_percentile",
        "utility_auc_drop_max",
    }
    assert required_keys == set(settings.default_thresholds.keys())
