"""Unit tests for SyntheticGenerator (CTGAN Wrapper).

Verifies functionality, schema fidelity, edge cases, and Definition of Done
from Section 5.2 of SyntheLoop Implementation Methodology.
"""

import pytest
import pandas as pd
import numpy as np

from backend.generator.ctgan_wrapper import SyntheticGenerator


@pytest.fixture
def sample_mixed_df() -> pd.DataFrame:
    """Generates a small reproducible mixed-type dataset for fast testing."""
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.uniform(20000.0, 120000.0, size=n),
        "education": np.random.choice(["High School", "Bachelors", "Masters", "PhD"], size=n),
        "employed": np.random.choice([0, 1], size=n),
    })


@pytest.fixture
def default_test_config() -> dict:
    """Fast test configuration adhering to Data Contract 4.2."""
    return {
        "categorical_columns": ["education", "employed"],
        "epochs": 1,
        "batch_size": 10,
        "generator_dim": [128, 128],
        "discriminator_dim": [128, 128],
        "pac": 1,
        "reasoning": "Fast unit test configuration",
    }


def test_generator_fit_and_sample(sample_mixed_df, default_test_config):
    """Definition of Done test: fits on <=500 row dataset, samples 100 rows,

    asserts output shape/dtypes match input.
    """
    generator = SyntheticGenerator(default_test_config)
    generator.fit(sample_mixed_df)

    sampled = generator.sample(100)

    # Assert shape
    assert isinstance(sampled, pd.DataFrame)
    assert sampled.shape == (100, sample_mixed_df.shape[1])

    # Assert exact column names and order
    assert list(sampled.columns) == list(sample_mixed_df.columns)

    # Assert dtypes match
    for col in sample_mixed_df.columns:
        assert sampled[col].dtype == sample_mixed_df[col].dtype, (
            f"Dtype mismatch on column '{col}': expected {sample_mixed_df[col].dtype}, got {sampled[col].dtype}"
        )


def test_generator_hallucinated_column_validation(sample_mixed_df, default_test_config):
    """NFR-7 Defense: validates that hallucinated column names raise ValueError."""
    bad_config = default_test_config.copy()
    bad_config["categorical_columns"] = ["education", "hallucinated_feature_x", "invented_col"]

    generator = SyntheticGenerator(bad_config)
    with pytest.raises(ValueError) as exc_info:
        generator.fit(sample_mixed_df)

    msg = str(exc_info.value)
    assert "hallucinated_feature_x" in msg
    assert "invented_col" in msg


def test_generator_sample_before_fit_raises(default_test_config):
    """Calling sample() prior to fit() must raise RuntimeError."""
    generator = SyntheticGenerator(default_test_config)
    with pytest.raises(RuntimeError) as exc_info:
        generator.sample(50)

    assert "must be fitted" in str(exc_info.value).lower()


def test_generator_invalid_sample_n_rows(sample_mixed_df, default_test_config):
    """Passing n_rows <= 0 or invalid types raises ValueError."""
    generator = SyntheticGenerator(default_test_config)
    generator.fit(sample_mixed_df)

    with pytest.raises(ValueError):
        generator.sample(0)

    with pytest.raises(ValueError):
        generator.sample(-10)


def test_generator_all_numeric():
    """Generator handles datasets with no categorical columns."""
    df_numeric = pd.DataFrame({
        "x": np.linspace(0, 10, 60),
        "y": np.linspace(10, 20, 60),
    })
    config = {
        "categorical_columns": [],
        "epochs": 1,
        "batch_size": 10,
        "pac": 1,
    }
    generator = SyntheticGenerator(config)
    generator.fit(df_numeric)

    sampled = generator.sample(25)
    assert sampled.shape == (25, 2)
    assert list(sampled.columns) == ["x", "y"]
    assert sampled["x"].dtype == df_numeric["x"].dtype


def test_generator_empty_dataframe_raises(default_test_config):
    """fit() must reject empty DataFrames with a clear descriptive message."""
    generator = SyntheticGenerator(default_test_config)
    with pytest.raises(ValueError) as exc_info:
        generator.fit(pd.DataFrame())

    assert "empty" in str(exc_info.value).lower()


def test_generator_dtype_preservation():
    """Validates integer rounding and type preservation across sampling."""
    df = pd.DataFrame({
        "int_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 6,
        "float_val": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 6,
        "str_cat": ["A", "B"] * 30,
    })
    config = {
        "categorical_columns": ["str_cat"],
        "epochs": 1,
        "batch_size": 10,
        "pac": 1,
    }
    generator = SyntheticGenerator(config)
    generator.fit(df)

    sampled = generator.sample(30)
    assert pd.api.types.is_integer_dtype(sampled["int_id"].dtype)
    assert pd.api.types.is_float_dtype(sampled["float_val"].dtype)
    assert pd.api.types.is_object_dtype(sampled["str_cat"].dtype)
