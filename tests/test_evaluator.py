"""Unit tests for Evaluator Metrics and evaluate_all().

Verifies individual metric computations, edge cases, and Definition of Done
from Section 5.3 and Data Contract 4.3 of the SyntheLoop Implementation Methodology.
"""

import pytest
import numpy as np
import pandas as pd

from backend.config import settings
from backend.eda.analyzer import EDAAnalyzer
from backend.generator.ctgan_wrapper import SyntheticGenerator
from backend.evaluator.metrics import (
    compute_ks_per_column,
    compute_correlation_diff,
    compute_class_balance_js,
    compute_privacy_dcr,
    compute_ml_utility,
    evaluate_all,
)


@pytest.fixture
def sample_real_synthetic_pair():
    """Generates a reproducible real and synthetic pair for deterministic testing."""
    np.random.seed(42)
    n = 100
    real = pd.DataFrame({
        "age": np.random.normal(40, 10, size=n),
        "salary": np.random.normal(60000, 15000, size=n),
        "department": np.random.choice(["HR", "Sales", "Engineering"], size=n),
        "churn": np.random.choice([0, 1], p=[0.7, 0.3], size=n),
    })

    # Synthetic with slight noise
    synth = pd.DataFrame({
        "age": np.random.normal(40, 11, size=n),
        "salary": np.random.normal(59000, 16000, size=n),
        "department": np.random.choice(["HR", "Sales", "Engineering"], size=n),
        "churn": np.random.choice([0, 1], p=[0.68, 0.32], size=n),
    })
    return real, synth


def test_compute_ks_per_column():
    """Validates KS statistic is near 0 for identical and near 1 for disjoint distributions."""
    np.random.seed(42)
    r = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
    s_same = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
    s_diff = pd.DataFrame({"x": np.random.normal(10, 1, 500)})

    res_same = compute_ks_per_column(r, s_same, ["x"])
    res_diff = compute_ks_per_column(r, s_diff, ["x"])

    assert res_same["x"] < 0.15
    assert res_diff["x"] > 0.80


def test_compute_correlation_diff():
    """Validates Frobenius norm correlation difference."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 200)
    y = x * 2.0 + np.random.normal(0, 0.1, 200)

    df_real = pd.DataFrame({"x": x, "y": y})
    df_synth = pd.DataFrame({"x": x, "y": y})  # Identical correlation

    diff_zero = compute_correlation_diff(df_real, df_synth, ["x", "y"])
    assert diff_zero == 0.0

    # Opposite correlation
    df_opp = pd.DataFrame({"x": x, "y": -y})
    diff_large = compute_correlation_diff(df_real, df_opp, ["x", "y"])
    assert diff_large > 0.5


def test_compute_class_balance_js():
    """Validates Jensen-Shannon divergence across class balances."""
    s1 = pd.Series(["A"] * 50 + ["B"] * 50)
    s2 = pd.Series(["A"] * 50 + ["B"] * 50)
    assert compute_class_balance_js(s1, s2) == 0.0

    s_skewed = pd.Series(["A"] * 95 + ["B"] * 5)
    js_skewed = compute_class_balance_js(s1, s_skewed)
    assert js_skewed > 0.20


def test_compute_privacy_dcr():
    """Validates 5th percentile nearest neighbor distance."""
    # Identical dataset: DCR should be 0.0
    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    dcr_zero = compute_privacy_dcr(df1, df1, ["a", "b"])
    assert dcr_zero == 0.0

    # Shifted synthetic points
    df2 = pd.DataFrame({"a": [100.0, 200.0], "b": [1000.0, 2000.0]})
    dcr_positive = compute_privacy_dcr(df1, df2, ["a", "b"])
    assert dcr_positive > 0.0


def test_compute_ml_utility(sample_real_synthetic_pair):
    """Validates TRTR and TSTR utility calculation."""
    real, synth = sample_real_synthetic_pair
    utility = compute_ml_utility(real, synth, target_col="churn")

    assert "trtr_auc" in utility
    assert "tstr_auc" in utility
    assert "auc_drop" in utility
    assert isinstance(utility["trtr_auc"], float)
    assert isinstance(utility["tstr_auc"], float)
    assert isinstance(utility["auc_drop"], float)


def test_evaluate_all_definition_of_done():
    """Definition of Done test: runs evaluate_all on real/synthetic pair from

    SyntheticGenerator output, asserts every key in Data Contract 4.3 is present
    with correct types, and that overall_passed is a bool.
    """
    np.random.seed(42)
    n = 120
    real_df = pd.DataFrame({
        "age": np.random.randint(20, 65, size=n),
        "income": np.random.uniform(30000, 100000, size=n),
        "status": np.random.choice(["Single", "Married"], size=n),
        "label": np.random.choice([0, 1], size=n),
    })

    # Run EDA to get contract 4.1 summary
    analyzer = EDAAnalyzer(real_df, target_col="label")
    eda_summary = analyzer.analyze()

    # Run CTGAN generator
    gen_config = {
        "categorical_columns": ["status", "label"],
        "epochs": 1,
        "batch_size": 10,
        "pac": 1,
    }
    generator = SyntheticGenerator(gen_config)
    generator.fit(real_df)
    synth_df = generator.sample(120)

    # Run evaluate_all
    report = evaluate_all(
        real=real_df,
        synth=synth_df,
        eda_summary=eda_summary,
        thresholds=settings.default_thresholds,
        iteration=1,
    )

    # Assert Data Contract 4.3 Schema Keys and Types
    assert isinstance(report["iteration"], int)
    assert report["iteration"] == 1

    assert isinstance(report["per_column_ks"], dict)
    for col, val in report["per_column_ks"].items():
        assert isinstance(col, str)
        assert isinstance(val, float)

    assert isinstance(report["correlation_diff_frobenius"], float)
    assert isinstance(report["class_balance_js_divergence"], float)
    assert isinstance(report["privacy_dcr_5th_percentile"], float)

    assert isinstance(report["utility"], dict)
    assert "trtr_auc" in report["utility"]
    assert "tstr_auc" in report["utility"]
    assert "auc_drop" in report["utility"]
    assert isinstance(report["utility"]["trtr_auc"], float)
    assert isinstance(report["utility"]["tstr_auc"], float)
    assert isinstance(report["utility"]["auc_drop"], float)

    assert isinstance(report["thresholds"], dict)
    assert report["thresholds"] == settings.default_thresholds

    assert isinstance(report["passed"], dict)
    for metric_name in ["ks", "correlation", "balance", "privacy", "utility"]:
        assert metric_name in report["passed"]
        assert isinstance(report["passed"][metric_name], bool)

    assert isinstance(report["overall_passed"], bool)


def test_evaluate_all_no_target_column():
    """Validates evaluate_all when target_column is not set (unsupervised dataset)."""
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0] * 5,
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0] * 5,
    })
    eda_summary = EDAAnalyzer(df).analyze()

    report = evaluate_all(
        real=df,
        synth=df,
        eda_summary=eda_summary,
        thresholds=settings.default_thresholds,
        iteration=1,
    )

    assert report["class_balance_js_divergence"] == 0.0
    assert report["utility"]["auc_drop"] == 0.0
    assert isinstance(report["overall_passed"], bool)
