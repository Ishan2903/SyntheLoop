import json
import numpy as np
import pandas as pd
import pytest
from backend.eda.analyzer import EDAAnalyzer


def assert_conforms_to_contract_4_1(summary: dict, expected_target: str | None = None):
    """Helper to assert that a summary dictionary conforms to Data Contract 4.1."""
    assert "n_rows" in summary and isinstance(summary["n_rows"], int)
    assert "n_cols" in summary and isinstance(summary["n_cols"], int)
    assert "target_column" in summary
    assert summary["target_column"] == expected_target
    assert "columns" in summary and isinstance(summary["columns"], dict)
    assert "categorical_columns" in summary and isinstance(summary["categorical_columns"], list)

    for col_name, col_meta in summary["columns"].items():
        assert "dtype" in col_meta and isinstance(col_meta["dtype"], str)
        assert "type" in col_meta and col_meta["type"] in ("continuous", "categorical")
        assert "missing_pct" in col_meta and isinstance(col_meta["missing_pct"], (int, float))
        assert "n_unique" in col_meta and isinstance(col_meta["n_unique"], int)

        if col_meta["type"] == "continuous":
            for k in ["mean", "std", "skew", "kurtosis", "min", "max"]:
                assert k in col_meta
                assert col_meta[k] is None or isinstance(col_meta[k], (int, float))
        else:
            assert "top_categories" in col_meta and isinstance(col_meta["top_categories"], dict)
            assert "balance_ratio" in col_meta and isinstance(col_meta["balance_ratio"], (int, float))

    # Correlation matrix check
    if "correlation_matrix" in summary:
        assert isinstance(summary["correlation_matrix"], dict)
        for col_a, row in summary["correlation_matrix"].items():
            assert isinstance(row, dict)
            for col_b, val in row.items():
                assert isinstance(val, (int, float))

    # Class balance check
    if "class_balance" in summary:
        assert isinstance(summary["class_balance"], dict)
        for cls, prob in summary["class_balance"].items():
            assert isinstance(prob, (int, float))


def test_eda_mixed_type_dataset():
    """Test (a): mixed-type sample with target column."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),  # > 20 unique -> continuous
        "salary": np.random.uniform(30000, 120000, size=n),  # continuous
        "education": np.random.choice(["HighSchool", "Bachelors", "Masters", "PhD"], size=n),  # categorical
        "department": np.random.choice(["HR", "Sales", "Engineering"], size=n),  # categorical
        "target": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),  # target
    })

    analyzer = EDAAnalyzer(df, target_col="target")
    summary = analyzer.analyze()

    assert_conforms_to_contract_4_1(summary, expected_target="target")
    assert summary["n_rows"] == 100
    assert summary["n_cols"] == 5
    assert summary["columns"]["age"]["type"] == "continuous"
    assert summary["columns"]["salary"]["type"] == "continuous"
    assert summary["columns"]["education"]["type"] == "categorical"
    assert "correlation_matrix" in summary
    assert "class_balance" in summary
    assert set(summary["class_balance"].keys()) == {"0", "1"}
    assert "education" in summary["categorical_columns"]


def test_eda_all_numeric_dataset():
    """Test (b): all-numeric dataset."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "feature_1": np.random.randn(n) * 10,
        "feature_2": np.random.randn(n) * 5 + 20,
        "feature_3": np.random.exponential(scale=2.0, size=n),
    })

    analyzer = EDAAnalyzer(df)
    summary = analyzer.analyze()

    assert_conforms_to_contract_4_1(summary, expected_target=None)
    assert summary["n_rows"] == 50
    assert summary["n_cols"] == 3
    assert len(summary["categorical_columns"]) == 0
    assert "correlation_matrix" in summary
    assert "feature_1" in summary["correlation_matrix"]
    assert "class_balance" not in summary  # No target column


def test_eda_fully_missing_column():
    """Test (c): dataset containing a 100% missing column."""
    df = pd.DataFrame({
        "num_col": list(range(25)),
        "cat_col": ["A", "B", "C", "D", "E"] * 5,
        "missing_num": [np.nan] * 25,
        "missing_obj": [None] * 25,
    })

    analyzer = EDAAnalyzer(df)
    summary = analyzer.analyze()

    assert_conforms_to_contract_4_1(summary, expected_target=None)
    assert summary["columns"]["missing_num"]["missing_pct"] == 100.0
    assert summary["columns"]["missing_obj"]["missing_pct"] == 100.0
    assert summary["columns"]["missing_num"]["n_unique"] == 0
    # Because n_unique == 0 (<= 20), column is classified as categorical per the rule
    assert summary["columns"]["missing_num"]["type"] == "categorical"
    assert summary["columns"]["missing_num"]["top_categories"] == {}
    assert summary["columns"]["missing_num"]["balance_ratio"] == 0.0


def test_eda_continuous_with_partial_missing():
    """Test continuous column (n_unique > 20) with partial missing values."""
    np.random.seed(42)
    vals = list(np.random.randn(50))
    vals[0] = np.nan
    vals[5] = np.nan
    df = pd.DataFrame({"cont_missing": vals})

    analyzer = EDAAnalyzer(df)
    summary = analyzer.analyze()

    assert_conforms_to_contract_4_1(summary)
    assert summary["columns"]["cont_missing"]["type"] == "continuous"
    assert summary["columns"]["cont_missing"]["missing_pct"] == 4.0
    assert summary["columns"]["cont_missing"]["mean"] is not None
    assert summary["columns"]["cont_missing"]["min"] is not None



def test_eda_all_categorical_dataset():
    """Test dataset with all categorical columns (fewer than 2 numeric cols)."""
    df = pd.DataFrame({
        "city": ["Paris", "London", "Tokyo", "Berlin"] * 10,
        "status": ["Active", "Pending", "Suspended", "Active"] * 10,
    })

    analyzer = EDAAnalyzer(df)
    summary = analyzer.analyze()

    assert_conforms_to_contract_4_1(summary)
    assert "correlation_matrix" not in summary  # < 2 numeric cols, must omit
    assert set(summary["categorical_columns"]) == {"city", "status"}


def test_eda_single_class_target():
    """Test target column containing only a single unique class."""
    df = pd.DataFrame({
        "x": list(range(30)),
        "target": ["only_class"] * 30,
    })

    analyzer = EDAAnalyzer(df, target_col="target")
    summary = analyzer.analyze()

    assert_conforms_to_contract_4_1(summary, expected_target="target")
    assert "class_balance" in summary
    assert summary["class_balance"] == {"only_class": 1.0}


def test_eda_validation_errors():
    """Test dataset validation errors according to Section 8."""
    # Non-DataFrame
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        EDAAnalyzer("not_a_df")  # type: ignore

    # Empty DataFrame (0 rows)
    with pytest.raises(ValueError, match="empty \\(0 rows\\)"):
        EDAAnalyzer(pd.DataFrame())

    # Single row dataset
    with pytest.raises(ValueError, match="more than 1 row"):
        EDAAnalyzer(pd.DataFrame({"a": [1], "b": ["x"]}))

    # Zero columns
    with pytest.raises(ValueError, match="empty \\(0 rows\\)|zero columns"):
        EDAAnalyzer(pd.DataFrame(index=[0, 1]))


def test_eda_json_serializability():
    """Ensure the returned dict is strictly JSON serializable (no raw numpy types)."""
    df = pd.DataFrame({
        "int_col": np.arange(30, dtype=np.int64),
        "float_col": np.linspace(0.0, 1.0, 30, dtype=np.float64),
        "cat_col": ["category_" + str(i % 3) for i in range(30)],
    })

    analyzer = EDAAnalyzer(df, target_col="cat_col")
    summary = analyzer.analyze()

    json_str = json.dumps(summary)
    assert isinstance(json_str, str)
    decoded = json.loads(json_str)
    assert decoded["n_rows"] == 30
