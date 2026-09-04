"""EDA Analyzer module for SyntheLoop.

Analyzes an input pandas DataFrame and generates an EDA summary matching
Data Contract 4.1.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class EDAAnalyzer:
    """Performs Exploratory Data Analysis on a tabular dataset.
    
    Generates statistics for continuous and categorical columns, class balance,
    and correlation matrix according to Data Contract 4.1.
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """Initializes and validates the input dataset.

        Args:
            df: Input pandas DataFrame.
            target_col: Optional name of the target column.

        Raises:
            ValueError: If the dataset is empty, single-row, has zero columns,
                        or contains no usable numeric or categorical columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        if df.empty or len(df) == 0:
            raise ValueError("Dataset is empty (0 rows).")

        if len(df) == 1:
            raise ValueError("Dataset must have more than 1 row (single-row dataset provided).")

        if len(df.columns) == 0:
            raise ValueError("Dataset has zero columns.")

        # Ensure there is at least one usable column (numeric or categorical/object)
        usable_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_object_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ]
        if not usable_cols:
            raise ValueError("Dataset has no usable numeric or categorical columns.")

        self.df = df
        self.target_col = target_col if (target_col and target_col in df.columns) else None

    def analyze(self) -> Dict[str, Any]:
        """Runs the exploratory data analysis and returns a summary conforming to Data Contract 4.1.

        Returns:
            Dictionary matching Data Contract 4.1.
        """
        n_rows, n_cols = self.df.shape
        columns_meta: Dict[str, Dict[str, Any]] = {}
        categorical_columns: List[str] = []

        for col in self.df.columns:
            series = self.df[col]
            dtype_str = str(series.dtype)
            n_unique = int(series.nunique(dropna=True))
            missing_count = int(series.isna().sum())
            missing_pct = round(float((missing_count / n_rows) * 100.0), 2)

            # Rule: A column is continuous if pd.api.types.is_numeric_dtype is true
            # AND n_unique > 20; otherwise categorical. This rule determines what
            # gets sent to CTGAN as discrete vs. continuous.
            is_numeric = bool(pd.api.types.is_numeric_dtype(series))
            is_continuous = is_numeric and (n_unique > 20)

            col_summary: Dict[str, Any] = {
                "dtype": dtype_str,
                "type": "continuous" if is_continuous else "categorical",
                "missing_pct": missing_pct,
                "n_unique": n_unique,
            }

            if is_continuous:
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    col_summary["mean"] = round(float(non_null_series.mean()), 4)
                    col_summary["std"] = round(float(non_null_series.std()), 4) if len(non_null_series) > 1 and not pd.isna(non_null_series.std()) else 0.0
                    col_summary["skew"] = round(float(non_null_series.skew()), 4) if len(non_null_series) > 2 and not pd.isna(non_null_series.skew()) else 0.0
                    col_summary["kurtosis"] = round(float(non_null_series.kurtosis()), 4) if len(non_null_series) > 3 and not pd.isna(non_null_series.kurtosis()) else 0.0
                    col_summary["min"] = round(float(non_null_series.min()), 4)
                    col_summary["max"] = round(float(non_null_series.max()), 4)
                else:
                    # Column is 100% missing
                    col_summary["mean"] = None
                    col_summary["std"] = None
                    col_summary["skew"] = None
                    col_summary["kurtosis"] = None
                    col_summary["min"] = None
                    col_summary["max"] = None
            else:
                categorical_columns.append(col)
                val_counts = series.dropna().value_counts(normalize=True)
                top_cats = {str(k): round(float(v), 4) for k, v in val_counts.head(5).items()}
                col_summary["top_categories"] = top_cats

                # Balance ratio: frequency of least common category / frequency of most common category
                raw_counts = series.dropna().value_counts()
                if len(raw_counts) > 1:
                    balance_ratio = round(float(raw_counts.min() / raw_counts.max()), 4)
                elif len(raw_counts) == 1:
                    balance_ratio = 1.0
                else:
                    balance_ratio = 0.0
                col_summary["balance_ratio"] = balance_ratio

            columns_meta[col] = col_summary

        summary: Dict[str, Any] = {
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "target_column": self.target_col,
            "columns": columns_meta,
            "categorical_columns": categorical_columns,
        }

        # Correlation matrix computed only over numeric columns;
        # omit the key entirely if fewer than 2 numeric columns exist.
        numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        if len(numeric_cols) >= 2:
            corr_df = self.df[numeric_cols].corr()
            corr_matrix: Dict[str, Dict[str, float]] = {}
            for col_a in numeric_cols:
                corr_matrix[col_a] = {}
                for col_b in numeric_cols:
                    val = corr_df.loc[col_a, col_b]
                    corr_matrix[col_a][col_b] = round(float(val), 4) if not pd.isna(val) else 0.0
            summary["correlation_matrix"] = corr_matrix

        # class_balance computed only if target_col is provided and exists in df.columns
        if self.target_col and self.target_col in self.df.columns:
            target_series = self.df[self.target_col].dropna()
            target_counts = target_series.value_counts(normalize=True)
            summary["class_balance"] = {
                str(k): round(float(v), 4) for k, v in target_counts.items()
            }

        return summary
