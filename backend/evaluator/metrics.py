"""Evaluation metrics for SyntheLoop synthetic data quality, privacy, and utility.

Implements all 5 evaluators and evaluate_all() conforming to Data Contract 4.3
and Section 5.3 of the SyntheLoop Implementation Methodology.
"""

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def compute_ks_per_column(
    real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: list[str]
) -> dict[str, float]:
    """Computes the 2-sample Kolmogorov-Smirnov statistic per numeric column.

    Args:
        real: Real DataFrame.
        synth: Synthetic DataFrame.
        numeric_cols: List of continuous/numeric column names.

    Returns:
        dict[str, float]: Mapping of column name to KS statistic.
    """
    ks_results: dict[str, float] = {}
    for col in numeric_cols:
        if col in real.columns and col in synth.columns:
            r_vals = real[col].dropna().to_numpy()
            s_vals = synth[col].dropna().to_numpy()
            if len(r_vals) == 0 or len(s_vals) == 0:
                ks_results[col] = 1.0
            else:
                stat = stats.ks_2samp(r_vals, s_vals).statistic
                ks_results[col] = float(round(stat, 4))
    return ks_results


def compute_correlation_diff(
    real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: list[str]
) -> float:
    """Computes Frobenius norm of (real_corr - synth_corr) normalized by matrix size.

    Args:
        real: Real DataFrame.
        synth: Synthetic DataFrame.
        numeric_cols: List of numeric column names.

    Returns:
        float: Normalized Frobenius norm difference. Returns 0.0 if < 2 numeric columns.
    """
    valid_cols = [c for c in numeric_cols if c in real.columns and c in synth.columns]
    if len(valid_cols) < 2:
        return 0.0

    real_corr = real[valid_cols].corr().fillna(0.0).to_numpy()
    synth_corr = synth[valid_cols].corr().fillna(0.0).to_numpy()

    diff = real_corr - synth_corr
    norm_val = np.linalg.norm(diff, ord="fro")
    normalized_diff = float(norm_val / len(valid_cols))
    return float(round(normalized_diff, 4))


def compute_class_balance_js(real: pd.Series, synth: pd.Series) -> float:
    """Computes Jensen-Shannon divergence between real and synthetic class distributions.

    Aligns categories and fills unobserved classes with 0.0 before calculation.

    Args:
        real: Series of target labels from real data.
        synth: Series of target labels from synthetic data.

    Returns:
        float: Jensen-Shannon divergence (distance squared or metric distance).
    """
    if real.empty or synth.empty:
        return 0.0

    p = real.value_counts(normalize=True)
    q = synth.value_counts(normalize=True)

    all_categories = p.index.union(q.index)
    p_aligned = p.reindex(all_categories, fill_value=0.0).to_numpy(dtype=float)
    q_aligned = q.reindex(all_categories, fill_value=0.0).to_numpy(dtype=float)

    # distance.jensenshannon computes the JS distance (square root of JS divergence)
    js_dist = distance.jensenshannon(p_aligned, q_aligned, base=2)
    if np.isnan(js_dist):
        js_dist = 0.0

    return float(round(js_dist, 4))


def compute_privacy_dcr(
    real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: list[str], percentile: float = 5.0
) -> float:
    """Computes Distance to Closest Record (DCR) for synthetic rows.

    For each synthetic row, calculates nearest-neighbor Euclidean distance
    (on min-max normalized numeric columns) to any real row, and returns
    the specified percentile (default: 5th percentile).

    Args:
        real: Real DataFrame.
        synth: Synthetic DataFrame.
        numeric_cols: List of numeric column names.
        percentile: Percentile of distance distribution to return (default: 5th).

    Returns:
        float: The 5th percentile nearest neighbor distance. Returns 1.0 if no numeric columns.
    """
    valid_cols = [c for c in numeric_cols if c in real.columns and c in synth.columns]
    if not valid_cols or real.empty or synth.empty:
        return 1.0

    # Min-max normalization based on real dataset statistics
    real_norm = pd.DataFrame(index=real.index)
    synth_norm = pd.DataFrame(index=synth.index)

    for col in valid_cols:
        r_col = real[col].dropna()
        col_min = float(r_col.min()) if len(r_col) > 0 else 0.0
        col_max = float(r_col.max()) if len(r_col) > 0 else 1.0
        span = (col_max - col_min) if (col_max - col_min) > 1e-9 else 1.0

        real_norm[col] = (real[col].fillna(col_min) - col_min) / span
        synth_norm[col] = (synth[col].fillna(col_min) - col_min) / span

    # Nearest neighbor search
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean").fit(
        real_norm.to_numpy()
    )
    distances, _ = nn.kneighbors(synth_norm.to_numpy())
    dcr_val = float(np.percentile(distances.flatten(), percentile))
    return float(round(dcr_val, 4))


def compute_ml_utility(
    real: pd.DataFrame, synth: pd.DataFrame, target_col: str | None
) -> dict[str, float]:
    """Computes ML utility comparing TRTR AUC and TSTR AUC with RandomForestClassifier.

    TRTR: Train on Real, Test on Real holdout (80/20 split).
    TSTR: Train on Synthetic, Test on SAME Real holdout.

    Args:
        real: Real DataFrame.
        synth: Synthetic DataFrame.
        target_col: Target column name for classification.

    Returns:
        dict[str, float]: {"trtr_auc": float, "tstr_auc": float, "auc_drop": float}
    """
    default_result = {"trtr_auc": 1.0, "tstr_auc": 1.0, "auc_drop": 0.0}
    if (
        not target_col
        or target_col not in real.columns
        or target_col not in synth.columns
        or len(real) < 10
        or len(synth) < 10
    ):
        return default_result

    # Ensure target has at least 2 unique classes in real data
    y_real_raw = real[target_col].dropna()
    unique_classes = np.unique(y_real_raw)
    if len(unique_classes) < 2:
        return default_result

    # Align rows with non-null target
    real_clean = real.dropna(subset=[target_col]).copy()
    synth_clean = synth.dropna(subset=[target_col]).copy()
    if len(real_clean) < 10 or len(synth_clean) < 10:
        return default_result

    # Encode target column
    le = LabelEncoder()
    y_real = le.fit_transform(real_clean[target_col].astype(str))
    n_classes = len(le.classes_)

    # Filter synthetic rows to only those whose target was seen in training
    known_classes = set(le.classes_)
    synth_mask = synth_clean[target_col].astype(str).isin(known_classes)
    synth_clean = synth_clean[synth_mask].copy()
    if len(synth_clean) < 5:
        return default_result

    y_synth = le.transform(synth_clean[target_col].astype(str))

    # Prepare feature sets
    X_real = real_clean.drop(columns=[target_col]).copy()
    X_synth = synth_clean.drop(columns=[target_col]).copy()

    # Identify categorical/non-numeric features and encode them
    cat_cols = [c for c in X_real.columns if not pd.api.types.is_numeric_dtype(X_real[c])]
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_real[cat_cols] = oe.fit_transform(X_real[cat_cols].astype(str))
        X_synth[cat_cols] = oe.transform(X_synth[cat_cols].astype(str))

    X_real = X_real.fillna(0.0)
    X_synth = X_synth.fillna(0.0)

    # 80/20 train/test split on real dataset
    try:
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_real,
            y_real,
            test_size=0.2,
            random_state=42,
            stratify=y_real if len(unique_classes) <= 10 else None,
        )
    except ValueError:
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42
        )

    # Check that test set contains >= 2 classes
    if len(np.unique(y_test_r)) < 2:
        return default_result

    # Train TRTR model
    clf_trtr = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf_trtr.fit(X_train_r, y_train_r)

    # Train TSTR model
    clf_tstr = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf_tstr.fit(X_synth, y_synth)

    # Predict probabilities on the SAME real holdout
    def get_auc(clf: RandomForestClassifier) -> float:
        probas = clf.predict_proba(X_test_r)
        # If clf did not observe all target classes during training, align probability array
        if probas.shape[1] < n_classes:
            aligned_probas = np.zeros((len(X_test_r), n_classes))
            for i, c in enumerate(clf.classes_):
                aligned_probas[:, c] = probas[:, i]
            probas = aligned_probas

        if n_classes == 2:
            return float(roc_auc_score(y_test_r, probas[:, 1]))
        else:
            return float(roc_auc_score(y_test_r, probas, multi_class="ovr"))

    try:
        trtr_auc = get_auc(clf_trtr)
        tstr_auc = get_auc(clf_tstr)
        auc_drop = trtr_auc - tstr_auc
    except Exception:
        return default_result

    return {
        "trtr_auc": float(round(trtr_auc, 4)),
        "tstr_auc": float(round(tstr_auc, 4)),
        "auc_drop": float(round(auc_drop, 4)),
    }


def evaluate_all(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    eda_summary: dict[str, Any],
    thresholds: dict[str, Any],
    iteration: int,
) -> dict[str, Any]:
    """Executes all 5 evaluation metrics and compiles Data Contract 4.3 output.

    Args:
        real: Real training DataFrame.
        synth: Synthetic generated DataFrame.
        eda_summary: Output dictionary from EDAAnalyzer.analyze() (Data Contract 4.1).
        thresholds: Dictionary of metric thresholds (from Settings.default_thresholds or override).
        iteration: Current loop iteration integer.

    Returns:
        dict[str, Any]: Comprehensive evaluation report conforming to Data Contract 4.3.
    """
    # Extract continuous/numeric column names
    columns_meta = eda_summary.get("columns", {})
    numeric_cols = [
        c for c, meta in columns_meta.items() if meta.get("type") == "continuous"
    ]
    if not numeric_cols:
        numeric_cols = list(real.select_dtypes(include=[np.number]).columns)

    target_col = eda_summary.get("target_column")

    # 1. Kolmogorov-Smirnov per column
    ks_results = compute_ks_per_column(real, synth, numeric_cols)

    # 2. Correlation difference Frobenius norm
    corr_diff = compute_correlation_diff(real, synth, numeric_cols)

    # 3. Class balance Jensen-Shannon divergence
    class_balance_js = 0.0
    if target_col and target_col in real.columns and target_col in synth.columns:
        class_balance_js = compute_class_balance_js(real[target_col], synth[target_col])

    # 4. Privacy DCR 5th percentile
    dcr_percentile = float(thresholds.get("dcr_min_percentile", 5))
    privacy_dcr = compute_privacy_dcr(real, synth, numeric_cols, percentile=dcr_percentile)

    # 5. ML Utility TRTR vs TSTR
    utility = compute_ml_utility(real, synth, target_col)

    # Threshold checks
    ks_threshold = float(thresholds.get("ks_stat_max", 0.15))
    corr_threshold = float(thresholds.get("corr_diff_max", 0.20))
    js_threshold = float(thresholds.get("js_divergence_max", 0.10))
    dcr_floor = float(thresholds.get("dcr_safe_floor", 0.01))
    auc_drop_threshold = float(thresholds.get("utility_auc_drop_max", 0.10))

    ks_mean = float(np.mean(list(ks_results.values()))) if ks_results else 0.0
    ks_passed = ks_mean <= ks_threshold
    corr_passed = corr_diff <= corr_threshold
    balance_passed = class_balance_js <= js_threshold
    privacy_passed = privacy_dcr >= dcr_floor
    utility_passed = utility["auc_drop"] <= auc_drop_threshold

    passed_dict = {
        "ks": bool(ks_passed),
        "correlation": bool(corr_passed),
        "balance": bool(balance_passed),
        "privacy": bool(privacy_passed),
        "utility": bool(utility_passed),
    }

    overall_passed = bool(all(passed_dict.values()))

    return {
        "iteration": int(iteration),
        "per_column_ks": ks_results,
        "correlation_diff_frobenius": corr_diff,
        "class_balance_js_divergence": class_balance_js,
        "privacy_dcr_5th_percentile": privacy_dcr,
        "utility": utility,
        "thresholds": dict(thresholds),
        "passed": passed_dict,
        "overall_passed": overall_passed,
    }
