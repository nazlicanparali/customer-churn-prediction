"""EDA helpers: missing/constant columns, cardinality, association with the
target and correlated pairs. Written without column names so they can be
reused on other tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def missing_constant_report(df: pd.DataFrame, missing_thresh: float = 0.75,
                            constant_thresh: float = 0.90) -> pd.DataFrame:
    """Missing ratio and share of the most common value for each column."""
    rows = []
    for col in df.columns:
        missing_ratio = df[col].isna().mean()
        top_ratio = df[col].value_counts(normalize=True, dropna=True).max() if len(df) else 0
        rows.append({
            "column": col,
            "missing_ratio": missing_ratio,
            "top_value_ratio": top_ratio,
            "drop_missing": missing_ratio > missing_thresh,
            "drop_constant": top_ratio > constant_thresh,
        })
    return pd.DataFrame(rows).set_index("column")


def apply_missing_constant_filter(df: pd.DataFrame, missing_thresh: float = 0.75,
                                  constant_thresh: float = 0.90) -> tuple[pd.DataFrame, list[str]]:
    report = missing_constant_report(df, missing_thresh, constant_thresh)
    drop_cols = report.index[report["drop_missing"] | report["drop_constant"]].tolist()
    return df.drop(columns=drop_cols), drop_cols


def cap_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str], k: float = 1.5) -> pd.DataFrame:
    """Clip values outside [Q1 - k*IQR, Q3 + k*IQR]. Not needed for this
    dataset (no outliers in the 3 numeric columns), kept for reuse."""
    df = df.copy()
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - k * iqr, upper=q3 + k * iqr)
    return df


def cardinality_report(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    exclude = exclude or []
    rows = [{
        "column": col,
        "dtype": str(df[col].dtype),
        "n_unique": df[col].nunique(),
        "kind": "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical",
    } for col in df.columns if col not in exclude]
    return pd.DataFrame(rows).set_index("column")


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    confusion = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def categorical_target_association(df: pd.DataFrame, cat_cols: list[str], target_col: str) -> pd.DataFrame:
    rows = [{"column": c, "cramers_v": cramers_v(df[c], df[target_col])} for c in cat_cols]
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).set_index("column")


def numeric_target_association(df: pd.DataFrame, num_cols: list[str], target_col: str) -> pd.DataFrame:
    """Point-biserial correlation of each numeric column with the 0/1 target."""
    rows = []
    for col in num_cols:
        r, p = stats.pointbiserialr(df[target_col], df[col])
        rows.append({"column": col, "point_biserial_r": r, "p_value": p})
    out = pd.DataFrame(rows)
    return out.reindex(out["point_biserial_r"].abs().sort_values(ascending=False).index).set_index("column")


def high_correlation_pairs(df: pd.DataFrame, num_cols: list[str], threshold: float = 0.85) -> pd.DataFrame:
    corr = df[num_cols].corr().abs()
    pairs = [{"col_1": c1, "col_2": c2, "abs_corr": corr.loc[c1, c2]}
             for i, c1 in enumerate(num_cols) for c2 in num_cols[i + 1:]
             if corr.loc[c1, c2] > threshold]
    if not pairs:
        return pd.DataFrame(columns=["col_1", "col_2", "abs_corr"])
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False)
