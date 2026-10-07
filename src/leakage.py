"""Simple leakage checks.

single_feature_auc: CV AUC of a logistic regression on one column only. A
column that predicts churn almost perfectly on its own (AUC > ~0.95) is
suspicious.

ablation_drop: how much the AUC drops when a column is removed from the full
model. A very large drop for one column is also worth a look.

Both only point at columns to check; whether it is really leakage has to be
decided by knowing how the column was created.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df[cols], drop_first=True)


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def single_feature_auc(df: pd.DataFrame, feature_cols: list[str], target_col: str, cv: int = 5) -> pd.DataFrame:
    y = df[target_col]
    rows = []
    for col in feature_cols:
        X = _encode(df, [col])
        if X.shape[1] == 0:
            continue
        auc = cross_val_score(_lr(), X, y, cv=cv, scoring="roc_auc").mean()
        rows.append({"column": col, "solo_auc": auc})
    return pd.DataFrame(rows).sort_values("solo_auc", ascending=False).set_index("column")


def ablation_drop(df: pd.DataFrame, feature_cols: list[str], target_col: str, cv: int = 5) -> pd.DataFrame:
    y = df[target_col]
    baseline_auc = cross_val_score(_lr(), _encode(df, feature_cols), y, cv=cv, scoring="roc_auc").mean()

    rows = []
    for col in feature_cols:
        rest = [c for c in feature_cols if c != col]
        auc = cross_val_score(_lr(), _encode(df, rest), y, cv=cv, scoring="roc_auc").mean()
        rows.append({"removed_column": col, "auc_without_column": auc,
                     "auc_drop_vs_baseline": baseline_auc - auc})
    result = (pd.DataFrame(rows).sort_values("auc_drop_vs_baseline", ascending=False)
              .set_index("removed_column"))
    result.attrs["baseline_auc"] = baseline_auc
    return result
