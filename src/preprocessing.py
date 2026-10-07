"""Feature engineering, encoding and the train/test split."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDON_SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # how many add-on services the customer has
    df["num_addon_services"] = df[ADDON_SERVICE_COLS].eq("Yes").sum(axis=1)

    # tenure buckets, new customers (first year) churn the most
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, 200],
        labels=["0-12m", "13-24m", "25-48m", "49-60m", "60m+"],
    ).astype(str)

    # TotalCharges vs tenure * MonthlyCharges: != 1 if the monthly price changed
    expected_total = df["tenure"] * df["MonthlyCharges"]
    df["charge_consistency_ratio"] = (df["TotalCharges"] + 1) / (expected_total + 1)
    return df


def get_feature_lists(df: pd.DataFrame, target_col: str, id_col: str) -> tuple[list[str], list[str]]:
    exclude = {target_col, id_col}
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    categorical_cols = [c for c in df.columns if c not in exclude and c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
    ])


def stratified_split(df: pd.DataFrame, target_col: str, test_size: float = 0.25, random_state: int = 42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[target_col],
                                         random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
