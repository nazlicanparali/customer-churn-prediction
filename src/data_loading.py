"""Load the Telco churn CSV and do the basic cleaning.

TotalCharges is stored as text and is a blank string for the 11 customers
with tenure = 0 (they haven't been billed yet).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # so it works from notebooks/ too
RAW_PATH = str(PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv")
TARGET_COL = "Churn"
ID_COL = "customerID"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # tenure 0 -> nothing paid yet, so 0 is the real value, not a missing one
    zero_tenure = (df["tenure"] == 0) & df["TotalCharges"].isna()
    df.loc[zero_tenure, "TotalCharges"] = 0.0
    if df["TotalCharges"].isna().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    # treat SeniorCitizen like the other yes/no columns
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def load_clean(path: str = RAW_PATH) -> pd.DataFrame:
    return basic_clean(load_raw(path))


if __name__ == "__main__":
    data = load_clean()
    print(data.shape)
    print(data[TARGET_COL].value_counts(normalize=True))
