"""
Feature engineering ve ön işleme.

Ham kolonlardan iş anlamı olan yeni değişkenler türetiyoruz: add-on
hizmet kolonlarından bir "toplam hizmet sayısı" ve tenure'dan iş anlamı
olan bir risk segmenti.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDON_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def _addon_services_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Ek hizmet kolonlarını 'Yes'=1 / diğer=0 maskesine çevirir."""
    return df[ADDON_SERVICE_COLS].eq("Yes").astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Toplam abone olunan ek hizmet sayısı (bir ürün/hizmet kolon
    # ailesinden toplam sayım türetimine benzer bir agregasyon)
    df["num_addon_services"] = _addon_services_mask(df).sum(axis=1)

    # Tenure'u iş anlamı olan segmentlere ayır (0-12 ay: yeni müşteri /
    # yüksek risk, 60+ ay: sadık müşteri) - zaman temelli bir risk kategorisi
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 200],
        labels=["0-12ay", "13-24ay", "25-48ay", "49-60ay", "60ay+"],
    ).astype(str)

    # Müşteri başına ortalama aylık harcama tutarlılığı: TotalCharges ile
    # tenure*MonthlyCharges arasındaki fark - anormal/tutarsız faturalama
    # örüntülerini yakalamak için bir veri tutarlılığı kontrolü
    expected_total = df["tenure"] * df["MonthlyCharges"]
    df["charge_consistency_ratio"] = (df["TotalCharges"] + 1) / (expected_total + 1)

    return df


def get_feature_lists(df: pd.DataFrame, target_col: str, id_col: str) -> tuple[list[str], list[str]]:
    """Encoding için sayısal / kategorik kolon listelerini döner."""
    exclude = {target_col, id_col}
    numeric_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in exclude
    ]
    categorical_cols = [
        c for c in df.columns if c not in exclude and c not in numeric_cols
    ]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    scikit-learn ColumnTransformer: sayısal kolonlar StandardScaler,
    kategorik kolonlar OneHotEncoder ile dönüştürülür - kolon adı hardcode
    edilmeden, tipe göre dinamik olarak işlenir.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                categorical_cols,
            ),
        ]
    )


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    %75/25, stratified (hedef dağılımı korunarak), seed=42.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
