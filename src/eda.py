"""
Keşifsel Veri Analizi (EDA) ve istatistiksel testler.

EDA zinciri: Missing/Constant Filter -> Outlier Capping -> Cardinality ->
Target Association + Multicollinearity. Bu veri setinde eksik değer/sabit
kolon bulunmuyor (Telco seti temiz), ama fonksiyonlar genel amaçlı
yazıldı - kolon adı hardcode edilmeden herhangi bir tabular veri setine
uygulanabilir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1) Missing / Constant filtreleme
# ---------------------------------------------------------------------------
def missing_constant_report(
    df: pd.DataFrame,
    missing_thresh: float = 0.75,
    constant_thresh: float = 0.90,
) -> pd.DataFrame:
    """
    Her kolon için missing oranı ve en sık değerin oranını (constant ratio)
    hesaplar; standart eşiklerle (>%75 missing, >%90 tek değer) hangi
    kolonların elenmesi gerektiğini işaretler.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        missing_ratio = df[col].isna().mean()
        top_value_ratio = df[col].value_counts(normalize=True, dropna=True).max() if n else 0
        rows.append(
            {
                "column": col,
                "missing_ratio": missing_ratio,
                "top_value_ratio": top_value_ratio,
                "drop_missing": missing_ratio > missing_thresh,
                "drop_constant": top_value_ratio > constant_thresh,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def apply_missing_constant_filter(
    df: pd.DataFrame,
    missing_thresh: float = 0.75,
    constant_thresh: float = 0.90,
) -> tuple[pd.DataFrame, list[str]]:
    report = missing_constant_report(df, missing_thresh, constant_thresh)
    drop_cols = report.index[report["drop_missing"] | report["drop_constant"]].tolist()
    return df.drop(columns=drop_cols), drop_cols


# ---------------------------------------------------------------------------
# 2) Outlier capping (satır silme yok)
# ---------------------------------------------------------------------------
def cap_outliers_iqr(
    df: pd.DataFrame, numeric_cols: list[str], k: float = 1.5
) -> pd.DataFrame:
    """
    IQR tabanlı capping (winsorization). Satır silinmez, sadece uç
    değerler alt/üst sınıra çekilir ("closest permitted value" mantığı).
    """
    df = df.copy()
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ---------------------------------------------------------------------------
# 3) Cardinality analizi
# ---------------------------------------------------------------------------
def cardinality_report(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    exclude = exclude or []
    cols = [c for c in df.columns if c not in exclude]
    rows = []
    for col in cols:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": df[col].nunique(),
                "kind": "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical",
            }
        )
    return pd.DataFrame(rows).set_index("column")


# ---------------------------------------------------------------------------
# 4) Hedef ile ilişki testleri
# ---------------------------------------------------------------------------
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Cramér's V - iki kategorik değişken arasındaki ilişki gücü (0-1).
    """
    confusion = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((k_corr - 1), (r_corr - 1))
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def categorical_target_association(
    df: pd.DataFrame, cat_cols: list[str], target_col: str
) -> pd.DataFrame:
    rows = []
    for col in cat_cols:
        v = cramers_v(df[col], df[target_col])
        rows.append({"column": col, "cramers_v": v})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).set_index("column")


def numeric_target_association(
    df: pd.DataFrame, num_cols: list[str], target_col: str
) -> pd.DataFrame:
    """
    Sayısal değişken - ikili hedef ilişkisi için point-biserial korelasyon
    (Welch's t-test ile eşdeğer bilgiyi verir).
    """
    rows = []
    for col in num_cols:
        corr, p_value = stats.pointbiserialr(df[target_col], df[col])
        rows.append({"column": col, "point_biserial_r": corr, "p_value": p_value})
    return (
        pd.DataFrame(rows)
        .assign(abs_r=lambda d: d["point_biserial_r"].abs())
        .sort_values("abs_r", ascending=False)
        .drop(columns="abs_r")
        .set_index("column")
    )


# ---------------------------------------------------------------------------
# 5) Multicollinearity kontrolü
# ---------------------------------------------------------------------------
def high_correlation_pairs(
    df: pd.DataFrame, num_cols: list[str], threshold: float = 0.85
) -> pd.DataFrame:
    """
    Sayısal kolonlar arası yüksek korelasyonlu (>|threshold|) çiftleri
    listeler - multicollinearity riski taşıyan kolon çiftlerinin tespit
    adımı.
    """
    corr = df[num_cols].corr().abs()
    pairs = []
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1 :]:
            r = corr.loc[c1, c2]
            if r > threshold:
                pairs.append({"col_1": c1, "col_2": c2, "abs_corr": r})
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False) if pairs else pd.DataFrame(
        columns=["col_1", "col_2", "abs_corr"]
    )
