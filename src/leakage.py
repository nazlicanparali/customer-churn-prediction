"""
Leakage tespiti: tek-kolon ablation. Şüpheli bir kolonun gerçek bir
leakage kaynağı olup olmadığını iki açıdan test ediyoruz:

1. single_feature_auc: SADECE o kolonu kullanarak model kur, AUC'a bak.
   Bir kolon TEK BAŞINA çok yüksek AUC veriyorsa (örn. >0.97) bu genelde
   hedefin kendisinden türetilmiş ya da hedefle aynı anda/sonra oluşan
   bir bilgi taşıdığının işaretidir (leakage şüphesi).
2. ablation_drop: Tüm kolonlarla kurulan baseline modelden o TEK kolonu
   çıkar, metrik ne kadar düşüyor bak. Kolon çıkarılınca metrikte büyük
   bir düşüş görülmesi, o kolonun modele "hile yoluyla" baskın olduğunu
   gösterir.

Bu modül genel amaçlı: hangi kolonun hedefle aşırı güçlü tekil ilişkisi
olduğunu otomatik tespit eder, ama nihai karar (gerçekten leakage mi,
yoksa meşru güçlü bir feature mı) hâlâ domain bilgisiyle verilmelidir.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _encode_for_test(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Hızlı, tek amaçlı encoding (leakage testi için) - one-hot."""
    return pd.get_dummies(df[cols], drop_first=True)


def single_feature_auc(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, cv: int = 5
) -> pd.DataFrame:
    """Her kolonu TEK BAŞINA kullanarak CV ROC-AUC hesaplar."""
    y = df[target_col]
    rows = []
    for col in feature_cols:
        X = _encode_for_test(df, [col])
        if X.shape[1] == 0:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            rows.append({"column": col, "solo_auc": scores.mean()})
        except Exception:
            continue
    return (
        pd.DataFrame(rows)
        .sort_values("solo_auc", ascending=False)
        .set_index("column")
    )


def ablation_drop(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, cv: int = 5
) -> pd.DataFrame:
    """
    Baseline (tüm kolonlar) skorundan, her kolon tek tek çıkarıldığında
    oluşan AUC düşüşünü hesaplar. Büyük düşüş = o kolon modelin
    performansını orantısız şekilde taşıyor (leakage şüphesi ya da
    çok güçlü bir feature; ayrımı domain bilgisiyle yapılmalı).
    """
    y = df[target_col]
    X_full = _encode_for_test(df, feature_cols)
    baseline_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    baseline_auc = cross_val_score(X=X_full, y=y, estimator=baseline_model, cv=cv, scoring="roc_auc").mean()

    rows = []
    for col in feature_cols:
        remaining = [c for c in feature_cols if c != col]
        X_sub = _encode_for_test(df, remaining)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        auc = cross_val_score(model, X_sub, y, cv=cv, scoring="roc_auc").mean()
        rows.append(
            {
                "removed_column": col,
                "auc_without_column": auc,
                "auc_drop_vs_baseline": baseline_auc - auc,
            }
        )
    result = pd.DataFrame(rows).sort_values("auc_drop_vs_baseline", ascending=False).set_index(
        "removed_column"
    )
    result.attrs["baseline_auc"] = baseline_auc
    return result
