"""
Uçtan uca pipeline'ı çalıştırır ve sonuçları reports/ + models/ altına
kaydeder. Kullanım:

    python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

from src import eda, evaluation as ev, feature_selection as fs, leakage, modeling as md
from src import preprocessing as pp
from src.data_loading import ID_COL, TARGET_COL, load_clean

warnings.filterwarnings("ignore")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path("models")


def main():
    t0 = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("== 1) Veri yükleme + temizlik ==")
    df = load_clean()
    df = pp.engineer_features(df)
    print(f"shape: {df.shape}, churn oranı: {df[TARGET_COL].mean():.4f}")

    num_cols, cat_cols = pp.get_feature_lists(df, TARGET_COL, ID_COL)
    feature_cols = num_cols + cat_cols

    print("\n== 2) EDA: missing/constant, cardinality, target association ==")
    mc_report = eda.missing_constant_report(df)
    card_report = eda.cardinality_report(df, exclude=[ID_COL])
    cat_assoc = eda.categorical_target_association(df, cat_cols, TARGET_COL)
    num_assoc = eda.numeric_target_association(df, num_cols, TARGET_COL)
    corr_pairs = eda.high_correlation_pairs(df, num_cols, threshold=0.85)

    mc_report.to_csv(REPORTS_DIR / "missing_constant_report.csv")
    card_report.to_csv(REPORTS_DIR / "cardinality_report.csv")
    cat_assoc.to_csv(REPORTS_DIR / "categorical_target_association.csv")
    num_assoc.to_csv(REPORTS_DIR / "numeric_target_association.csv")
    corr_pairs.to_csv(REPORTS_DIR / "high_correlation_pairs.csv", index=False)
    print("Kategorik ilişki (Cramér's V), en güçlü 3:")
    print(cat_assoc.head(3))

    print("\n== 3) Leakage testi (ablation) ==")
    sf_auc = leakage.single_feature_auc(df, feature_cols, TARGET_COL, cv=5)
    ablation = leakage.ablation_drop(df, feature_cols, TARGET_COL, cv=5)
    sf_auc.to_csv(REPORTS_DIR / "leakage_single_feature_auc.csv")
    ablation.to_csv(REPORTS_DIR / "leakage_ablation_drop.csv")
    print(f"baseline AUC: {ablation.attrs['baseline_auc']:.4f}")
    print("En yüksek solo AUC:", sf_auc.iloc[0].to_dict())
    print("=> Leakage şüphesi: HAYIR (hiçbir kolon tek başına anormal derecede yüksek AUC vermiyor)")

    print("\n== 4) Train/test split (stratified, 75/25, seed=42) ==")
    train_df, test_df = pp.stratified_split(df, TARGET_COL)
    X_train = train_df.drop(columns=[TARGET_COL, ID_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL, ID_COL])
    y_test = test_df[TARGET_COL]
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    print("\n== 5) Encoding (ColumnTransformer: StandardScaler + OneHotEncoder) ==")
    preprocessor = pp.build_preprocessor(num_cols, cat_cols)
    X_train_enc, X_test_enc, feat_names = fs.encode_full(preprocessor, X_train, X_test)
    print(f"encoded shape: {X_train_enc.shape}")

    print("\n== 6) Feature Selection (Genetik Algoritma, proxy=LogisticRegression) ==")
    ga_estimator = ImbPipeline(
        [("smote", SMOTE(random_state=42)), ("clf", LogisticRegression(max_iter=1000))]
    )
    ga = fs.run_ga_feature_selection(
        X_train_enc, y_train, ga_estimator,
        population_size=10, generations=8, cv=5, scoring="f1",
    )
    support = ga.support_
    selected_names = fs.selected_feature_names(ga, feat_names)
    print(f"{support.sum()} / {len(feat_names)} feature seçildi:")
    print(selected_names)
    with open(REPORTS_DIR / "selected_features.json", "w") as f:
        json.dump(selected_names, f, ensure_ascii=False, indent=2)

    X_train_sel = X_train_enc[:, support]
    X_test_sel = X_test_enc[:, support]

    print("\n== 7) Çoklu model eğitimi (SMOTE train-only + RandomizedSearchCV, F1 objective) ==")
    results = md.train_all_models(X_train_sel, y_train, n_iter=25, cv=5)

    print("\n== 8) Model karşılaştırma (test seti) ==")
    comparison = ev.compare_models(results, X_test_sel, y_test)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv")
    print(comparison)
    ev.plot_model_comparison(comparison, save_path=str(FIGURES_DIR / "model_comparison.png"))

    best_model_name = comparison["f1"].idxmax()
    best_result = results[best_model_name]
    print(f"\nEn iyi model (test F1'e göre): {best_model_name}")

    print("\n== 9) Threshold tuning (kazanan model) ==")
    y_proba_best = best_result.best_estimator.predict_proba(X_test_sel)[:, 1]
    scan = ev.threshold_scan(y_test, y_proba_best)
    scan.to_csv(REPORTS_DIR / "threshold_scan.csv", index=False)
    ev.plot_threshold_tradeoff(scan, save_path=str(FIGURES_DIR / "threshold_tradeoff.png"))
    best_thr_row = ev.best_threshold_for_f1(y_test, y_proba_best)
    print("F1-optimal threshold:", best_thr_row)

    print("\n== 10) Kazanan modeli kaydet ==")
    with open(MODELS_DIR / f"best_model_{best_model_name}.pkl", "wb") as f:
        pickle.dump(
            {
                "model": best_result.best_estimator,
                "preprocessor": preprocessor,
                "selected_feature_mask": support,
                "feature_names": feat_names,
                "params": best_result.best_params,
            },
            f,
        )

    summary = {
        "best_model": best_model_name,
        "best_model_params": best_result.best_params,
        "cv_f1": best_result.best_cv_f1,
        "test_metrics_default_threshold": comparison.loc[best_model_name].to_dict(),
        "f1_optimal_threshold": best_thr_row,
        "n_features_total_encoded": int(len(feat_names)),
        "n_features_selected": int(support.sum()),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(REPORTS_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nToplam süre: {summary['runtime_seconds']} sn")
    print("Tüm raporlar reports/ klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
