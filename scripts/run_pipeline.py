"""Runs the whole pipeline and writes the results to reports/ and models/.

    python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src import eda, evaluation as ev, feature_selection as fs, leakage, modeling as md  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.data_loading import ID_COL, TARGET_COL, load_clean  # noqa: E402

warnings.filterwarnings("ignore")

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"


def main():
    t0 = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("1) load + clean")
    df = pp.engineer_features(load_clean())
    print(f"   {df.shape}, churn rate {df[TARGET_COL].mean():.4f}")
    num_cols, cat_cols = pp.get_feature_lists(df, TARGET_COL, ID_COL)
    feature_cols = num_cols + cat_cols

    print("2) EDA tables")
    eda.missing_constant_report(df).to_csv(REPORTS_DIR / "missing_constant_report.csv")
    eda.cardinality_report(df, exclude=[ID_COL]).to_csv(REPORTS_DIR / "cardinality_report.csv")
    cat_assoc = eda.categorical_target_association(df, cat_cols, TARGET_COL)
    cat_assoc.to_csv(REPORTS_DIR / "categorical_target_association.csv")
    eda.numeric_target_association(df, num_cols, TARGET_COL).to_csv(REPORTS_DIR / "numeric_target_association.csv")
    eda.high_correlation_pairs(df, num_cols, 0.85).to_csv(REPORTS_DIR / "high_correlation_pairs.csv", index=False)
    print(cat_assoc.head(3))

    print("3) leakage checks")
    sf_auc = leakage.single_feature_auc(df, feature_cols, TARGET_COL, cv=5)
    ablation = leakage.ablation_drop(df, feature_cols, TARGET_COL, cv=5)
    sf_auc.to_csv(REPORTS_DIR / "leakage_single_feature_auc.csv")
    ablation.to_csv(REPORTS_DIR / "leakage_ablation_drop.csv")
    print(f"   baseline AUC {ablation.attrs['baseline_auc']:.4f}, "
          f"highest single-column AUC {sf_auc.index[0]} = {sf_auc.iloc[0, 0]:.3f}")

    print("4) split + encode")
    train_df, test_df = pp.stratified_split(df, TARGET_COL)
    X_train, y_train = train_df.drop(columns=[TARGET_COL, ID_COL]), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=[TARGET_COL, ID_COL]), test_df[TARGET_COL]
    preprocessor = pp.build_preprocessor(num_cols, cat_cols)
    X_train_enc, X_test_enc, feat_names = fs.encode_full(preprocessor, X_train, X_test)
    print(f"   train {X_train_enc.shape}, test {X_test_enc.shape}")

    print("5) GA feature selection (logistic regression as the fast model)")
    ga_estimator = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", LogisticRegression(max_iter=1000))])
    ga = fs.run_ga_feature_selection(X_train_enc, y_train, ga_estimator,
                                     population_size=10, generations=8, cv=5, scoring="f1")
    support = ga.support_
    selected = fs.selected_feature_names(ga, feat_names)
    print(f"   {support.sum()} / {len(feat_names)} features kept")
    with open(REPORTS_DIR / "selected_features.json", "w") as f:
        json.dump(selected, f, indent=2)
    X_train_sel, X_test_sel = X_train_enc[:, support], X_test_enc[:, support]

    print("6) models")
    results = md.train_all_models(X_train_sel, y_train, n_iter=25, cv=5)

    print("7) test set")
    comparison = ev.compare_models(results, X_test_sel, y_test)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv")
    print(comparison.round(3))
    ev.plot_model_comparison(comparison, save_path=str(FIGURES_DIR / "model_comparison.png"))

    # pick the model by CV F1, not by test F1
    best_name = comparison["cv_f1"].idxmax()
    best = results[best_name]

    print(f"8) threshold for {best_name} (out-of-fold on train)")
    oof = ev.oof_proba(best.best_estimator, X_train_sel, y_train)
    scan = ev.threshold_scan(y_train, oof)
    thr = ev.best_threshold_for_f1(y_train, oof)
    scan.to_csv(REPORTS_DIR / "threshold_scan.csv", index=False)
    ev.plot_threshold_tradeoff(scan, chosen=thr, save_path=str(FIGURES_DIR / "threshold_tradeoff.png"))
    test_proba = best.best_estimator.predict_proba(X_test_sel)[:, 1]
    test_at_thr = ev.evaluate_at_threshold(y_test, test_proba, thr)
    print(f"   threshold {thr}: {test_at_thr}")

    with open(MODELS_DIR / f"best_model_{best_name}.pkl", "wb") as f:
        pickle.dump({"model": best.best_estimator, "preprocessor": preprocessor,
                     "selected_feature_mask": support, "feature_names": feat_names,
                     "threshold": thr, "params": best.best_params}, f)

    summary = {
        "best_model": best_name,
        "best_model_params": best.best_params,
        "cv_f1": best.best_cv_f1,
        "test_at_0.5": comparison.loc[best_name].to_dict(),
        "chosen_threshold": thr,
        "test_at_chosen_threshold": test_at_thr,
        "n_features_encoded": int(len(feat_names)),
        "n_features_selected": int(support.sum()),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(REPORTS_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"done in {summary['runtime_seconds']} s")


if __name__ == "__main__":
    main()
