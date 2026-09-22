"""
Değerlendirme: metrikler, threshold tuning, model karşılaştırması.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_at_threshold(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
    }


def threshold_scan(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Farklı threshold değerlerinde recall/precision/F1 trade-off'unu tarar."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.65, 0.05)
    rows = [evaluate_at_threshold(y_true, y_proba, t) for t in thresholds]
    return pd.DataFrame(rows)


def best_threshold_for_f1(y_true, y_proba, thresholds=None) -> dict:
    scan = threshold_scan(y_true, y_proba, thresholds)
    return scan.loc[scan["f1"].idxmax()].to_dict()


def compare_models(model_results: dict, X_test, y_test) -> pd.DataFrame:
    """Her modelin test setindeki (default 0.5 threshold) metriklerini
    tek bir tabloda karşılaştırır."""
    rows = []
    for name, result in model_results.items():
        estimator = result.best_estimator
        y_proba = estimator.predict_proba(X_test)[:, 1]
        metrics = evaluate_at_threshold(y_test, y_proba, threshold=0.5)
        metrics["model"] = name
        metrics["cv_f1"] = result.best_cv_f1
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("model")[
        ["cv_f1", "auc", "recall", "precision", "f1", "accuracy", "threshold"]
    ]


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str | None = None):
    metrics_to_plot = ["recall", "precision", "f1", "auc"]
    ax = comparison_df[metrics_to_plot].plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Skor")
    ax.set_title("Model Karşılaştırması (test seti, threshold=0.5)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax


def plot_threshold_tradeoff(scan_df: pd.DataFrame, save_path: str | None = None):
    ax = scan_df.plot(x="threshold", y=["recall", "precision", "f1"], figsize=(8, 5))
    ax.set_ylabel("Skor")
    ax.set_title("Threshold - Recall/Precision/F1 Trade-off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax
