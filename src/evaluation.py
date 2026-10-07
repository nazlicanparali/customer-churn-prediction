"""Metrics, threshold choice and plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def evaluate_at_threshold(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 2),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
    }


def threshold_scan(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    return pd.DataFrame([evaluate_at_threshold(y_true, y_proba, t) for t in thresholds])


def oof_proba(estimator, X_train, y_train, cv: int = 5, random_state: int = 42) -> np.ndarray:
    """Out-of-fold probabilities on the training set, used to pick the threshold
    so the test set is not used for it."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return cross_val_predict(clone(estimator), X_train, y_train, cv=folds, method="predict_proba")[:, 1]


def best_threshold_for_f1(y_true, y_proba, thresholds=None) -> float:
    scan = threshold_scan(y_true, y_proba, thresholds)
    return float(scan.loc[scan["f1"].idxmax(), "threshold"])


def compare_models(model_results: dict, X_test, y_test) -> pd.DataFrame:
    """Test set metrics of each model at threshold 0.5."""
    rows = []
    for name, res in model_results.items():
        m = evaluate_at_threshold(y_test, res.best_estimator.predict_proba(X_test)[:, 1], 0.5)
        m.update(model=name, cv_f1=res.best_cv_f1)
        rows.append(m)
    return pd.DataFrame(rows).set_index("model")[["cv_f1", "auc", "recall", "precision", "f1", "accuracy"]]


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str | None = None):
    ax = comparison_df[["recall", "precision", "f1", "auc"]].plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("score")
    ax.set_title("Test set, threshold = 0.5")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax


def plot_threshold_tradeoff(scan_df: pd.DataFrame, chosen: float | None = None, save_path: str | None = None):
    ax = scan_df.plot(x="threshold", y=["recall", "precision", "f1"], figsize=(8, 5))
    if chosen is not None:
        ax.axvline(chosen, color="grey", ls="--", lw=1)
    ax.set_ylabel("score")
    ax.set_title("Threshold vs recall / precision / F1 (out-of-fold, train)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax
