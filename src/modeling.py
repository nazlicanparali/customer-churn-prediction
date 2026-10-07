"""Train 4 models with SMOTE + RandomizedSearchCV (F1)."""
from __future__ import annotations

from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from .imbalance import smote_pipeline


@dataclass
class ModelResult:
    name: str
    best_estimator: object
    best_params: dict
    best_cv_f1: float


MODEL_SEARCH_SPACE = {
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_distributions": {
            "classifier__n_estimators": [100, 150, 200, 250, 300],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.05, 0.1, 0.2, 0.3],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_distributions": {
            "classifier__n_estimators": [100, 150, 200, 250, 300],
            "classifier__max_depth": [4, 6, 8, 10, 12],
            "classifier__max_features": ["sqrt", "log2", 0.5],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
        "param_distributions": {
            "classifier__n_estimators": [100, 150, 200, 250, 300],
            "classifier__max_depth": [3, 4, 5, 6, 10],
            "classifier__learning_rate": [0.05, 0.1, 0.2, 0.3],
        },
    },
    "CatBoost": {
        "estimator": CatBoostClassifier(random_state=42, verbose=False, allow_writing_files=False),
        "param_distributions": {
            "classifier__iterations": [100, 150, 200, 250, 300],
            "classifier__depth": [3, 4, 5, 6, 10],
            "classifier__learning_rate": [0.05, 0.1, 0.2, 0.3],
        },
    },
}


def train_all_models(X_train, y_train, n_iter: int = 20, cv: int = 3,
                     random_state: int = 42) -> dict[str, ModelResult]:
    results = {}
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for name, cfg in MODEL_SEARCH_SPACE.items():
        search = RandomizedSearchCV(
            smote_pipeline(cfg["estimator"], random_state), cfg["param_distributions"],
            n_iter=n_iter, cv=folds, scoring="f1", random_state=random_state, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        results[name] = ModelResult(name, search.best_estimator_, search.best_params_, search.best_score_)
        print(f"{name}: CV F1 = {search.best_score_:.4f}  {search.best_params_}")
    return results
