"""
Çoklu model eğitimi + hyperparameter tuning.

Dört model paralel olarak eğitiliyor ve karşılaştırılıyor: GradientBoosting,
Random Forest, XGBoost, CatBoost. Süre/verimlilik için tam ızgara taraması
yerine RandomizedSearchCV ile daha küçük ama temsili bir arama uzayı
taranıyor (SMOTE train-only + F1 objective + CV).
"""
from __future__ import annotations

from dataclasses import dataclass

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


@dataclass
class ModelResult:
    name: str
    best_estimator: object
    best_params: dict
    best_cv_f1: float


def _smote_pipeline(classifier, random_state: int = 42) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", classifier),
        ]
    )


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
        "estimator": XGBClassifier(
            random_state=42, eval_metric="logloss", n_jobs=-1
        ),
        "param_distributions": {
            "classifier__n_estimators": [100, 150, 200, 250, 300],
            "classifier__max_depth": [3, 4, 5, 6, 10],
            "classifier__learning_rate": [0.05, 0.1, 0.2, 0.3],
        },
    },
    "CatBoost": {
        "estimator": CatBoostClassifier(random_state=42, verbose=False),
        "param_distributions": {
            "classifier__iterations": [100, 150, 200, 250, 300],
            "classifier__depth": [3, 4, 5, 6, 10],
            "classifier__learning_rate": [0.05, 0.1, 0.2, 0.3],
        },
    },
}


def train_all_models(
    X_train,
    y_train,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> dict[str, ModelResult]:
    results: dict[str, ModelResult] = {}
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for name, cfg in MODEL_SEARCH_SPACE.items():
        pipe = _smote_pipeline(cfg["estimator"], random_state=random_state)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=cfg["param_distributions"],
            n_iter=n_iter,
            cv=cv_splitter,
            scoring="f1",
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        results[name] = ModelResult(
            name=name,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_cv_f1=search.best_score_,
        )
        print(f"[{name}] best CV F1 = {search.best_score_:.4f} | params = {search.best_params_}")

    return results
