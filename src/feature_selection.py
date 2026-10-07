"""Feature selection with a genetic algorithm (sklearn-genetic-opt)."""
from __future__ import annotations

import random

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn_genetic import GAFeatureSelectionCV


def encode_full(preprocessor: ColumnTransformer, X_train, X_test):
    """Fit the preprocessor on train, transform both, return dense arrays + names."""
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    names = preprocessor.get_feature_names_out()
    if hasattr(X_train_enc, "toarray"):
        X_train_enc, X_test_enc = X_train_enc.toarray(), X_test_enc.toarray()
    return np.asarray(X_train_enc), np.asarray(X_test_enc), names


def run_ga_feature_selection(X_train_enc, y_train, estimator, population_size: int = 10,
                             generations: int = 8, cv: int = 3, scoring: str = "f1",
                             seed: int = 42) -> GAFeatureSelectionCV:
    """`estimator` can be a SMOTE + classifier pipeline, then SMOTE stays inside the folds."""
    # the GA uses the global random generators (no random_state argument)
    random.seed(seed)
    np.random.seed(seed)
    ga = GAFeatureSelectionCV(
        estimator=estimator, cv=cv, scoring=scoring,
        population_size=population_size, generations=generations,
        crossover_probability=0.8, mutation_probability=0.2,
        criteria="max", n_jobs=1, verbose=False,
    )
    ga.fit(X_train_enc, np.asarray(y_train))
    return ga


def selected_feature_names(ga: GAFeatureSelectionCV, feature_names) -> list[str]:
    return list(np.asarray(feature_names)[ga.support_])
