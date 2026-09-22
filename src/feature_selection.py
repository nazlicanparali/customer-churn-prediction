"""
Feature Selection - Genetik Algoritma.

sklearn-genetic-opt kütüphanesinin GAFeatureSelectionCV'si, F1'i objective
olarak kullanarak hangi feature alt kümesinin en iyi CV skorunu verdiğini
genetik algoritma ile arar (popülasyon, nesil, seçilim/çaprazlama/mutasyon
mantığıyla).
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.space import Integer


def encode_full(preprocessor: ColumnTransformer, X_train, X_test):
    """Preprocessor'ı train'e fit eder, train/test'i dönüştürür ve
    encoded feature isimlerini döner (OneHot sonrası kolon adları)."""
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    # Bazı sparse çıktı durumlarına karşı yoğun (dense) array garantisi
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
        X_test_enc = X_test_enc.toarray()
    return np.asarray(X_train_enc), np.asarray(X_test_enc), feature_names


def run_ga_feature_selection(
    X_train_enc: np.ndarray,
    y_train,
    estimator,
    population_size: int = 10,
    generations: int = 8,
    cv: int = 3,
    scoring: str = "f1",
    random_state: int = 42,
) -> GAFeatureSelectionCV:
    """
    GA ile feature selection çalıştırır. `estimator`, SMOTE + classifier
    içeren bir imblearn Pipeline olabilir - böylece her CV fold'unda
    hem feature alt kümesi hem de SMOTE train-only kuralı aynı anda
    korunur.
    """
    ga = GAFeatureSelectionCV(
        estimator=estimator,
        cv=cv,
        scoring=scoring,
        population_size=population_size,
        generations=generations,
        crossover_probability=0.8,
        mutation_probability=0.2,
        criteria="max",
        n_jobs=1,
        verbose=False,
    )
    ga.fit(X_train_enc, np.asarray(y_train))
    return ga


def selected_feature_names(ga: GAFeatureSelectionCV, feature_names: np.ndarray) -> list[str]:
    support = ga.support_
    return list(np.asarray(feature_names)[support])
