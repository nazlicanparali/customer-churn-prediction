"""
Dengesiz veri yönetimi (SMOTE, yalnızca train setinde / CV-leakage'siz).

SMOTE'u fold ayrımından ÖNCE uygularsanız sentetik örnekler train ve
validation fold'lara sızabilir (CV-leakage). imbalanced-learn'ün Pipeline
sınıfı bunu native olarak çözüyor: `imblearn.pipeline.Pipeline` içine
konan SMOTE adımı, GridSearchCV/cross_val_score her fold'u ayırdığında
SADECE o an eğitim için kullanılan fold'a uygulanır, validation fold'a
asla dokunmaz.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer


def build_smote_pipeline(
    preprocessor: ColumnTransformer,
    classifier,
    smote_k_neighbors: int = 5,
    random_state: int = 42,
) -> ImbPipeline:
    """
    preprocessor -> SMOTE -> classifier sırasıyla kurulan pipeline.
    Bu sıra kritik: SMOTE, preprocessor'dan (encoding/scaling) SONRA
    çalışmalı, çünkü SMOTE sayısal uzayda komşuluk hesaplıyor (kategorik
    ham veri üzerinde anlamsız olur).
    """
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "smote",
                SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state),
            ),
            ("classifier", classifier),
        ]
    )
