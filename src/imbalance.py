"""SMOTE inside an imblearn Pipeline.

If SMOTE is applied before cross-validation, synthetic rows built from
validation rows end up in training. Inside an imblearn Pipeline it only runs
on the training part of each fold, and never on the test set.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def smote_pipeline(classifier, random_state: int = 42) -> ImbPipeline:
    return ImbPipeline([("smote", SMOTE(random_state=random_state)), ("classifier", classifier)])
