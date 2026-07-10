"""Random Forest model wrapper."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Wrapper around sklearn RandomForestClassifier."""

    def __init__(self, **kwargs):
        default_kwargs = {"n_estimators": 200, "random_state": 42, **kwargs}
        super().__init__(RandomForestClassifier(**default_kwargs))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.model.feature_names_in_)
