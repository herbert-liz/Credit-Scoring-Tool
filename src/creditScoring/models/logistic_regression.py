"""Logistic Regression model wrapper."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Wrapper around sklearn LogisticRegression."""

    def __init__(self, **kwargs):
        default_kwargs = {"solver": "lbfgs", "max_iter": 1000, **kwargs}
        super().__init__(LogisticRegression(**default_kwargs))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_[0], index=self.model.feature_names_in_)
