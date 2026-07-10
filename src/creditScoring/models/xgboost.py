"""XGBoost model wrapper (optional dependency)."""

import pandas as pd

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """Wrapper around xgboost.XGBClassifier with graceful dependency checks."""

    def __init__(self, **kwargs):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Install extras: pip install credit-scoring-tool[all]") from exc

        default_kwargs = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
            **kwargs,
        }
        super().__init__(XGBClassifier(**default_kwargs))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)[:, 1]
