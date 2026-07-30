"""Base model abstraction for credit scoring models."""

from abc import ABC, abstractmethod

import joblib
import pandas as pd


class BaseModel(ABC):
    """Abstract model interface used by all wrapped estimators."""

    def __init__(self, model=None):
        self.model = model

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame):
        raise NotImplementedError

    def get_params(self) -> dict:
        return self.model.get_params() if self.model is not None else {}

    def set_params(self, **params):
        if self.model is None:
            raise ValueError("Model has not been initialized")
        self.model.set_params(**params)
        return self

    def save_model(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load_model(cls, path: str):
        return joblib.load(path)
