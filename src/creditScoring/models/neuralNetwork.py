"""Neural network model wrapper."""

import pandas as pd
from sklearn.neural_network import MLPClassifier

from .base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """Wrapper around sklearn MLPClassifier."""

    def __init__(self, **kwargs):
        default_kwargs = {"hidden_layer_sizes": (64, 32), "max_iter": 500, "random_state": 42, **kwargs}
        super().__init__(MLPClassifier(**default_kwargs))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)[:, 1]
