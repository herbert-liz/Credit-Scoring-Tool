"""Categorical encoding tools."""

from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible categorical encoder."""

    def __init__(self, method: Literal["target", "onehot", "frequency"] = "target"):
        self.method = method
        self.columns_: list[str] = []
        self.mapping_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.columns_ = X.select_dtypes(exclude=["number"]).columns.tolist()
        if self.method == "frequency":
            for col in self.columns_:
                self.mapping_[col] = X[col].value_counts(normalize=True, dropna=False).to_dict()
        elif self.method == "target":
            if y is None:
                raise ValueError("Target encoding requires y to be provided during fit.")
            y_series = pd.Series(y, index=X.index)
            self.global_mean_ = y_series.mean()
            for col in self.columns_:
                self.mapping_[col] = y_series.groupby(X[col]).mean().to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.method == "onehot":
            return pd.get_dummies(X, columns=self.columns_, drop_first=False)

        for col in self.columns_:
            mapping = self.mapping_.get(col, {})
            if self.method == "frequency":
                X[col] = X[col].map(mapping).fillna(0.0)
            elif self.method == "target":
                X[col] = X[col].map(mapping).fillna(self.global_mean_)
            else:
                raise ValueError(
                    f"Unsupported encoding method: {self.method}. "
                    f"Supported methods: target, onehot, frequency"
                )
        return X

