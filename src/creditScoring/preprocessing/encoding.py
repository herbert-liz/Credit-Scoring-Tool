"""Categorical encoding tools."""

from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible categorical encoder."""

    def __init__(self, method: Literal["label", "onehot", "frequency"] = "onehot"):
        self.method = method
        self.columns_: list[str] = []
        self.mapping_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.columns_ = X.select_dtypes(exclude=["number"]).columns.tolist()
        if self.method in {"label", "frequency"}:
            for col in self.columns_:
                if self.method == "label":
                    categories = X[col].astype("category").cat.categories
                    self.mapping_[col] = {cat: idx for idx, cat in enumerate(categories)}
                else:
                    self.mapping_[col] = X[col].value_counts(normalize=True, dropna=False).to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.method == "onehot":
            return pd.get_dummies(X, columns=self.columns_, drop_first=False)

        for col in self.columns_:
            mapping = self.mapping_.get(col, {})
            if self.method == "label":
                X[col] = X[col].map(mapping).fillna(-1).astype(int)
            elif self.method == "frequency":
                X[col] = X[col].map(mapping).fillna(0.0)
            else:
                raise ValueError(f"Unsupported encoding method: {self.method}")
        return X
