"""Information Value based feature selection."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from creditScoring.preprocessing.woe import calculate_iv_for_dataframe


class IVFeatureSelector(BaseEstimator, TransformerMixin):
    """Select features whose IV is above configurable threshold."""

    def __init__(self, min_iv: float = 0.02):
        self.min_iv = min_iv
        self.selected_features_: list[str] = []
        self.iv_ranking_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.iv_ranking_ = calculate_iv_for_dataframe(X, y)
        self.selected_features_ = (
            self.iv_ranking_.loc[self.iv_ranking_["iv"] >= self.min_iv, "feature"].tolist()
        )
        return self

    def transform(self, X: pd.DataFrame):
        if not self.selected_features_:
            return X.copy()
        return X[self.selected_features_].copy()

    def get_ranking(self) -> pd.DataFrame:
        return self.iv_ranking_.copy() if self.iv_ranking_ is not None else pd.DataFrame(columns=["feature", "iv"])
