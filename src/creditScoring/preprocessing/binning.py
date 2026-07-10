"""Binning/discretization strategies for credit scoring."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier


def equal_width_binning(series: pd.Series, n_bins: int = 5) -> pd.Series:
    return pd.cut(series, bins=n_bins, duplicates="drop")


def equal_frequency_binning(series: pd.Series, n_bins: int = 5) -> pd.Series:
    return pd.qcut(series, q=min(n_bins, series.nunique()), duplicates="drop")


def supervised_binning(series: pd.Series, target: pd.Series, max_bins: int = 5) -> pd.Series:
    data = pd.DataFrame({"x": series, "y": target}).dropna()
    if data["x"].nunique() <= 2:
        return pd.cut(series, bins=min(2, series.nunique()), duplicates="drop")

    tree = DecisionTreeClassifier(max_leaf_nodes=max_bins, min_samples_leaf=max(10, int(len(data) * 0.05)), random_state=42)
    tree.fit(data[["x"]], data["y"])
    thresholds = sorted(t for t in tree.tree_.threshold if t != -2)
    bins = [-np.inf, *thresholds, np.inf]
    return pd.cut(series, bins=bins, duplicates="drop")


def monotonic_binning(series: pd.Series, target: pd.Series, n_bins: int = 5) -> pd.Series:
    binned = equal_frequency_binning(series, n_bins=n_bins)
    stats = pd.DataFrame({"bin": binned, "target": target}).dropna().groupby("bin", observed=False)["target"].mean()
    ordered_bins = {bin_label: idx for idx, bin_label in enumerate(stats.sort_values().index)}
    return binned.map(ordered_bins)


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Apply selected binning strategy to numeric columns."""

    def __init__(self, method: str = "quantile", n_bins: int = 5):
        self.method = method
        self.n_bins = n_bins
        self.numeric_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_columns_ = X.select_dtypes(include=["number"]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        data = X.copy()
        for col in self.numeric_columns_:
            if self.method == "equal_width":
                data[col] = equal_width_binning(data[col], self.n_bins)
            elif self.method == "quantile":
                data[col] = equal_frequency_binning(data[col], self.n_bins)
            elif self.method == "supervised":
                if y is None:
                    raise ValueError("y is required for supervised binning")
                data[col] = supervised_binning(data[col], y, self.n_bins)
            elif self.method == "monotonic":
                if y is None:
                    raise ValueError("y is required for monotonic binning")
                data[col] = monotonic_binning(data[col], y, self.n_bins)
            else:
                raise ValueError(
                    f"Unsupported binning method: {self.method}. Supported methods: equal_width, quantile, supervised, monotonic"
                )
        return data
