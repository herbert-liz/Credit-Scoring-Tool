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


def monotonic_binning(
    series: pd.Series,
    target: pd.Series,
    n_bins: int = 5,
    min_bin_pct: float = 0.05,
) -> pd.Series:
    """Bin a numeric series ensuring monotonic event rate across bins.

    Parameters
    ----------
    series : pd.Series
        Numeric feature to bin.
    target : pd.Series
        Binary target variable.
    n_bins : int
        Number of initial equal-frequency bins.
    min_bin_pct : float
        Minimum percentage of training records that must fall in each bin.
        Bins with fewer records are merged with adjacent bins.
    """
    edges, mapping = _fit_monotonic_binning(series, target, n_bins, min_bin_pct)
    binned = pd.cut(series, bins=edges, duplicates="drop")
    return binned.map(mapping)


def _fit_monotonic_binning(
    series: pd.Series,
    target: pd.Series,
    n_bins: int = 5,
    min_bin_pct: float = 0.05,
) -> tuple[list[float], dict]:
    """Fit monotonic binning and return (edges, mapping)."""
    binned = equal_frequency_binning(series, n_bins=n_bins)
    df = pd.DataFrame({"bin": binned, "target": target}).dropna()

    # Merge bins that don't meet the minimum record percentage
    total_records = len(df)
    min_records = int(total_records * min_bin_pct)

    bin_counts = df["bin"].value_counts()
    valid_bins = bin_counts[bin_counts >= min_records].index.tolist()

    if valid_bins:
        all_edges: set[float] = set()
        for interval in valid_bins:
            all_edges.add(interval.left)
            all_edges.add(interval.right)
        sorted_edges = sorted(all_edges)
        if sorted_edges:
            sorted_edges[0] = -np.inf
            sorted_edges[-1] = np.inf
        if len(sorted_edges) >= 2:
            edges = sorted_edges
            binned = pd.cut(series, bins=edges, duplicates="drop")
        else:
            edges = [-np.inf, np.inf]
    else:
        edges = [-np.inf, np.inf]
        binned = pd.cut(series, bins=edges, duplicates="drop")

    stats = pd.DataFrame({"bin": binned, "target": target}).dropna().groupby("bin", observed=False)["target"].mean()
    mapping = {bin_label: idx for idx, bin_label in enumerate(stats.sort_values().index)}
    return edges, mapping


def _fit_supervised_binning(
    series: pd.Series, target: pd.Series, max_bins: int = 5
) -> list[float]:
    """Fit supervised binning and return edges."""
    data = pd.DataFrame({"x": series, "y": target}).dropna()
    if data["x"].nunique() <= 2:
        bins = [-np.inf, np.inf]
        return bins

    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=max(10, int(len(data) * 0.05)),
        random_state=42,
    )
    tree.fit(data[["x"]], data["y"])
    thresholds = sorted(t for t in tree.tree_.threshold if t != -2)
    return [-np.inf, *thresholds, np.inf]


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Apply selected binning strategy to numeric columns."""

    def __init__(self, method: str = "monotonic", n_bins: int = 5, min_bin_pct: float = 0.05):
        self.method = method
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self.numeric_columns_: list[str] = []
        self._bin_edges_: dict[str, list[float]] = {}
        self._bin_mappings_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_columns_ = X.select_dtypes(include=["number"]).columns.tolist()
        self._bin_edges_ = {}
        self._bin_mappings_ = {}

        if self.method in {"supervised", "monotonic"} and y is not None:
            for col in self.numeric_columns_:
                if self.method == "monotonic":
                    edges, mapping = _fit_monotonic_binning(
                        X[col], y, self.n_bins, self.min_bin_pct
                    )
                    self._bin_edges_[col] = edges
                    self._bin_mappings_[col] = mapping
                elif self.method == "supervised":
                    edges = _fit_supervised_binning(X[col], y, self.n_bins)
                    self._bin_edges_[col] = edges
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        data = X.copy()
        for col in self.numeric_columns_:
            if self.method == "equal_width":
                data[col] = equal_width_binning(data[col], self.n_bins)
            elif self.method == "quantile":
                data[col] = equal_frequency_binning(data[col], self.n_bins)
            elif self.method == "supervised":
                if col in self._bin_edges_:
                    data[col] = pd.cut(data[col], bins=self._bin_edges_[col], duplicates="drop")
                elif y is not None:
                    data[col] = supervised_binning(data[col], y, self.n_bins)
                else:
                    raise ValueError("y is required for supervised binning (fit first or pass y)")
            elif self.method == "monotonic":
                if col in self._bin_edges_:
                    binned = pd.cut(data[col], bins=self._bin_edges_[col], duplicates="drop")
                    data[col] = binned.map(self._bin_mappings_[col])
                elif y is not None:
                    data[col] = monotonic_binning(data[col], y, self.n_bins, self.min_bin_pct)
                else:
                    raise ValueError("y is required for monotonic binning (fit first or pass y)")
            else:
                raise ValueError(
                    f"Unsupported binning method: {self.method}. Supported methods: equal_width, quantile, supervised, monotonic"
                )
        return data
