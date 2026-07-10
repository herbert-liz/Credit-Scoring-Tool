"""WOE and IV utilities."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def calculate_woe_iv(feature: pd.Series, target: pd.Series, event: int = 1, eps: float = 1e-6):
    """Calculate WOE table and IV for a binned/categorical feature."""
    df = pd.DataFrame({"feature": feature, "target": target}).dropna()
    grouped = df.groupby("feature", observed=False)["target"].agg(["count", "sum"])
    grouped = grouped.rename(columns={"sum": "event_count"})
    grouped["non_event_count"] = grouped["count"] - grouped["event_count"]

    total_events = (df["target"] == event).sum()
    total_non_events = len(df) - total_events

    grouped["event_rate"] = (grouped["event_count"] + eps) / (total_events + eps)
    grouped["non_event_rate"] = (grouped["non_event_count"] + eps) / (total_non_events + eps)
    grouped["woe"] = np.log(grouped["event_rate"] / grouped["non_event_rate"])
    grouped["iv_component"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped["woe"]
    iv = grouped["iv_component"].sum()

    return grouped.reset_index(), float(iv)


def calculate_iv_for_dataframe(df: pd.DataFrame, target: pd.Series, columns: list[str] | None = None) -> pd.DataFrame:
    """Calculate IV ranking for multiple columns."""
    cols = columns or df.columns.tolist()
    rows = []
    for col in cols:
        _, iv = calculate_woe_iv(df[col], target)
        rows.append({"feature": col, "iv": iv})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


class WOETransformer(BaseEstimator, TransformerMixin):
    """Fit WOE mappings and transform features."""

    def __init__(self):
        self.woe_mappings_: dict[str, dict] = {}
        self.iv_table_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        iv_rows = []
        for col in X.columns:
            table, iv = calculate_woe_iv(X[col], y)
            mapping = dict(zip(table["feature"], table["woe"]))
            self.woe_mappings_[col] = mapping
            iv_rows.append({"feature": col, "iv": iv})
        self.iv_table_ = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
        return self

    def transform(self, X: pd.DataFrame):
        data = X.copy()
        for col, mapping in self.woe_mappings_.items():
            data[col] = data[col].map(mapping).fillna(0.0)
        return data
