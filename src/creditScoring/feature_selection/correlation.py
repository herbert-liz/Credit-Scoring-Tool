"""Correlation-based feature filtering."""

import pandas as pd


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Return absolute correlation matrix for numeric features."""
    return df.select_dtypes(include=["number"]).corr(method=method).abs()


def remove_correlated_features(
    df: pd.DataFrame,
    iv_ranking: pd.DataFrame,
    threshold: float = 0.8,
    method: str = "pearson",
) -> list[str]:
    """Keep less-correlated variables using IV ranking priority."""
    corr = correlation_matrix(df, method=method)
    ranked = iv_ranking.sort_values("iv", ascending=False)["feature"].tolist()
    selected: list[str] = []

    for feature in ranked:
        if feature not in corr.columns:
            continue
        if all(corr.loc[feature, kept] < threshold for kept in selected):
            selected.append(feature)
    return selected
