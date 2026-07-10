"""Scorecard point assignment helpers."""

import pandas as pd
import numpy as np

from .scaling import ScoreScaler


def build_scorecard_points(coefficients: pd.Series, woe_tables: dict[str, pd.DataFrame], scaler: ScoreScaler) -> pd.DataFrame:
    """Create variable/bin point table from coefficients and WOE values."""
    rows = []
    n_features = max(len(coefficients), 1)

    for feature, coef in coefficients.items():
        table = woe_tables.get(feature)
        if table is None or "woe" not in table.columns:
            continue
        for _, row in table.iterrows():
            partial_score = -(coef * row["woe"] * scaler.factor) / n_features
            rows.append({"feature": feature, "bin": row["feature"], "woe": row["woe"], "points": partial_score})

    return pd.DataFrame(rows)


def score_from_points(transformed_woe: pd.DataFrame, coefficients: pd.Series, scaler: ScoreScaler) -> pd.Series:
    """Compute final score by linear model and convert to score scale."""
    logit = transformed_woe[coefficients.index].mul(coefficients, axis=1).sum(axis=1)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob.apply(scaler.probability_to_score)
