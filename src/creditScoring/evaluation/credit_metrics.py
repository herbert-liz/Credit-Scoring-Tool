"""Credit-specific model performance metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))


def gini_coefficient(y_true, y_prob) -> float:
    """Gini = 2*AUC - 1."""
    auc = roc_auc_score(y_true, y_prob)
    return float(2 * auc - 1)


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute PSI between expected and actual score distributions."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    expected_bins = pd.cut(expected, bins=quantiles, include_lowest=True)
    actual_bins = pd.cut(actual, bins=quantiles, include_lowest=True)

    exp_dist = expected_bins.value_counts(normalize=True, sort=False)
    act_dist = actual_bins.value_counts(normalize=True, sort=False).reindex(exp_dist.index).fillna(1e-6)

    exp_dist = exp_dist.clip(lower=1e-6)
    act_dist = act_dist.clip(lower=1e-6)
    psi = ((act_dist - exp_dist) * np.log(act_dist / exp_dist)).sum()
    return float(psi)


def lift_gain_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Generate lift/gain table by quantile bins."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values("y_prob", ascending=False)
    df["bucket"] = pd.qcut(df.index + 1, q=n_bins, labels=False)

    base_rate = df["y_true"].mean()
    result = df.groupby("bucket", observed=False).agg(events=("y_true", "sum"), total=("y_true", "count"))
    result["event_rate"] = result["events"] / result["total"]
    result["lift"] = result["event_rate"] / base_rate
    result["cum_gain"] = result["events"].cumsum() / result["events"].sum()
    return result.reset_index()


def divergence_index(good_scores: pd.Series, bad_scores: pd.Series) -> float:
    """Simple divergence index using means/std devs."""
    good_scores = pd.Series(good_scores)
    bad_scores = pd.Series(bad_scores)
    return float((good_scores.mean() - bad_scores.mean()) / (0.5 * (good_scores.std() + bad_scores.std())))
