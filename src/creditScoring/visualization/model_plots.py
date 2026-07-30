"""Model-performance plotting helpers."""

import pandas as pd
from sklearn.metrics import roc_curve

from creditScoring.evaluation.credit_metrics import ks_statistic, lift_gain_table


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization extras") from exc
    return plt


def plot_roc_curve(y_true, y_prob):
    plt = _require_matplotlib()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve")
    ax.legend()
    return fig, ax


def plot_ks_curve(y_true, y_prob):
    plt = _require_matplotlib()
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df["cum_bad"], label="Cum Bad")
    ax.plot(df["cum_good"], label="Cum Good")
    ax.set_title(f"KS Curve (KS={ks_statistic(y_true, y_prob):.3f})")
    ax.legend()
    return fig, ax


def plot_feature_importance(importances: pd.Series):
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4))
    importances.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title("Feature Importance")
    return fig, ax


def plot_lift_gain(y_true, y_prob):
    plt = _require_matplotlib()
    table = lift_gain_table(y_true, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(table["bucket"], table["lift"])
    axes[0].set_title("Lift")
    axes[0].set_xlabel("Bucket")
    axes[0].set_ylabel("Lift")
    axes[1].plot(table["bucket"], table["cum_gain"])
    axes[1].set_title("Cumulative Gain")
    axes[1].set_xlabel("Bucket")
    axes[1].set_ylabel("Cumulative Gain")
    return fig, axes
