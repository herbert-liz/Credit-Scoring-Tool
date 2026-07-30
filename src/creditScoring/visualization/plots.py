"""General plotting helpers for exploratory credit scoring analysis."""

import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization extras") from exc
    return plt


def plot_distributions(df: pd.DataFrame, columns: list[str]):
    plt = _require_matplotlib()
    axes = df[columns].hist(figsize=(4 * len(columns), 3), bins=20)
    return plt.gcf(), axes


def plot_correlation_heatmap(df: pd.DataFrame):
    plt = _require_matplotlib()
    corr = df.select_dtypes(include=["number"]).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(cax)
    return fig, ax


def plot_score_distribution(scores: pd.Series, y: pd.Series | None = None):
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4))
    if y is None:
        ax.hist(scores, bins=30)
    else:
        for cls in sorted(set(y)):
            ax.hist(scores[pd.Series(y) == cls], bins=30, alpha=0.5, label=f"class_{cls}")
        ax.legend()
    ax.set_title("Score Distribution")
    return fig, ax
