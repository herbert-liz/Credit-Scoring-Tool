"""Result container utilities for model evaluation."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ModelResults:
    """Container for model metrics and optional artifacts."""

    classification_metrics: dict = field(default_factory=dict)
    credit_metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "classification_metrics": self.classification_metrics,
            "credit_metrics": self.credit_metrics,
            "metadata": self.metadata,
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for metric_group, values in {
            "classification": self.classification_metrics,
            "credit": self.credit_metrics,
        }.items():
            for metric, value in values.items():
                if isinstance(value, (int, float)):
                    rows.append({"group": metric_group, "metric": metric, "value": value})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        df = self.to_dataframe()
        if df.empty:
            return "No numeric metrics available"
        return "\n".join(f"[{row.group}] {row.metric}: {row.value:.4f}" for row in df.itertuples())
