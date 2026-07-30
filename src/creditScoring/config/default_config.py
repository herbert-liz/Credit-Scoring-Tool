"""Default configuration for the credit scoring pipeline."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BinningConfig:
    n_bins: int = 5
    method: str = "quantile"  # quantile|equal_width|supervised|monotonic


@dataclass
class MissingConfig:
    numeric_strategy: str = "median"
    categorical_strategy: str = "mode"
    fill_value: Any = -1
    categorical_fill_value: str = "Unknown"


@dataclass
class FeatureSelectionConfig:
    min_iv: float = 0.02
    max_correlation: float = 0.8
    correlation_method: str = "pearson"


@dataclass
class EvaluationConfig:
    threshold: float = 0.5


@dataclass
class PipelineConfig:
    binning: BinningConfig = field(default_factory=BinningConfig)
    missing: MissingConfig = field(default_factory=MissingConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model_type: str = "logistic"
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_default_config() -> dict[str, Any]:
    """Return default pipeline configuration as a nested dictionary."""
    return PipelineConfig().to_dict()
