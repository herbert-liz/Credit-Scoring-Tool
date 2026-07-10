"""Domain constants for credit scoring."""

SCORE_MIN = 300
SCORE_MAX = 850
DEFAULT_BAD_LABEL = 1
DEFAULT_GOOD_LABEL = 0

RISK_CATEGORIES = {
    "very_low": (750, SCORE_MAX),
    "low": (700, 749),
    "medium": (650, 699),
    "high": (600, 649),
    "very_high": (SCORE_MIN, 599),
}

CLASSIFICATION_METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

CREDIT_METRIC_NAMES = ["ks", "gini", "psi", "divergence_index"]
