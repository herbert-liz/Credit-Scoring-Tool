"""Feature selection exports."""

from .correlation import correlation_matrix, remove_correlated_features
from .iv_selection import IVFeatureSelector

__all__ = ["IVFeatureSelector", "correlation_matrix", "remove_correlated_features"]
