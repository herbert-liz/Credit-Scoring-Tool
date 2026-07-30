"""General helper utilities used across package modules."""

import logging
from typing import Iterable

import pandas as pd


class CreditScoringError(Exception):
    """Base exception for package-specific errors."""


class ValidationError(CreditScoringError):
    """Raised for invalid user input or unsupported data."""


def get_logger(name: str = "creditScoring", level: int = logging.INFO) -> logging.Logger:
    """Create or return configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dataframe(data: object, name: str = "data") -> pd.DataFrame:
    """Validate that input is a pandas DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise ValidationError(f"{name} must be a pandas DataFrame")
    return data


def ensure_columns_exist(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Validate required columns exist in a DataFrame."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing columns: {missing}")


def validate_binary_target(y: pd.Series) -> pd.Series:
    """Validate binary target series with exactly two classes."""
    unique_values = set(pd.Series(y).dropna().unique())
    if len(unique_values) != 2:
        raise ValidationError("Target must contain exactly two classes")
    return pd.Series(y)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers safely."""
    if denominator == 0:
        return default
    return numerator / denominator
