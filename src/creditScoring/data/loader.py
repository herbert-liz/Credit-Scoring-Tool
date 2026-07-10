"""Module for loading and validating data for credit scoring."""

import pandas as pd


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path, **kwargs)


def ensure_dataframe(data) -> None:
    """Ensure input is a pandas DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")


def check_empty(df: pd.DataFrame) -> None:
    """Check if DataFrame is empty."""
    if df.shape[0] == 0:
        raise ValueError("DataFrame is empty")


def check_duplicate_columns(df: pd.DataFrame) -> None:
    """Check for duplicated column names."""
    if df.columns.duplicated().any():
        raise ValueError("Duplicated column names detected")


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run basic validations on DataFrame and return validated copy."""
    ensure_dataframe(df)
    check_empty(df)
    check_duplicate_columns(df)
    return df
