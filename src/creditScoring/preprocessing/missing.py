"""Missing value handling utilities."""

import pandas as pd


def handle_numeric_missing_values(df: pd.DataFrame, strategy="replace", columns=None, value=-1):
    """Handle missing values in numeric columns using selected strategy."""
    df_copy = df.copy()
    if columns is None:
        cols_to_process = df_copy.select_dtypes(include=["number"]).columns
    else:
        cols_to_process = columns

    if strategy == "mean":
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(df_copy[cols_to_process].mean(skipna=True))
    elif strategy == "median":
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(df_copy[cols_to_process].median(skipna=True))
    elif strategy == "replace":
        if value is None:
            raise ValueError("Value must be provided for 'replace' strategy")
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(value)
    elif strategy == "drop":
        df_copy = df_copy.dropna(subset=cols_to_process)
    else:
        raise ValueError(
            f"Unsupported numeric strategy: {strategy}. Supported strategies: mean, median, replace, drop"
        )
    return df_copy


def handle_categorical_missing_values(
    df: pd.DataFrame,
    strategy: str = "mode",
    columns=None,
    fill_value: str = "Unknown",
) -> pd.DataFrame:
    """Handle missing values in categorical columns."""
    df_copy = df.copy()
    if columns is None:
        cols_to_process = df_copy.select_dtypes(exclude=["number"]).columns
    else:
        cols_to_process = columns

    if strategy == "mode":
        for col in cols_to_process:
            mode = df_copy[col].mode(dropna=True)
            df_copy[col] = df_copy[col].fillna(mode.iloc[0] if not mode.empty else fill_value)
    elif strategy == "constant":
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(fill_value)
    elif strategy == "drop":
        df_copy = df_copy.dropna(subset=cols_to_process)
    else:
        raise ValueError(
            f"Unsupported categorical strategy: {strategy}. Supported strategies: mode, constant, drop"
        )

    return df_copy


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    numeric_fill_value=-1,
    categorical_fill_value: str = "Unknown",
) -> pd.DataFrame:
    """Handle missing values for both numeric and categorical columns."""
    result = handle_numeric_missing_values(
        df,
        strategy=numeric_strategy,
        value=numeric_fill_value,
    )
    result = handle_categorical_missing_values(
        result,
        strategy=categorical_strategy,
        fill_value=categorical_fill_value,
    )
    return result
