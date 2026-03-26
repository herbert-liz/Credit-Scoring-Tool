import pandas as pd

def handle_numeric_missing_values(df: pd.DataFrame, strategy='replace', columns=None, value=-1):
    """
    Handle missing values in numeric columns of a DataFrame using specified strategy.
    
    Parameters
    df : pd.DataFrame
        Input DataFrame with potential missing values.
    strategy : str, default='replace'
        Strategy to handle missing values. Options: 
            'mean': fill missing values with the mean (excluding missing values) of the column
            'median': fill missing values with the median (excluding missing values) of the column
            'replace': fill missing values with a specified numeric value (requires additional parameter)
            'drop': drop rows with missing values
    columns : list, optional
        List of column names to apply the strategy. If None, applies to all numeric columns.
    value : numeric, optional
        Value to use for 'replace' strategy. Required if strategy is 'replace'. Default is -1.
    
    Returns
    pd.DataFrame
        DataFrame with missing values handled.
    
    Examples
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
    >>> handle_numeric_missing_values(df)
    """
    
    df_copy = df.copy()
    if columns is None:
        cols_to_process = df_copy.select_dtypes(include=['number']).columns
    else:
        cols_to_process = columns
    
    if strategy == 'mean':
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(df_copy[cols_to_process].mean(skipna=True))
    elif strategy == 'median':
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(df_copy[cols_to_process].median(skipna=True))
    elif strategy == 'replace':
        if value is None:
            raise ValueError("Value must be provided for 'replace' strategy")
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(value)
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=cols_to_process)
    
    return df_copy