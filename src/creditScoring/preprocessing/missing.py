import pandas as pd

def handle_missing_values(df, strategy='mean', columns=None, replace_value=0):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential missing values.
    strategy : str, default='mean'
        Strategy to handle missing values. Options: 'mean', 'median', 'mode', 'drop', 'replace'.
    columns : list, optional
        List of column names to apply the strategy. If None, applies to all columns.
    replace_value : scalar, default=0
        Value to replace missing values with when strategy is 'replace'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
    >>> handle_missing_values(df, strategy='mean')
    """
    
    df_copy = df.copy()
    cols_to_process = columns if columns else df_copy.columns
    
    if strategy == 'mean':
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(df_copy[cols_to_process].mean())
    elif strategy == 'median':
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(df_copy[cols_to_process].median())
    elif strategy == 'mode':
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(df_copy[cols_to_process].mode().iloc[0])
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=cols_to_process)
    elif strategy == 'replace':
        df_copy[cols_to_process] = df_copy[cols_to_process].fillna(replace_value)
    
    return df_copy