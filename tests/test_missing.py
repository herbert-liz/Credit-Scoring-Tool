import pandas as pd

from creditScoring.preprocessing.missing import handle_missing_values


def test_handle_missing_values_numeric_and_categorical():
    df = pd.DataFrame({
        "num": [1.0, None, 3.0],
        "cat": ["A", None, "B"],
    })

    out = handle_missing_values(df, numeric_strategy="replace", numeric_fill_value=0, categorical_strategy="constant")

    assert out["num"].isna().sum() == 0
    assert out["cat"].isna().sum() == 0
    assert out.loc[1, "cat"] == "Unknown"
