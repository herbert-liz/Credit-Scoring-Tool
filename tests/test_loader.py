import pandas as pd
import pytest

from creditScoring.data.loader import validate_dataframe


def test_validate_dataframe_returns_df():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = validate_dataframe(df)
    assert out.equals(df)


def test_validate_dataframe_empty_raises():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame())
