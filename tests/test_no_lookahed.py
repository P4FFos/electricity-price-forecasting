import pandas as pd
import pytest

HORIZON_HOURS = 48


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet("./data/processed/dataset.parquet")


def test_lags_match_actual_prices(df):
    """price_lag_Nd must equal the real price N days before the target."""
    se3 = df[df["zone"] == "SE3"].set_index("timestamp")

    for days in [2, 7]:
        shifted = se3["price"].reindex(se3.index - pd.Timedelta(days=days))
        expected = pd.Series(shifted.values, index=se3.index)
        actual = se3[f"price_lag_{days}d"]

        both = expected.notna() & actual.notna()
        assert (expected[both] - actual[both]).abs().max() < 1e-9


def test_rolling_window_ends_before_issue_time(df):
    """price_mean_7d must not include any data after t0."""
    se3 = df[df["zone"] == "SE3"].reset_index(drop=True)
    row = se3.iloc[5000]
    cutoff = row["timestamp"] - pd.Timedelta(hours=HORIZON_HOURS)

    window = se3[
        (se3["timestamp"] <= cutoff)
        & (se3["timestamp"] > cutoff - pd.Timedelta(hours=24 * 7))
    ]
    assert abs(window["price"].mean() - row["price_mean_7d"]) < 1e-9
