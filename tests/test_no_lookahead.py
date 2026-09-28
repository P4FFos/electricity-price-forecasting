"""Check that no feature at hour t uses a price newer than t-48h."""

import pandas as pd
import pytest

HORIZON_HOURS = 48


@pytest.fixture(scope="module")
def df():
    """The built dataset, read once."""
    return pd.read_parquet("./data/processed/dataset.parquet")


def test_lags_match_actual_prices(df):
    """price_lag_Nd must equal the real price N days before the target."""
    se3 = df[df["zone"] == "SE3"].set_index("timestamp")

    # Look up by time, not by row shift, so the test doesn't
    # assume rows are one hour apart.
    for days in [2, 7]:
        shifted = se3["price"].reindex(se3.index - pd.Timedelta(days=days))
        expected = pd.Series(shifted.values, index=se3.index)
        actual = se3[f"price_lag_{days}d"]

        both = expected.notna() & actual.notna()
        assert (expected[both] - actual[both]).abs().max() < 1e-9


def test_rolling_window_ends_before_issue_time(df):
    """price_mean_7d must only use prices at least 48h old."""
    se3 = df[df["zone"] == "SE3"].reset_index(drop=True)
    row = se3.iloc[5000]
    cutoff = row["timestamp"] - pd.Timedelta(hours=HORIZON_HOURS)

    window = se3[
        (se3["timestamp"] <= cutoff)
        & (se3["timestamp"] > cutoff - pd.Timedelta(hours=24 * 7))
    ]
    assert abs(window["price"].mean() - row["price_mean_7d"]) < 1e-9


@pytest.mark.parametrize("zone", ["SE1", "SE2", "SE3", "SE4"])
@pytest.mark.parametrize("col", ["price_mean_7d", "price_mean_30d"])
def test_rolling_means_stay_within_zone(df, zone, col):
    """A zone's first rolling values must be empty, not taken from another zone."""
    rows = df[df["zone"] == zone].sort_values("timestamp").reset_index(drop=True)

    # The first 48 rows have no 48h-old price, so a value here
    # must come from another zone.
    borrowed = rows.loc[: HORIZON_HOURS - 1, col].notna().sum()
    assert borrowed == 0, f"{zone} {col}: {borrowed} values from another zone"

    # The first value needs 24 prices, all from this zone.
    first = rows[col].first_valid_index()
    assert first == HORIZON_HOURS + 23
    assert abs(rows.loc[first, col] - rows["price"].iloc[:24].mean()) < 1e-9
