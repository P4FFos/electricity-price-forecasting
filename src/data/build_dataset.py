"""Build the model table: hourly prices, weather, calendar and lag features.
Price features only use prices at least 48h old. A null price means future."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PRICES_PATH = Path("data/processed/prices.parquet")
WEATHER_PATH = Path("data/processed/weather.parquet")
OUT_PATH = Path("data/processed/dataset.parquet")

# Shortest forecast horizon. Features only use data this many hours old.
HORIZON_HOURS = 48

LAG_DAYS = [2, 3, 4, 5, 6, 7]
ROLLING_DAYS = [7, 30]

QUARTER_ERA_START = "2025-10-01"

# Not used now: TRAIN_END and VAL_END were first picked with these.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85

FORECAST_DAYS = 7

# Fixed dates, so daily rebuilds don't move rows between splits.
# New rows only go into test.
TRAIN_END = pd.Timestamp("2025-07-18 12:00", tz="Europe/Stockholm")
VAL_END = pd.Timestamp("2026-02-15 16:00", tz="Europe/Stockholm")


def load_hourly_prices():
    """Average each zone's prices to one value per hour."""
    prices = pd.read_parquet(PRICES_PATH)

    return (
        prices.set_index("time_start")
        .groupby("zone")["eur_per_kwh"]
        .resample("h")
        .mean()
        .reset_index()
        .rename(columns={"time_start": "timestamp", "eur_per_kwh": "price"})
    )


def join_weather(prices_h):
    """Add temperature and wind speed per zone and hour."""
    weather = pd.read_parquet(WEATHER_PATH)

    # Left join, so hours without weather are kept and the hourly grid stays whole.
    # LightGBM can handle the missing weather.
    return prices_h.merge(
        weather[["zone", "timestamp", "temperature", "wind_speed"]],
        on=["zone", "timestamp"],
        how="left",
    )


def add_calendar_features(df):
    """Add hour, weekday, month, weekend and 15-minute era columns."""
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["dayofweek"] >= 5
    df["is_quarter_era"] = df["timestamp"] >= QUARTER_ERA_START
    return df


def add_price_history_features(df):
    """Add past prices and rolling means, all at least 48 hours old.
    Shifts count rows, so every hour must have a row."""
    for days in LAG_DAYS:
        df[f"price_lag_{days}d"] = df.groupby("zone")["price"].shift(24 * days)

    for days in ROLLING_DAYS:
        # Shift before rolling, or the mean would include the target price.
        # Done per zone, so one zone's prices don't leak into the next.
        df[f"price_mean_{days}d"] = df.groupby("zone")["price"].transform(
            lambda s, n=24 * days: (
                s.shift(HORIZON_HOURS).rolling(n, min_periods=24).mean()
            )
        )

    return df


def check_hourly_grid(df):
    """Raise if rows in a zone are not one hour apart."""
    gaps = df.groupby("zone")["timestamp"].diff().dropna()
    bad = gaps[gaps != pd.Timedelta("1h")]

    if not bad.empty:
        raise ValueError(
            f"Hourly grid has {len(bad)} gaps; lag features would be wrong.\n"
            f"{bad.value_counts()}"
        )


def report(df):
    """Print rows, date range, nulls and missing weather."""
    print(f"\nRows: {len(df):,}")
    print(f"Range: {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print(f"Columns: {len(df.columns)}")

    print("\nNulls:")
    print(df.isna().sum())

    print("\nMissing weather per zone:")
    print(df[df["temperature"].isna()].groupby("zone").size())


def split_by_time(df):
    """Split rows into train, val and test at TRAIN_END and VAL_END."""
    train = df[df["timestamp"] <= TRAIN_END]
    val = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
    test = df[df["timestamp"] > VAL_END]

    return train, val, test


def add_future_rows(df):
    """Add FORECAST_DAYS of empty future hours per zone.
    daily_job.py forecasts these rows."""
    last = df["timestamp"].max()

    future = pd.DataFrame(
        [
            {"zone": zone, "timestamp": ts, "price": np.nan}
            for zone in df["zone"].unique()
            for ts in pd.date_range(
                last + pd.Timedelta(hours=1),
                last + pd.Timedelta(days=FORECAST_DAYS),
                freq="h",
                tz="Europe/Stockholm",
            )
        ]
    )

    return pd.concat([df, future], ignore_index=True)


def check_no_past_gaps(df):
    """Raise if any past hour has no price.
    A null price must only mean future."""
    last_known = df.loc[df["price"].notna(), "timestamp"].max()
    holes = df[(df["timestamp"] < last_known) & (df["price"].isna())]

    if not holes.empty:
        raise ValueError(
            f"{len(holes)} hours before {last_known} have no price. "
            f"First: {holes['timestamp'].min()}. Fetch a wider window."
        )


def main():
    """Build dataset.parquet and the train, val and test files."""
    df = load_hourly_prices()
    df = join_weather(df)

    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)
    check_hourly_grid(df)

    check_no_past_gaps(df)
    # Add future rows first, so they get lag features too.
    df = add_future_rows(df)
    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_price_history_features(df)

    report(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWritten to {OUT_PATH}")

    future = df[df["price"].isna()]
    print(f"\nfuture rows: {len(future)}")
    print(
        future[
            ["zone", "timestamp", "price_lag_2d", "price_lag_7d", "price_mean_7d"]
        ].head()
    )

    known = df[df["price"].notna()]
    train, val, test = split_by_time(known)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(
            f"{name:6} {len(part):>7,} rows  "
            f"{part['timestamp'].min()} -> {part['timestamp'].max()}"
        )
        part.to_parquet(OUT_PATH.parent / f"{name}.parquet", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
