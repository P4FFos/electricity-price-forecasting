"""
Build the modelling dataset by joining prices to weather and adding features

Input:  data/processed/prices.parquet
        data/processed/weather.parquet
Output: data/processed/dataset.parquet
"""

import sys
from pathlib import Path

import pandas as pd

PRICES_PATH = Path("data/processed/prices.parquet")
WEATHER_PATH = Path("data/processed/weather.parquet")
OUT_PATH = Path("data/processed/dataset.parquet")

# The shortest forecast horizon, features may only use data from at least this many hours before the target
HORIZON_HOURS = 48

LAG_DAYS = [2, 3, 4, 5, 6, 7]
ROLLING_DAYS = [7, 30]

QUARTER_ERA_START = "2025-10-01"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.85


def load_hourly_prices():
    """Load prices and average the quarter-hourly ones into hourly values"""
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
    """Attach temperature and wind to each price hour A left join so every price hour survives
    Missing weather becomes null, which gradient boosting handles natively
    """
    weather = pd.read_parquet(WEATHER_PATH)

    return prices_h.merge(
        weather[["zone", "timestamp", "temperature", "wind_speed"]],
        on=["zone", "timestamp"],
        how="left",
    )


def add_calendar_features(df):
    """Calendar features are safe: a calendar is known in advance"""
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["dayofweek"] >= 5
    df["is_quarter_era"] = df["timestamp"] >= QUARTER_ERA_START
    return df


def add_price_history_features(df):
    """Past prices, all stepped back to the forecast issue time"""
    for days in LAG_DAYS:
        df[f"price_lag_{days}d"] = df.groupby("zone")["price"].shift(24 * days)

    for days in ROLLING_DAYS:
        # The shift MUST come before the rolling window
        df[f"price_mean_{days}d"] = (
            df.groupby("zone")["price"]
            .shift(HORIZON_HOURS)
            .rolling(24 * days, min_periods=24)
            .mean()
        )

    return df


def check_hourly_grid(df):
    """Fail if the hourly grid has gaps"""
    gaps = df.groupby("zone")["timestamp"].diff().dropna()
    bad = gaps[gaps != pd.Timedelta("1h")]

    if not bad.empty:
        raise ValueError(
            f"Hourly grid has {len(bad)} gaps; lag features would be wrong.\n"
            f"{bad.value_counts()}"
        )


def report(df):
    """Print the coverage report"""
    print(f"\nRows: {len(df):,}")
    print(f"Range: {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print(f"Columns: {len(df.columns)}")

    print("\nNulls:")
    print(df.isna().sum())

    print("\nMissing weather per zone:")
    print(df[df["temperature"].isna()].groupby("zone").size())


def split_by_time(df):
    """ "Split chronologically"""
    cut1 = df["timestamp"].quantile(TRAIN_FRAC)
    cut2 = df["timestamp"].quantile(VAL_FRAC)

    train = df[df["timestamp"] <= cut1]
    val = df[(df["timestamp"] > cut1) & (df["timestamp"] <= cut2)]
    test = df[df["timestamp"] > cut2]

    return train, val, test


def main():
    df = load_hourly_prices()
    df = join_weather(df)

    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)
    check_hourly_grid(df)

    df = add_calendar_features(df)
    df = add_price_history_features(df)

    report(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWritten to {OUT_PATH}")

    train, val, test = split_by_time(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(
            f"{name:6} {len(part):>7,} rows  "
            f"{part['timestamp'].min()} -> {part['timestamp'].max()}"
        )
        part.to_parquet(OUT_PATH.parent / f"{name}.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
