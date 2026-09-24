import sys
from pathlib import Path

import pandas as pd
import numpy as np

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

FORECAST_DAYS = 7

TRAIN_END = pd.Timestamp("2025-07-18 12:00", tz="Europe/Stockholm")
VAL_END = pd.Timestamp("2026-02-15 16:00", tz="Europe/Stockholm")


def load_hourly_prices():
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
    weather = pd.read_parquet(WEATHER_PATH)

    return prices_h.merge(
        weather[["zone", "timestamp", "temperature", "wind_speed"]],
        on=["zone", "timestamp"],
        how="left",
    )


def add_calendar_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["dayofweek"] >= 5
    df["is_quarter_era"] = df["timestamp"] >= QUARTER_ERA_START
    return df


def add_price_history_features(df):
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
    gaps = df.groupby("zone")["timestamp"].diff().dropna()
    bad = gaps[gaps != pd.Timedelta("1h")]

    if not bad.empty:
        raise ValueError(
            f"Hourly grid has {len(bad)} gaps; lag features would be wrong.\n"
            f"{bad.value_counts()}"
        )


def report(df):
    print(f"\nRows: {len(df):,}")
    print(f"Range: {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print(f"Columns: {len(df.columns)}")

    print("\nNulls:")
    print(df.isna().sum())

    print("\nMissing weather per zone:")
    print(df[df["temperature"].isna()].groupby("zone").size())


def split_by_time(df):
    train = df[df["timestamp"] <= TRAIN_END]
    val = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
    test = df[df["timestamp"] > VAL_END]

    return train, val, test


def add_future_rows(df):
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
    last_known = df.loc[df["price"].notna(), "timestamp"].max()
    holes = df[(df["timestamp"] < last_known) & (df["price"].isna())]

    if not holes.empty:
        raise ValueError(
            f"{len(holes)} hours before {last_known} have no price. "
            f"First: {holes['timestamp'].min()}. Fetch a wider window."
        )


def main():
    df = load_hourly_prices()
    df = join_weather(df)

    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)
    check_hourly_grid(df)

    check_no_past_gaps(df)
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
