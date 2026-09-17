"""
Consolidate raw SMHI weather CSV files into one tidy parquet table.

Input:  data/raw/weather/{ZONE}_{temperature,wind_speed}.csv
Output: data/processed/weather.parquet

Data provided by SMHI (https://www.smhi.se).
"""

import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/weather")
OUT_PATH = Path("data/processed/weather.parquet")
ZONES = ["SE1", "SE2", "SE3", "SE4"]

PERIODS = ["corrected-archive", "latest-months"]

START = "2022-11-01"


def find_header_row(path):
    """Find the line number where the real data starts"""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith("Datum"):
                return i
    raise ValueError(f"No 'Datum' header found in {path}")


def read_one(path, value_column):
    """Read one SMHI CSV into a tidy frame with timestamp, value, quality"""
    skip = find_header_row(path)
    df = pd.read_csv(
        path,
        sep=";",
        skiprows=skip,
        usecols=[0, 1, 2, 3],
        names=["date", "time", value_column, "quality"],
        header=0,
    )

    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"], utc=True
    ).dt.tz_convert("Europe/Stockholm")
    df = df.drop(columns=["date", "time"])

    return df[df["timestamp"] >= START]


def read_parameter(zone, name, value_column):
    """Read both periods for one parameter and combine them"""
    frames = [
        read_one(RAW_DIR / f"{zone}_{name}_{period}.csv", value_column)
        for period in PERIODS
    ]
    df = pd.concat(frames, ignore_index=True)

    # The periods overlap by a few weeks. keep="first" keeps the
    # corrected-archive value, which is quality-checked
    df = df.drop_duplicates(subset="timestamp", keep="first")

    return df.sort_values("timestamp").reset_index(drop=True)


def load_zone(zone):
    """Load and merge temperature and wind for one zone."""
    temp = read_parameter(zone, "temperature", "temperature")
    wind = read_parameter(zone, "wind_speed", "wind_speed")

    merged = temp.merge(wind, on="timestamp", how="outer", suffixes=("_temp", "_wind"))
    merged["zone"] = zone
    return merged


def report(df):
    """Print the coverage report"""
    print(f"\nRows: {len(df):,}")

    print("\nRows per zone:")
    print(df.groupby("zone").size())

    print("\nDate range per zone:")
    print(df.groupby("zone")["timestamp"].agg(["min", "max"]))

    print("\nNulls:")
    print(df.isna().sum())

    print("\nQuality flags:")
    print(df["quality_temp"].value_counts())
    print(df["quality_wind"].value_counts())

    print("\nTemperature and wind:")
    print(df.groupby("zone")[["temperature", "wind_speed"]].describe())


def main():
    frames = [load_zone(zone) for zone in ZONES]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)

    report(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWritten to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
