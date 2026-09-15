"""
Consolidate raw elprisetjustnu.se JSON into one tidy parquet table.

Input:  data/raw/prices/{ZONE}/{YYYY-MM-DD}.json
Output: data/processed/prices.parquet
"""

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/prices")
OUT_PATH = Path("data/processed/prices.parquet")


def load_all():
    """Read every raw JSON file into a single DataFrame"""
    records = []

    for zone_dir in sorted(RAW_DIR.iterdir()):
        if not zone_dir.is_dir():
            continue

        files = sorted(zone_dir.glob("*.json"))
        print(f"{zone_dir.name}: {len(files)} files")

        for path in files:
            day_records = json.loads(path.read_text(encoding="utf-8"))
            for record in day_records:
                record["zone"] = zone_dir.name
                record["intervals_in_day"] = len(day_records)
            records.extend(day_records)
    return pd.DataFrame.from_records(records)


def clean(df):
    """Rename columns, parse timestamps, tag pricing regime."""
    df = df.rename(
        columns={
            "SEK_per_kWh": "sek_per_kwh",
            "EUR_per_kWh": "eur_per_kwh",
            "EXR": "exr",
        },
        errors="raise",
    )

    for col in ("time_start", "time_end"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert("Europe/Stockholm")

    df["resolution"] = df["intervals_in_day"].map(
        lambda n: "quarter" if n > 48 else "hourly"
    )
    df = df.drop(columns=["intervals_in_day"])
    return df.sort_values(["zone", "time_start"]).reset_index(drop=True)


def report(df):
    """Print the coverage report."""

    print(f"\nRows: {len(df):,}")
    print(f"Range: {df['time_start'].min()} to {df['time_start'].max()}")

    duplicates = df.duplicated(subset=["zone", "time_start"]).sum()
    if duplicates:
        print(f"WARNING: {duplicates} duplicate (zone, time_start) rows")
    else:
        print("Duplicates: none")

    negatives = (df["eur_per_kwh"] < 0).sum()
    print(f"Negative prices: {negatives:,} rows")

    nulls = df[["eur_per_kwh", "sek_per_kwh"]].isna().sum().to_dict()
    print(f"Nulls: {nulls}")

    print("\nRows per zone and resolution:")
    print(df.groupby(["zone", "resolution"]).size().unstack(fill_value=0))

    print("\nDistinct days per zone:")
    print(df.assign(day=df["time_start"].dt.date).groupby("zone")["day"].nunique())


def main():
    df = load_all()
    df = clean(df)
    report(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
