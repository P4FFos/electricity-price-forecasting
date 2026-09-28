"""Combine the raw price JSON files into data/processed/prices.parquet.
Keeps the source resolution; build_dataset.py turns it into hourly."""

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/prices")
OUT_PATH = Path("data/processed/prices.parquet")
INTERVALS_COL = "intervals_in_day"


def load_all():
    """Read all raw JSON files into one DataFrame."""
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
                record[INTERVALS_COL] = len(day_records)
            records.extend(day_records)
    return pd.DataFrame.from_records(records)


def clean(df):
    """Rename columns, parse times and mark hourly or 15-minute days."""
    df = df.rename(
        columns={
            "SEK_per_kWh": "sek_per_kwh",
            "EUR_per_kWh": "eur_per_kwh",
            "EXR": "exr",
        },
        errors="raise",
    )

    # The offset changes with summer time, so parse as UTC first.
    for col in ("time_start", "time_end"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert("Europe/Stockholm")

    # Hourly days have 23-25 prices, 15-minute days have 92-100.
    df["resolution"] = df[INTERVALS_COL].map(
        lambda n: "quarter" if n > 48 else "hourly"
    )
    df = df.drop(columns=[INTERVALS_COL])
    return df.sort_values(["zone", "time_start"]).reset_index(drop=True)


def report(df):
    """Print a short summary of the data."""

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
    """Load, clean and save prices.parquet."""
    df = load_all()
    df = clean(df)
    report(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
