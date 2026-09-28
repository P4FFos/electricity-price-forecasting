"""Simple baselines for the 2-7 day forecast, scored on val.
Writes results/baselines.csv."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("./data/processed")
RESULTS_PATH = Path("results/baselines.csv")

HORIZON_HOURS = 48


def seasonal_naive(df):
    """Price at the same hour, 7 days earlier."""
    return df["price_lag_7d"]


def persistence(df):
    """Last known price at the same hour (2 days earlier)."""
    return df["price_lag_2d"]


def climatology(df, train):
    """Mean train price for the same zone, hour and month."""
    lookup = train.groupby(["zone", "hour", "month"])["price"].mean()
    index = pd.MultiIndex.from_arrays([df["zone"], df["hour"], df["month"]])
    return pd.Series(lookup.reindex(index).values, index=df.index)


def score(actual, predicted):
    """MAE and RMSE over rows where both values exist."""
    mask = actual.notna() & predicted.notna()
    a, p = actual[mask], predicted[mask]

    return {
        "n": len(a),
        "mae": np.abs(a - p).mean(),
        "rmse": np.sqrt(((a - p) ** 2).mean()),
    }


def main():
    """Score each baseline on val, in total and per zone."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")

    predictions = {
        "seasonal_naive": seasonal_naive(val),
        "persistence": persistence(val),
        "climatology": climatology(val, train),
    }

    rows = []
    for name, pred in predictions.items():
        overall = score(val["price"], pred)
        rows.append({"baseline": name, "zone": "all", **overall})

        for zone in ["SE1", "SE2", "SE3", "SE4"]:
            m = val["zone"] == zone
            rows.append(
                {"baseline": name, "zone": zone, **score(val["price"][m], pred[m])}
            )

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
