"""
Check whether the 80% prediction intervals are honest, and calibrate them
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_PATH = Path("data/processed/val_predictions.parquet")
ZONES = ["SE1", "SE2", "SE3", "SE4"]
TARGET = "price"

HORIZON_DAYS = 2
WINDOW_DAYS = 28


def evaluate_intervals(df, low, high, title):
    inside = (df[TARGET] >= low) & (df[TARGET] <= high)
    width = high - low

    print(f"\n{title}")
    print(
        f"coverage: {inside.mean():.1%}   (target 80%)   mean width: {width.mean():.4f}"
    )

    out = df.assign(inside=inside, width=width)
    for zone in ZONES:
        m = out["zone"] == zone
        print(
            f"  {zone}  coverage {out.loc[m, 'inside'].mean():.1%}  "
            f"width {out.loc[m, 'width'].mean():.4f}"
        )


def conformal_correction(y, low, high, target=0.8):
    scores = np.maximum(low - y, y - high)
    return np.quantile(scores, target)


def rolling_correction(timestamps, y, low, high, target=0.8):
    scores = np.maximum(low - y, y - high)
    days = timestamps.dt.floor("D")
    q = np.full(len(y), np.nan)

    for day in days.unique():
        end = day - pd.Timedelta(days=HORIZON_DAYS)
        past = (days < end) & (days >= end - pd.Timedelta(days=WINDOW_DAYS))
        if past.sum() > 0:
            q[(days == day).values] = np.quantile(scores[past.values], target)

    return q


def main():
    df = pd.read_parquet(PREDICTIONS_PATH)
    y = df[TARGET].values
    low = df["pred_low"].values
    high = df["pred_high"].values

    # 1. Raw intervals
    evaluate_intervals(df, low, high, "raw intervals")
    print(
        f"above interval: {(y > high).mean():.1%}   below: {(y < low).mean():.1%}"
        "   (each should be ~10%)"
    )

    # 2. Coverage over time
    inside = (y >= low) & (y <= high)
    months = df["timestamp"].dt.tz_localize(None).dt.to_period("M")
    print("\ncoverage by month:")
    print(pd.Series(inside).groupby(months.values).mean().to_string())

    # 3. Static conformal: calibrate on the first half, check on the second
    cal = (df["timestamp"] <= df["timestamp"].quantile(0.5)).values
    chk = ~cal
    q = conformal_correction(y[cal], low[cal], high[cal])
    print(f"\nstatic correction: {q:.4f}")
    evaluate_intervals(
        df[chk], low[chk] - q, high[chk] + q, "static conformal (second half)"
    )

    # 4. Rolling calibration, per zone
    q_roll = np.full(len(df), np.nan)
    for zone in ZONES:
        m = (df["zone"] == zone).values
        q_roll[m] = rolling_correction(df.loc[m, "timestamp"], y[m], low[m], high[m])

    ok = ~np.isnan(q_roll)
    evaluate_intervals(
        df[ok],
        low[ok] - q_roll[ok],
        high[ok] + q_roll[ok],
        "rolling calibration (per zone)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
