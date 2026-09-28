"""Find how much to widen the 80% interval when price_lag_2d is missing.
Copy the result by hand to HORIZON_CORRECTION in daily_job.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling.train import (
    CATEGORICAL,
    DATA_DIR,
    FEATURES,
    QUANTILES,
    TARGET,
    prepare,
    train_model,
)

RESULTS_PATH = Path("results/horizon_calibration.csv")


def blank_lag(X):
    """Copy of X with price_lag_2d empty, like forecasts past 2 days."""
    X = X.copy()
    X["price_lag_2d"] = np.nan
    return X


def coverage(y, low, high):
    """Share of y inside [low, high]."""
    return float(((y >= low) & (y <= high)).mean())


def main():
    """Compare coverage with and without the lag, and find the correction."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")

    X_train, y_train = prepare(train)
    X_val, y_val = prepare(val)

    preds = {}
    for alpha in [0.1, 0.9]:
        model = train_model(
            X_train, y_train, X_val, y_val, objective="quantile", alpha=alpha
        )
        preds[("with_lag", alpha)] = model.predict(
            X_val, num_iteration=model.best_iteration
        )
        preds[("no_lag", alpha)] = model.predict(
            blank_lag(X_val), num_iteration=model.best_iteration
        )

    for case in ["with_lag", "no_lag"]:
        low = preds[(case, 0.1)]
        high = preds[(case, 0.9)]
        print(
            f"{case:9}  coverage {coverage(y_val, low, high):.1%}  "
            f"width {np.mean(high - low):.4f}"
        )

    low_nl = preds[("no_lag", 0.1)]
    high_nl = preds[("no_lag", 0.9)]

    # Fit and checked on the same rows, so "after" is always ~80%.
    scores = np.maximum(low_nl - y_val.values, y_val.values - high_nl)
    q = float(np.quantile(scores, 0.8))

    print(f"\ncorrection for horizons > 2 days: {q:.4f}")
    print(
        f"after correction: coverage "
        f"{coverage(y_val, low_nl - q, high_nl + q):.1%}  "
        f"width {np.mean((high_nl + q) - (low_nl - q)):.4f}"
    )

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {
                "correction": q,
                "coverage_before": coverage(y_val, low_nl, high_nl),
                "coverage_after": coverage(y_val, low_nl - q, high_nl + q),
                "width_before": float(np.mean(high_nl - low_nl)),
                "width_after": float(np.mean((high_nl + q) - (low_nl - q))),
            }
        ]
    ).to_csv(RESULTS_PATH, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
