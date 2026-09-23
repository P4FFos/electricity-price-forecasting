import sys
from pathlib import Path

import pandas as pd

from src.train import DATA_DIR, TARGET, prepare, score, train_model

SWITCH = pd.Timestamp("2025-10-01", tz="Europe/Stockholm")


def run(df, train_end, period_a, period_b, label):
    """Train up to train_end, then score two later periods separately."""
    train = df[df["timestamp"] <= train_end]
    X_train, y_train = prepare(train)

    # A small slice right after training, used only for early stopping.
    stop = df[(df["timestamp"] > train_end) & (df["timestamp"] <= period_a[0])]
    X_stop, y_stop = prepare(stop)

    model = train_model(X_train, y_train, X_stop, y_stop)

    print(f"\n{label}  (trained to {train_end.date()})")
    for name, (start, end) in [("period A", period_a), ("period B", period_b)]:
        part = df[(df["timestamp"] > start) & (df["timestamp"] <= end)]
        X, y = prepare(part)
        pred = model.predict(X, num_iteration=model.best_iteration)

        model_mae = score(y, pred)["mae"]
        base_mae = score(y, part["price_lag_7d"])["mae"]
        gain = (base_mae - model_mae) / base_mae

        print(
            f"  {name}  {start.date()} to {end.date()}   "
            f"model {model_mae:.6f}  baseline {base_mae:.6f}  gain {gain:.1%}"
        )


def main():
    df = pd.read_parquet(DATA_DIR / "dataset.parquet")

    # Real: training ends before the switch, period B is after it.
    run(
        df,
        train_end=pd.Timestamp("2025-06-30", tz="Europe/Stockholm"),
        period_a=(
            pd.Timestamp("2025-07-15", tz="Europe/Stockholm"),
            pd.Timestamp("2025-09-30", tz="Europe/Stockholm"),
        ),
        period_b=(SWITCH, pd.Timestamp("2026-02-15", tz="Europe/Stockholm")),
        label="REAL: period B is post-switch",
    )

    # Placebo: same shape, one year earlier. Both periods pre-switch.
    run(
        df,
        train_end=pd.Timestamp("2024-06-30", tz="Europe/Stockholm"),
        period_a=(
            pd.Timestamp("2024-07-15", tz="Europe/Stockholm"),
            pd.Timestamp("2024-09-30", tz="Europe/Stockholm"),
        ),
        period_b=(
            pd.Timestamp("2024-10-01", tz="Europe/Stockholm"),
            pd.Timestamp("2025-02-15", tz="Europe/Stockholm"),
        ),
        label="PLACEBO: both periods pre-switch",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
