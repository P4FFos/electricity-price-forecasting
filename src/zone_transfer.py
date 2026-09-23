import sys
import pandas as pd
from train import DATA_DIR, FEATURES, TARGET, score, train_model
from pathlib import Path 

TRANSFER_FEATURES = [f for f in FEATURES if f != "zone"]


def run(train, val, source, target):
    tr = train[train["zone"].isin(source)]
    va_src = val[val["zone"].isin(source)]
    va_tgt = val[val["zone"].isin(target)]

    model = train_model(
        tr[TRANSFER_FEATURES],
        tr[TARGET],
        va_src[TRANSFER_FEATURES],
        va_src[TARGET],
        categorical=[],
    )
    pred = model.predict(va_tgt[TRANSFER_FEATURES], num_iteration=model.best_iteration)
    return score(va_tgt[TARGET], pred)["mae"]

ZONES = ["SE1", "SE2", "SE3", "SE4"]

EXPERIMENTS = [
    (["SE1"], ["SE1"]),
    (["SE2"], ["SE2"]),
    (["SE3"], ["SE3"]),
    (["SE4"], ["SE4"]),
    (["SE1"], ["SE2"]),
    (["SE1"], ["SE4"]),
    (["SE4"], ["SE1"]),
    (["SE3"], ["SE4"]),
    (["SE1", "SE2"], ["SE3", "SE4"]),
    (["SE3", "SE4"], ["SE1", "SE2"]),
    (ZONES, ZONES),
]


def main():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")

    rows = []
    for source, target in EXPERIMENTS:
        mae = run(train, val, source, target)
        rows.append({
            "train_on": "+".join(source),
            "test_on": "+".join(target),
            "mae": mae,
        })
        print(f"{rows[-1]['train_on']:>8} -> {rows[-1]['test_on']:<8} MAE {mae:.6f}")

    results = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/zone_transfer.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())