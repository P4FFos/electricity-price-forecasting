import sys
from pathlib import Path
import pandas as pd
from train import DATA_DIR, TARGET, prepare, score, train_model

ZONES = ["SE1", "SE2", "SE3", "SE4"]
RESULTS_PATH = Path("results/test_results.csv")


def main():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    X_train, y_train = prepare(train)
    X_val, y_val = prepare(val)
    X_test, y_test = prepare(test)

    model = train_model(X_train, y_train, X_val, y_val)
    pred = model.predict(X_test, num_iteration=model.best_iteration)

    rows = []
    for name, subset, p in [
        ("all", test, pred),
        *[
            (z, test[test["zone"] == z], pred[(test["zone"] == z).values])
            for z in ZONES
        ],
    ]:
        model_s = score(subset[TARGET], p)
        base_mae = score(subset[TARGET], subset["price_lag_7d"])["mae"]
        rows.append(
            {
                "zone": name,
                "mae": model_s["mae"],
                "rmse": model_s["rmse"],
                "baseline_mae": base_mae,
                "gain": (base_mae - model_s["mae"]) / base_mae,
            }
        )

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
