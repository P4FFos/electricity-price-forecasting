import lightgbm as lgb
import pandas as pd

from src.train import CATEGORICAL, DATA_DIR, SEED, prepare, score

GRID = [
    {"learning_rate": 0.05, "num_leaves": 31},
    {"learning_rate": 0.02, "num_leaves": 31},
    {"learning_rate": 0.02, "num_leaves": 15},
    {"learning_rate": 0.02, "num_leaves": 63},
    {"learning_rate": 0.01, "num_leaves": 31},
]


def main():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    X_train, y_train = prepare(train)
    X_val, y_val = prepare(val)

    for settings in GRID:
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbose": -1,
            "seed": SEED,
            **settings,
        }

        model = lgb.train(
            params,
            lgb.Dataset(X_train, y_train, categorical_feature=CATEGORICAL),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(X_val, y_val, categorical_feature=CATEGORICAL)],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

        pred = model.predict(X_val, num_iteration=model.best_iteration)
        s = score(y_val, pred)
        print(f"{settings} MAE {s['mae']:.6f} trees {model.best_iteration}")


if __name__ == "__main__":
    main()
