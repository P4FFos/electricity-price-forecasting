"""
Train a LightGBM model for the 2-7 day ahead forecast.
Compared against the baselines in results/baselines.csv.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
SEED = 42

BASELINE_MAE = 0.031558  # seasonal naive, from results/baselines.csv

FEATURES = [
    "zone",
    "hour",
    "dayofweek",
    "month",
    "temperature",
    "wind_speed",
    "price_lag_2d",
    "price_lag_7d",
    "price_mean_7d",
    "price_mean_30d",
]

CATEGORICAL = ["zone"]
TARGET = "price"


def prepare(df):
    X = df[FEATURES].copy()

    for col in CATEGORICAL:
        X[col] = X[col].astype("category")

    return X, df[TARGET]


def score(actual, predicted):
    return {
        "mae": float(np.abs(actual - predicted).mean()),
        "rmse": float(np.sqrt(((actual - predicted) ** 2).mean())),
    }


def train_model(X_train, y_train, X_val, y_val):
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": SEED,
    }

    train_set = lgb.Dataset(X_train, y_train, categorical_feature=CATEGORICAL)
    val_set = lgb.Dataset(X_val, y_val, categorical_feature=CATEGORICAL)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )

    return model


def report(model, val, pred):
    overall = score(val[TARGET], pred)
    improvement = (BASELINE_MAE - overall["mae"]) / BASELINE_MAE

    print(f"\nbest iteration: {model.best_iteration}")
    print(f"MAE  {overall['mae']:.6f}   (baseline {BASELINE_MAE:.6f})")
    print(f"RMSE {overall['rmse']:.6f}")
    print(f"improvement: {improvement:.1%}")

    print("\nper zone:")
    val_out = val.assign(pred=pred)
    for zone in ["SE1", "SE2", "SE3", "SE4"]:
        m = val_out["zone"] == zone
        s = score(val_out.loc[m, TARGET], val_out.loc[m, "pred"])
        print(f"  {zone}  MAE {s['mae']:.6f}  RMSE {s['rmse']:.6f}")

    print("\nfeature importance:")
    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(
        ascending=False
    )
    print(imp.to_string())


def main():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")

    X_train, y_train = prepare(train)
    X_val, y_val = prepare(val)

    model = train_model(X_train, y_train, X_val, y_val)
    pred = model.predict(X_val, num_iteration=model.best_iteration)

    report(model, val, pred)
    
    val.assign(pred=pred).to_parquet(DATA_DIR / "val_predictions.parquet", index=False)

    model_path = Path("models/lgbm.txt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    print(f"\nsaved to {model_path}")
    
    val.assign(pred=pred).to_parquet("data/processed/val_predictions.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
