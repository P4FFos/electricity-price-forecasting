"""Train LightGBM models for the 2-7 day forecast and save them to models/.
daily_job.py uses FEATURES, so keep it the same as the saved models."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
SEED = 42
QUANTILES = [0.1, 0.5, 0.9]

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
    """Return features and target, with zone as a category."""
    X = df[FEATURES].copy()

    for col in CATEGORICAL:
        X[col] = X[col].astype("category")

    return X, df[TARGET]


def score(actual, predicted):
    """Return MAE and RMSE.
    No MAPE, because prices can be zero or negative."""
    return {
        "mae": float(np.abs(actual - predicted).mean()),
        "rmse": float(np.sqrt(((actual - predicted) ** 2).mean())),
    }


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    objective="regression",
    alpha=None,
    categorical=CATEGORICAL,
):
    """Train LightGBM with early stopping on val.
    For a quantile model, pass objective="quantile" and alpha."""
    params = {
        "objective": objective,
        "metric": "mae" if objective == "regression" else "quantile",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": SEED,
    }
    if alpha is not None:
        params["alpha"] = alpha

    train_set = lgb.Dataset(X_train, y_train, categorical_feature=categorical)
    val_set = lgb.Dataset(X_val, y_val, categorical_feature=categorical)

    return lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def report(model, val, pred):
    """Print val scores against the baseline, and feature importance."""
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
    """Train and save the main and quantile models, and val predictions."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")

    X_train, y_train = prepare(train)
    X_val, y_val = prepare(val)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Point forecast
    model = train_model(X_train, y_train, X_val, y_val)
    pred = model.predict(X_val, num_iteration=model.best_iteration)
    report(model, val, pred)
    model.save_model(str(MODEL_DIR / "lgbm.txt"), num_iteration=model.best_iteration)

    quantile_preds = {}
    for alpha in QUANTILES:
        q_model = train_model(
            X_train, y_train, X_val, y_val, objective="quantile", alpha=alpha
        )
        quantile_preds[alpha] = q_model.predict(
            X_val, num_iteration=q_model.best_iteration
        )
        q_model.save_model(
            str(MODEL_DIR / f"lgbm_q{int(alpha * 100)}.txt"),
            num_iteration=q_model.best_iteration,
        )

    val.assign(
        pred=pred,
        pred_low=quantile_preds[0.1],
        pred_high=quantile_preds[0.9],
    ).to_parquet(DATA_DIR / "val_predictions.parquet", index=False)

    print(f"\nmodels saved to {MODEL_DIR}/")
    print(f"predictions saved to {DATA_DIR / 'val_predictions.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
