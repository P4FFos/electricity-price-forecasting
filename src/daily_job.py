import logging
import subprocess
import sys
from datetime import datetime, timezone

import lightgbm as lgb
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from src.db import Prediction, Session, init_db
from src.train import CATEGORICAL, FEATURES
import os
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily_job")

MODEL_DIR = "models"
DATA_PATH = "data/processed/dataset.parquet"
API_URL = os.getenv("API_URL", "http://api:8000")

HORIZON_CORRECTION = 0.0118  # results/horizon_calibration.csv
LAG_HORIZON_DAYS = 2

STEPS = [
    [sys.executable, "-m", "src.fetch_prices", "--recent", "14"],
    [sys.executable, "-m", "src.load_prices"],
    [sys.executable, "-m", "src.fetch_weather"],
    [sys.executable, "-m", "src.load_weather"],
    [sys.executable, "-m", "src.build_dataset"],
]


def refresh_data():
    """Fetch the latest prices and weather, then rebuild the dataset."""
    for step in STEPS:
        log.info("running %s", " ".join(step))
        result = subprocess.run(step, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("failed: %s", result.stderr[-2000:])
            raise RuntimeError(f"{step[-1]} failed")


def forecast_future():
    """Predict every future row and store the forecasts."""
    dataset = pd.read_parquet(DATA_PATH)
    future = dataset[dataset["price"].isna()].copy()

    if future.empty:
        log.warning("no future rows to forecast")
        return 0

    X = future[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")

    median = lgb.Booster(model_file=f"{MODEL_DIR}/lgbm.txt")
    low = lgb.Booster(model_file=f"{MODEL_DIR}/lgbm_q10.txt")
    high = lgb.Booster(model_file=f"{MODEL_DIR}/lgbm_q90.txt")

    future["pred_median"] = median.predict(X)
    future["pred_low"] = low.predict(X)
    future["pred_high"] = high.predict(X)

    future = apply_horizon_correction(future)

    issued = datetime.now(timezone.utc)
    rows = [
        {
            "zone": r.zone,
            "target_time": r.timestamp,
            "issued_at": issued,
            "pred_low": float(r.pred_low),
            "pred_median": float(r.pred_median),
            "pred_high": float(r.pred_high),
        }
        for r in future.itertuples()
    ]

    stmt = (
        insert(Prediction)
        .values(rows)
        .on_conflict_do_nothing(constraint="uq_prediction")
    )

    with Session() as session:
        session.execute(stmt)
        session.commit()

    return len(rows)


def apply_horizon_correction(future):
    beyond = future["price_lag_2d"].isna()
    future.loc[beyond, "pred_low"] -= HORIZON_CORRECTION
    future.loc[beyond, "pred_high"] += HORIZON_CORRECTION
    return future


def fill_actuals():
    """Fill in the real price for forecasts whose hour has passed."""
    dataset = pd.read_parquet(DATA_PATH)
    known = dataset[dataset["price"].notna()].set_index(["zone", "timestamp"])["price"]

    with Session() as session:
        pending = session.query(Prediction).filter(Prediction.actual.is_(None)).all()

        filled = 0
        for row in pending:
            key = (
                row.zone,
                pd.Timestamp(row.target_time).tz_convert("Europe/Stockholm"),
            )
            if key in known.index:
                row.actual = float(known.loc[key])
                filled += 1

        session.commit()

    return filled


def reload_api():
    try:
        r = requests.post(f"{API_URL}/admin/reload", timeout=30)
        log.info("api reload: %s", r.json())
    except requests.RequestException as exc:
        log.warning("api reload failed: %s", exc)


def main():
    init_db()

    refresh_data()

    n = forecast_future()
    log.info("stored %d forecasts", n)

    f = fill_actuals()
    log.info("filled %d actuals", f)

    reload_api()

    return 0


if __name__ == "__main__":
    sys.exit(main())
