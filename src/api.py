from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from sqlalchemy.dialects.postgresql import insert

from src.db import Prediction, Session, init_db
from src.train import CATEGORICAL, FEATURES

MODEL_DIR = Path("models")
DATA_PATH = Path("data/processed/dataset.parquet")
ZONES = ["SE1", "SE2", "SE3", "SE4"]

models = {}
dataset = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data once at startup, not per request."""
    global dataset

    models["median"] = lgb.Booster(model_file=str(MODEL_DIR / "lgbm.txt"))
    models["low"] = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_q10.txt"))
    models["high"] = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_q90.txt"))

    dataset = pd.read_parquet(DATA_PATH)
    init_db()

    yield


app = FastAPI(title="Swedish electricity price forecast", lifespan=lifespan)


def features_for(zone, timestamp):
    """Find the feature row for one zone and hour."""
    row = dataset[(dataset["zone"] == zone) & (dataset["timestamp"] == timestamp)]
    if row.empty:
        raise HTTPException(404, f"no data for {zone} at {timestamp}")

    X = row[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    return X


def log_prediction(zone, target_time, low, median, high):
    """Store the forecast. Duplicates from a re-run are ignored."""
    stmt = (
        insert(Prediction)
        .values(
            zone=zone,
            target_time=target_time,
            issued_at=datetime.now(timezone.utc),
            pred_low=low,
            pred_median=median,
            pred_high=high,
        )
        .on_conflict_do_nothing(constraint="uq_prediction")
    )

    with Session() as session:
        session.execute(stmt)
        session.commit()


@app.get("/health")
def health():
    return {"status": "ok", "rows": len(dataset)}


@app.get("/predict")
def predict(zone: str, timestamp: datetime):
    if zone not in ZONES:
        raise HTTPException(400, f"zone must be one of {ZONES}")

    ts = pd.Timestamp(timestamp).tz_convert("Europe/Stockholm")
    X = features_for(zone, ts)

    low = float(models["low"].predict(X)[0])
    median = float(models["median"].predict(X)[0])
    high = float(models["high"].predict(X)[0])

    log_prediction(zone, ts, low, median, high)

    return {
        "zone": zone,
        "timestamp": ts.isoformat(),
        "low": low,
        "median": median,
        "high": high,
    }
