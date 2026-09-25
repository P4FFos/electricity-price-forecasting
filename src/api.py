from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.dialects.postgresql import insert
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.db import Prediction, Session, init_db
from src.train import CATEGORICAL, FEATURES

MODEL_DIR = Path("models")
DATA_PATH = Path("data/processed/dataset.parquet")
FRONTEND_DIR = Path("frontend/dist")
ZONES = ["SE1", "SE2", "SE3", "SE4"]

models = {}
dataset = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dataset

    models["median"] = lgb.Booster(model_file=str(MODEL_DIR / "lgbm.txt"))
    models["low"] = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_q10.txt"))
    models["high"] = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_q90.txt"))

    dataset = pd.read_parquet(DATA_PATH)
    init_db()

    yield


app = FastAPI(title="Swedish electricity price forecast", lifespan=lifespan)


def features_for(zone, timestamp):
    row = dataset[(dataset["zone"] == zone) & (dataset["timestamp"] == timestamp)]
    if row.empty:
        raise HTTPException(404, f"no data for {zone} at {timestamp}")

    X = row[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    return X


def log_prediction(zone, target_time, low, median, high):
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


def fill_actuals():
    with Session() as session:
        pending = session.query(Prediction).filter(Prediction.actual.is_(None)).all()

        filled = 0
        for row in pending:
            target = pd.Timestamp(row.target_time).tz_convert("Europe/Stockholm")
            match = dataset[
                (dataset["zone"] == row.zone) & (dataset["timestamp"] == target)
            ]
            if not match.empty:
                row.actual = float(match["price"].iloc[0])
                filled += 1

        session.commit()

    return filled


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


@app.post("/admin/fill-actuals")
def admin_fill_actuals():
    return {"filled": fill_actuals()}


@app.get("/history")
def history(zone: str, days: int = 14):
    if zone not in ZONES:
        raise HTTPException(400, f"zone must be one of {ZONES}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with Session() as session:
        rows = (
            session.query(Prediction)
            .filter(Prediction.zone == zone, Prediction.target_time >= cutoff)
            .order_by(Prediction.target_time, Prediction.issued_at.desc())
            .all()
        )

    seen = set()
    points = []
    for r in rows:
        if r.target_time in seen:
            continue
        seen.add(r.target_time)
        points.append(
            {
                "target_time": r.target_time.isoformat(),
                "low": r.pred_low,
                "median": r.pred_median,
                "high": r.pred_high,
                "actual": r.actual,
            }
        )

    return {"zone": zone, "points": points}


@app.get("/metrics")
def metrics(zone: str = None, days: int = 30):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with Session() as session:
        q = session.query(Prediction).filter(
            Prediction.actual.isnot(None), Prediction.target_time >= cutoff
        )
        if zone:
            q = q.filter(Prediction.zone == zone)
        rows = q.all()

    if not rows:
        return {
            "zone": zone or "all",
            "days": days,
            "n": 0,
            "mae": None,
            "coverage": None,
        }

    errors = [abs(r.pred_median - r.actual) for r in rows]
    inside = [r.pred_low <= r.actual <= r.pred_high for r in rows]

    return {
        "zone": zone or "all",
        "days": days,
        "n": len(rows),
        "mae": sum(errors) / len(errors),
        "coverage": sum(inside) / len(inside),
    }


@app.post("/admin/reload")
def reload_dataset():
    """Re-read the dataset after the daily job rewrites it."""
    global dataset
    dataset = pd.read_parquet(DATA_PATH)
    return {"rows": len(dataset), "last": dataset["timestamp"].max().isoformat()}


# The built dashboard (npm --prefix frontend run build). Without a build the API
# runs exactly as before.
if (FRONTEND_DIR / "index.html").exists():
    app.mount(
        "/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets"
    )

    @app.get("/", include_in_schema=False)
    def frontend_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.exception_handler(StarletteHTTPException)
    async def frontend_fallback(request: Request, exc: StarletteHTTPException):
        # Fall back to the dashboard only when no route matched at all. A 404
        # raised inside an API route, or a 405 for the wrong method on one,
        # keeps its normal JSON response.
        unmatched = "endpoint" not in request.scope
        if exc.status_code == 404 and unmatched and request.method in ("GET", "HEAD"):
            return FileResponse(FRONTEND_DIR / "index.html")
        return await http_exception_handler(request, exc)
