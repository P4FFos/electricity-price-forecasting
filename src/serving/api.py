"""FastAPI service for forecast history and accuracy.
Also serves the React dashboard if frontend/dist exists."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.serving.db import Prediction, Session, init_db

DATA_PATH = Path("data/processed/dataset.parquet")
FRONTEND_DIR = Path("frontend/dist")
ZONES = ["SE1", "SE2", "SE3", "SE4"]

dataset = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the dataset and create the table at startup."""
    global dataset

    dataset = pd.read_parquet(DATA_PATH)
    init_db()

    yield


app = FastAPI(title="Swedish electricity price forecast", lifespan=lifespan)


def fill_actuals():
    """Add the real price to forecasts whose hour has passed.
    Same as daily_job.fill_actuals, but uses the dataset in memory."""
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


@app.get("/health")
def health():
    """Health check with the number of rows loaded."""
    return {"status": "ok", "rows": len(dataset)}


@app.post("/admin/fill-actuals")
def admin_fill_actuals():
    """Run fill_actuals on demand."""
    return {"filled": fill_actuals()}


@app.get("/history")
def history(zone: str, days: int = 14):
    """Latest forecast per hour for the last `days` days, with real prices."""
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

    # An hour can have one forecast per daily run. Rows are sorted newest
    # first, so keeping the first row keeps the latest forecast.
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
    """MAE and interval coverage over the last `days` days."""
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
    """Load the dataset again after the daily job updates it."""
    global dataset
    dataset = pd.read_parquet(DATA_PATH)
    return {"rows": len(dataset), "last": dataset["timestamp"].max().isoformat()}


# Serve the dashboard if it was built (npm --prefix frontend run build).
if (FRONTEND_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def frontend_index():
        """Serve the dashboard."""
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.exception_handler(StarletteHTTPException)
    async def frontend_fallback(request: Request, exc: StarletteHTTPException):
        """Serve the dashboard for unknown GET paths.
        Errors from real API routes keep their normal JSON response."""
        unmatched = "endpoint" not in request.scope
        if exc.status_code == 404 and unmatched and request.method in ("GET", "HEAD"):
            return FileResponse(FRONTEND_DIR / "index.html")
        return await http_exception_handler(request, exc)
