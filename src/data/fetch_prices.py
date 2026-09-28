"""Download daily electricity prices from elprisetjustnu.se (data from 2022-11-01).
Prices have no VAT or taxes. Data provided by Elpriset just nu.se."""

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.elprisetjustnu.se/api/v1/prices/{year}/{month:02d}-{day:02d}_{zone}.json"
ZONES = ["SE1", "SE2", "SE3", "SE4"]
EARLIEST = date(2022, 11, 1)
RAW_DIR = Path("data/raw/prices")
MISSING_LOG = Path("data/raw/missing_dates.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_prices")


def build_session():
    """HTTP session that retries failed requests."""
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {"User-Agent": "elpris-forecast/0.1 (student portfolio project)"}
    )
    return session


def daterange(start, end):
    """Yield each day from start to end, both included."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def target_path(day, zone):
    """File path for one day and zone."""
    return RAW_DIR / zone / f"{day.isoformat()}.json"


def fetch_one(session, day, zone, delay):
    """Download one day for one zone.
    Returns 'ok', 'skipped', 'missing' or 'error'."""
    path = target_path(day, zone)

    if path.exists():
        return "skipped"

    url = BASE_URL.format(year=day.year, month=day.month, day=day.day, zone=zone)

    try:
        response = session.get(url, timeout=30)
    except requests.RequestException as exc:
        log.warning("%s %s  request failed: %s", zone, day, exc)
        return "error"

    time.sleep(delay)  # free API, don't spam it

    # 404 means not published yet. Nothing is saved, so the next run tries again.
    if response.status_code == 404:
        return "missing"
    if response.status_code != 200:
        log.warning("%s %s  HTTP %s", zone, day, response.status_code)
        return "error"

    try:
        payload = response.json()
    except ValueError:
        log.warning("%s %s  response was not valid JSON", zone, day)
        return "error"

    # Saved files are never downloaded again, so don't save bad data.
    if not isinstance(payload, list) or not payload:
        log.warning("%s %s  unexpected payload shape", zone, day)
        return "error"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    return "ok"


def main():
    """Download every missing day and zone in the range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=EARLIEST.isoformat(), help="YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=(date.today() + timedelta(days=1)).isoformat(),
        help="YYYY-MM-DD (defaults to tomorrow; day-ahead prices publish ~13:00)",
    )
    parser.add_argument("--zones", nargs="+", default=ZONES, choices=ZONES)
    parser.add_argument("--recent", type=int, help="fetch only the last N days")
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    # Check the dates, then move start up to EARLIEST if needed.
    if end < start:
        log.error("End date is before start date.")
        return 1

    if start < EARLIEST:
        log.warning("History starts %s; clamping start date.", EARLIEST)
        start = EARLIEST

    if end < EARLIEST:
        log.error("Entire range is before %s; nothing to fetch.", EARLIEST)
        return 1

    if args.recent:
        start = end - timedelta(days=args.recent)

    counts = {"ok": 0, "skipped": 0, "missing": 0, "error": 0}
    missing = []

    days = list(daterange(start, end))
    total = len(days) * len(args.zones)
    done = 0

    log.info("Fetching %d day/zone combinations (%s to %s)", total, start, end)

    session = build_session()

    for zone in args.zones:
        for day in days:
            result = fetch_one(session, day, zone, args.delay)
            counts[result] += 1
            if result == "missing":
                missing.append(f"{zone}:{day.isoformat()}")

            done += 1
            if done % 200 == 0:
                log.info("%d/%d  %s", done, total, counts)

    MISSING_LOG.parent.mkdir(parents=True, exist_ok=True)
    MISSING_LOG.write_text(json.dumps(sorted(missing), indent=2), encoding="utf-8")

    log.info("Done. %s", counts)
    log.info("Missing day/zone pairs written to %s", MISSING_LOG)

    if counts["error"]:
        log.warning("There were %d errors. Re-run to retry them.", counts["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
