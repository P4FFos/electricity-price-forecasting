import requests
from pathlib import Path
from datetime import date, timedelta

BASE_URL = "https://www.elprisetjustnu.se/api/v1/prices/{year}/{month:02d}-{day:02d}_{zone}.json"
RAW_DIR = Path("data/raw/prices")


def daterange(start, end):
    """Yield each date from start to end, inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def target_path(day, zone):
    """Where this day's JSON lives on disk."""
    return RAW_DIR / zone / f"{day.isoformat()}.json"


def fetch_one(day, zone):
    """Fetch one day for one zone. Returns 'ok', 'skipped', 'missing', or 'error'."""
    path = target_path(day, zone)

    if path.exists():
        return "skipped"

    url = BASE_URL.format(year=day.year, month=day.month, day=day.day, zone=zone)

    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        print(f"{zone} {day}  request failed: {exc}")
        return "error"

    if response.status_code == 404:
        return "missing"
    if response.status_code != 200:
        print(f"{zone} {day}  HTTP {response.status_code}")
        return "error"

    try:
        payload = response.json()
    except ValueError:
        print(f"{zone} {day}  response was not valid JSON")
        return "error"

    if not isinstance(payload, list) or not payload:
        print(f"{zone} {day}  unexpected payload shape")
        return "error"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    return "ok"


def main():
    counts = {"ok": 0, "skipped": 0, "missing": 0, "error": 0}

    for day in daterange(date(2026, 9, 1), date(2026, 9, 13)):
        result = fetch_one(day, "SE3")
        counts[result] += 1
        print(day, result)

    print(counts)


if __name__ == "__main__":
    main()
