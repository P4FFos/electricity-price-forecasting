"""Download temperature and wind from SMHI, one station per zone.
Writes data/raw/weather/. Data provided by SMHI (https://www.smhi.se)."""

import sys
from pathlib import Path
import time

import requests

BASE_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/1.0"
    "/parameter/{param}/station/{station}/period/{period}/data.csv"
)

# corrected-archive: checked data, but misses the last ~3 months.
# latest-months: those recent months, not fully checked.
PERIODS = ["corrected-archive", "latest-months"]

STATIONS = {
    "SE1": 162860,  # Lulea-Kallax Flygplats
    "SE2": 127310,  # Sundsvall-Timra Flygplats
    "SE3": 97400,  # Stockholm-Arlanda Flygplats
    "SE4": 53300,  # Malmo-Sturup Flygplats
}

PARAMETERS = {
    "temperature": 1,
    "wind_speed": 4,
}

RAW_DIR = Path("data/raw/weather")

STATIC_PERIODS = {"corrected-archive"}
MAX_AGE_HOURS = 12


def fetch_one(zone, station, name, param, period):
    """Download one parameter for one station and period.
    The archive is downloaded once; latest-months again after MAX_AGE_HOURS."""
    path = RAW_DIR / f"{zone}_{name}_{period}.csv"
    label = f"{zone} {name} {period}"

    # Check before the request, so skipped files are not downloaded.
    if path.exists() and period in STATIC_PERIODS:
        print(f"{label}  skipped (static)")
        return

    if is_fresh(path):
        print(f"{label}  skipped (fresh)")
        return

    url = BASE_URL.format(param=param, station=station, period=period)

    try:
        response = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        print(f"{label}  request failed: {exc}")
        return

    if response.status_code != 200:
        print(f"{label}  HTTP {response.status_code}")
        return

    # utf-8-sig removes the BOM, so the header check below works.
    text = response.content.decode("utf-8-sig")

    if not text.startswith("Stationsnamn"):
        print(f"{label}  unexpected file format")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"{label}  ok  ({len(text):,} bytes)")


def is_fresh(path):
    """True if the file exists and is newer than MAX_AGE_HOURS."""
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < MAX_AGE_HOURS


def main():
    """Download all parameters and periods for every zone."""
    for zone, station in STATIONS.items():
        for name, param in PARAMETERS.items():
            for period in PERIODS:
                fetch_one(zone, station, name, param, period)
    return 0


if __name__ == "__main__":
    sys.exit(main())
