"""
Fetch temperature and wind history from SMHI for one station per bidding zone.

API: https://opendata-download-metobs.smhi.se/api/version/1.0
     /parameter/{param}/station/{station}/period/corrected-archive/data.csv

Data provided by SMHI (https://www.smhi.se).
"""

import sys
from pathlib import Path

import requests

BASE_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/1.0"
    "/parameter/{param}/station/{station}/period/corrected-archive/data.csv"
)

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


def fetch_one(zone, station, name, param):
    """Fetch one parameter for one station. Skips if already downloaded"""
    path = RAW_DIR / f"{zone}_{name}.csv"

    if path.exists():
        print(f"{zone} {name}  skipped")
        return

    url = BASE_URL.format(param=param, station=station)

    try:
        response = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        print(f"{zone} {name}  request failed: {exc}")
        return

    if response.status_code != 200:
        print(f"{zone} {name}  HTTP {response.status_code}")
        return

    if not response.text.startswith("Stationsnamn"):
        print(f"{zone} {name}  unexpected file format")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    print(f"{zone} {name}  ok  ({len(response.text):,} bytes)")


def main():
    for zone, station in STATIONS.items():
        for name, param in PARAMETERS.items():
            fetch_one(zone, station, name, param)
    return 0


if __name__ == "__main__":
    sys.exit(main())
