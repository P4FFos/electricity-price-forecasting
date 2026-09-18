# Swedish Electricity Price Forecasting

Forecasting electricity spot prices for Sweden's four bidding zones (SE1-SE4),
2 to 7 days ahead

---

## The problem

Nord Pool publishes tomorrow's price at around 13:00 today. A model that predicts
tomorrow's price using data from after 13:00 is just repeating a number that
already exists. The score looks great and means nothing.

This project forecasts 2 to 7 days ahead instead. At that distance no published
price exists yet, so the model has to actually predict something.

The rule: to predict the price at time `t`, only data available at the time the
forecast was made can be used.

---

## Data

Both sources are free and open. Neither needs an API key

### Prices: elprisetjustnu.se

- 2022-11-01 to today, all four zones
- 236,636 rows across 1,415 days
- No missing days, no duplicates, no empty values
- Prices exclude VAT, taxes and surcharges

Data from [Elpriset just nu.se](https://www.elprisetjustnu.se)

### Weather: SMHI Open Data

- 2022-11-01 to 2026-06-01
- 122,404 rows
- Air temperature and wind speed, hourly

| Zone | Station | Name |
|---|---|---|
| SE1 | 162860 | Luleå-Kallax Flygplats |
| SE2 | 127310 | Sundsvall-Timrå Flygplats |
| SE3 | 97400 | Stockholm-Arlanda Flygplats |
| SE4 | 53300 | Malmö-Sturup Flygplats |

Data from [SMHI](https://www.smhi.se)

---

## Decisions

**The target is EUR, not SEK.** The API gives both, plus the exchange rate used
that day. Prices come from ENTSO-E in euro and are converted to SEK daily.
Training on SEK would mean also predicting currency movements, which have nothing
to do with electricity. SEK is kept for display only.

**One weather station per zone is a shortcut.** SE3 covers Stockholm, Gothenburg,
Uppsala, Örebro and Norrköping, and one reading from Arlanda cannot represent all
of that. Airports were chosen because they measure continuously with few gaps. If
this hurts the model, the fix is averaging several stations per zone.

**All timestamps carry a time zone** and are stored in Europe/Stockholm. Source
data is parsed as UTC first, then converted, so daylight saving works correctly.
Sweden has one 23-hour day and one 25-hour day each year, so nothing here assumes
a day has 24 hours.

**Baselines were measured before any model was trained** and are committed in
`results/baselines.csv`. They are not regenerated after the fact.

---

## The hourly to quarter-hourly switch

On 2025-10-01 Sweden moved from hourly to quarter-hourly prices. The API returns
96 prices per day after that date instead of 24, in the same format.

Quarter-hourly prices are averaged into hourly ones so the series stays
consistent, and the original data is kept too. Every row has a `resolution`
column, either `hourly` or `quarter`.

Planned experiment: train only on data from before the switch, test on data from
after, and measure how much accuracy is lost.

Currently about 1,065 days hourly and 350 days quarter-hourly.

---

## Code

```
src/
├── fetch_prices.py     downloads price JSON
├── load_prices.py      turns it into prices.parquet
├── fetch_weather.py    downloads SMHI CSV files
└── load_weather.py     turns them into weather.parquet
```

---

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_prices.py      # about 5,600 requests, 30-40 minutes
python src/load_prices.py

python src/fetch_weather.py     # 8 requests
python src/load_weather.py
```

Update with the latest days:

```bash
python src/fetch_prices.py --recent 3 && python src/load_prices.py
```

The `data/` folder is not in git. Everything in it rebuilds from the commands
above.

---

## Results

### Baselines

Measured on the validation set (20,372 rows, July 2025 to February 2026) before
any model was trained.

| Baseline | MAE | RMSE |
|---|---|---|
| Seasonal naive (same hour, 7 days earlier) | 0.0316 | 0.0476 |
| Persistence (same hour, 2 days earlier) | 0.0317 | 0.0472 |
| Climatology (zone x hour x month mean) | 0.0396 | 0.0529 |

Per zone, taking the better of the two naive methods:

| Zone | Best baseline MAE | Which |
|---|---|---|
| SE1 | 0.0265 | persistence |
| SE2 | 0.0264 | persistence |
| SE3 | 0.0328 | seasonal naive |
| SE4 | 0.0362 | seasonal naive |

Persistence wins in the north, seasonal naive in the south. The northern zones
are more autocorrelated, so two days ago predicts better than a week ago. The
southern zones have stronger weekly structure. MAE and RMSE only, no MAPE: the data contains 8,195 negative prices and the
median is near 0.02, so percentage error is meaningless here. Climatology is worst everywhere because it ignores recent conditions entirely.