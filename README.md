# Swedish Electricity Price Forecasting

Forecasting electricity spot prices for Sweden's four bidding zones (SE1-SE4),
2 to 7 days ahead.

**MAE 0.0267 EUR/kWh on unseen test data, 26.6% better than a seasonal naive
baseline.**

---

## The problem

Nord Pool publishes tomorrow's price at around 13:00 today. A model that
predicts tomorrow's price using data from after 13:00 is just repeating a number
that already exists. The score looks great and means nothing.

This project forecasts 2 to 7 days ahead instead. At that distance no published
price exists yet, so the model has to actually predict something.

---

## Results

### Baselines


| Baseline | MAE | RMSE |
|---|---|---|
| Seasonal naive (same hour, 7 days earlier) | 0.0316 | 0.0476 |
| Persistence (same hour, 2 days earlier) | 0.0317 | 0.0472 |
| Climatology (zone x hour x month mean) | 0.0396 | 0.0529 |

Persistence wins in the north, seasonal naive in the south: northern prices are
more autocorrelated, southern prices have stronger weekly structure.

### Final model

LightGBM, 10 features, evaluated once on the test set (February 2026 onward).
Every modelling decision was made on validation.

| Zone | MAE | RMSE | Baseline MAE | Gain |
|---|---|---|---|---|
| All | 0.0267 | 0.0383 | 0.0364 | 26.6% |
| SE1 | 0.0212 | 0.0320 | 0.0310 | 31.4% |
| SE2 | 0.0227 | 0.0338 | 0.0324 | 29.9% |
| SE3 | 0.0279 | 0.0372 | 0.0363 | 23.1% |
| SE4 | 0.0350 | 0.0481 | 0.0459 | 23.8% |

Validation MAE was 0.0237 against a baseline of 0.0316. Test error is higher
because the test period is harder, but the gain is the same or better, so the
model keeps its advantage on unseen data.

The test split is defined by a fixed date and grows as new data arrives, so these
numbers drift slightly between runs. All figures are from the run on 2026-09-25.

---

## Experiments

### Zone transfer

| Train on | Test on | MAE | vs own zone |
|---|---|---|---|
| SE1 | SE1 | 0.0225 | — |
| SE4 | SE4 | 0.0273 | — |
| SE1 | SE2 | 0.0232 | 2% better than SE2's own model |
| SE1 | SE4 | 0.0386 | 41% worse |
| SE4 | SE1 | 0.0387 | 72% worse |
| SE1+SE2 | SE3+SE4 | 0.0330 | 31% worse |
| SE3+SE4 | SE1+SE2 | 0.0249 | 6% worse |

SE1 and SE2 are effectively one market (correlation 0.992) — a model trained on
SE1 predicts SE2 better than SE2's own model, because it has more data.

**The transfer is asymmetric.** South to north costs 6%, north to south costs
31% (each against a model trained on the target zones themselves). A southern model has seen high prices and can predict low ones. A northern
model has never seen prices above a certain level and cannot reach them.


### October 2025 pricing change

On 2025-10-01 Sweden moved from hourly to quarter-hourly pricing. Two confounds
make a naive before/after comparison useless: winter is harder than summer, and
later periods are further from the training data.

The control is a placebo: the same experiment one year earlier, where both
periods are before the switch.

| Run | Period | Gain over baseline |
|---|---|---|
| Real | Jul-Sep 2025 | 21.0% |
| Real | Oct 2025 - Feb 2026 (post-switch) | 21.3% |
| Placebo | Jul-Sep 2024 | 20.4% |
| Placebo | Oct 2024 - Feb 2025 | 21.0% |

Raw MAE degrades in both runs, and worse in the placebo. That degradation is
seasonal, not caused by the switch. The gain over baseline controls for it, and
does not fall in either run.

---

## Prediction intervals

Three quantile models (10th, 50th, 90th percentile) give an 80% interval.

Raw coverage on validation was 71.6% — overconfident. Static conformal
calibration did not help: calibrated on a period where the intervals were
already fine, the correction came out near zero (0.0007) and coverage on the
later period fell to 66.6%. It assumes the future looks like the past.

**Rolling calibration** fixed most of it. Each day's interval is widened using
only the errors from the 28 days before the forecast was issued, computed per
zone. Validation coverage rose to 77.9%.

**Horizon calibration.** Live coverage was initially 51.8%, far worse than
validation. Forecasts 3-7 days ahead have no `price_lag_2d` — that price has not
happened yet — but the quantile models were trained only on rows where it
existed. Simulating the missing lag dropped validation coverage from 71.6% to
56.7%, and the intervals got *narrower* (0.0612 to 0.0569). LightGBM sends rows
with a missing feature down a default branch and has no mechanism for widening
when it knows less.

A fixed correction of 0.0119 on each bound beyond the 2-day horizon restores
coverage, at the cost of 42% wider intervals. Live coverage is now 86.6%, above
the 80% target — the correction may slightly over-widen, but the sample is only
two weeks of scored forecasts (1,344).

---

## Data

Both sources are free and open. Neither needs an API key.

**Prices — [elprisetjustnu.se](https://www.elprisetjustnu.se).** One JSON file
per day per zone, from 2022-11-01. No missing days. Prices exclude VAT and taxes.

**Weather — [SMHI](https://www.smhi.se).** Temperature and wind, one airport per
zone: Luleå-Kallax (SE1), Sundsvall-Timrå (SE2), Stockholm-Arlanda (SE3),
Malmö-Sturup (SE4). Two endpoints combined: `corrected-archive` for checked
history, `latest-months` for the recent months it omits.

### What the data looks like

- **Three price levels, not four.** SE1 and SE2 are nearly identical (0.992).
  SE3 sits closer to SE4 (0.914) than to SE1 (0.793).
- **Hour and season interact.** The winter evening peak is at 17:00, the summer
  one at 20:00. In summer the midday dip goes below the overnight price — solar.
- **Negative prices have two causes.** Spring midday (solar) and September
  overnight (wind). Negative hours average 4.93 m/s wind against 3.30 normally,
  and 10.7 °C against 4.74: warm and windy at once.
- **Spikes come in episodes.** 88.2% of spikes are adjacent to another spike.
- **The north-south spread is growing**, from about 0.02 in 2023-24 to 0.04-0.05
  in 2025-26.

### Where the error is

- **Not concentrated.** The worst 20% of rows hold 51% of the total error.
- **The model under-predicts every expensive hour.** The top 5% of prices have
  3.5x the normal error, and the signed error there almost equals the absolute
  error — never too high, always too low.
- **There is a ceiling.** Predictions stop around 0.2 EUR/kWh while real prices
  reach 0.41.
- **The worst errors had no warning.** The 20 largest come from five days. On 14
  October the price two days earlier was 0.036 and the 7-day average 0.050, then
  the price hit 0.41.

---

## Decisions

**Target is EUR, not SEK.** Training on SEK would mean also predicting currency
movements, which have nothing to do with electricity.

**Ten features, not sixteen.** The lags are highly redundant; removing six cost
0.4% MAE. `price_mean_7d` dominates importance — recent price *level* matters
more than any specific past hour.

**Splits are fixed dates, not quantiles.** Otherwise every rebuild would shuffle
rows between train, val and test as data arrives, and old results would stop
being reproducible.

**Weather features are measurements, not forecasts.** A deployed system would
only have a forecast, which carries error, so these features are an optimistic
upper bound. Adding artificial noise would mean inventing an error distribution.

**One station per zone is a shortcut.** SE3 covers Stockholm, Gothenburg,
Uppsala, Örebro and Norrköping; one Arlanda reading cannot represent all of it.

**Timestamps carry a time zone**, stored as Europe/Stockholm, parsed as UTC
first. Sweden has one 23-hour and one 25-hour day a year, so nothing assumes 24.

**Tuning barely helped** — at most 0.6% across learning rate and tree size.
Simpler trees did slightly better, which fits the limited signal. Defaults kept.

---

## Architecture

```
elprisetjustnu.se ──┐
                    ├──> raw files ──> parquet ──> features ──> LightGBM
SMHI ───────────────┘                                              │
                                                                   v
                        Postgres <── daily job ──> forecasts ──> FastAPI
                            │                                      │
                            └──> actuals, scored later             v
                                                            React dashboard
```

The daily job fetches new data, rebuilds the dataset, forecasts the next 7 days,
fills in actuals for hours that have passed, reloads the API, and checks for
drift. Every step is idempotent, so a re-run is safe and a missed day
self-corrects.

```
src/
├── data/          fetching, parsing, feature building
├── modelling/     training, tuning, baselines
├── experiments/   zone transfer, regime shift, evaluation, calibration
└── serving/       API, database, daily job

tests/             the no-look-ahead rule
frontend/          React dashboard
notebooks/         data understanding, error analysis
results/           committed metrics
```

---

## Running it

```bash
docker compose up -d --build
```

The job runs immediately, then every 24 hours.

Without Docker:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data.fetch_prices      # ~5,600 requests, 30-40 min
python -m src.data.load_prices
python -m src.data.fetch_weather     # 8 requests
python -m src.data.load_weather
python -m src.data.build_dataset
python -m src.modelling.baselines
python -m src.modelling.train
```

`data/` is not in git — everything in it rebuilds from the commands above.

---

## Limitations

- Weather features are measurements, not forecasts.
- SE4 weather is missing about 2,800 rows, roughly four months of 2025.
- Wind is reported in whole m/s, so it is coarser than temperature.
- The symmetric interval correction sometimes pushes the lower bound below zero
  on hours where negative prices are unlikely. Clipping at zero would be wrong —
  negative prices are real — but a correction scaling with the predicted level
  would handle this better.
- Drift alerting is log-only.

