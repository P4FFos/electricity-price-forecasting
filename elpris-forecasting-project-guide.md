# Swedish Electricity Price Forecasting — Build Guide

**Timeline:** mid-September → mid-December 2026
**Budget:** 2–3 h/day (the ECG course project keeps its own separate slot)
**Zones:** SE1–SE4, with SE3 (Stockholm) and SE4 (Malmö) as the primary pair

---

## The framing decision that makes this a real project

Nord Pool publishes day-ahead prices around 13:00 for the following day. If your model
"predicts tomorrow's price" using anything available after that publication, you have
predicted a number the market has already printed. That is leakage, and it produces
suspiciously good metrics.

**Valid framings — pick one and write it at the top of your README:**

- **A. Pre-auction forecast.** Predict tomorrow's hourly prices using only information
  available *before* the day-ahead result is published.
- **B. Multi-day horizon.** Predict prices 2–7 days ahead, where no market price exists
  yet. Cleaner, harder, more defensible.

Recommended: **B**, with A as a secondary experiment. B has no ambiguity about what was
knowable when.

**Hard rule enforced in code:** every feature for a prediction at target time `t` must be
derivable from data timestamped at or before the forecast issue time `t₀`. Write a unit
test for this. It is the single most important piece of engineering in the project.

---

## Phase 0 — Get the data (Days 1–3)

No API key, no registration, no request. This is why we picked it.

**Price data**

- elprisetjustnu.se: `https://www.elprisetjustnu.se/api/v1/prices/{YYYY}/{MM}-{DD}_{ZONE}.json`
  — e.g. `.../2026/09-10_SE3.json`. Open and free; they ask for attribution if you
  publish results. Add the credit line to your README.
- mgrey.se/espot: day-ahead prices in JSON with history from 2022-09-01, sourced from
  the ENTSO-E Transparency Platform. Useful as a cross-check and for older history.

**Weather data**

- SMHI open data API (keyless). Pull temperature and wind speed for one representative
  station per zone — e.g. Luleå (SE1), Sundsvall (SE2), Stockholm (SE3), Malmö (SE4).

**Steps**

1. Write `fetch_prices.py`: loop over dates × zones, save raw JSON to
   `data/raw/prices/{zone}/{date}.json`. Be polite — add a small delay between requests.
2. Backfill as far as the endpoints allow, for all four zones.
3. Write `fetch_weather.py` for SMHI, same pattern.
4. Verify coverage: plot a calendar heatmap of which (date, zone) pairs you actually
   have. Gaps are normal; know where they are.

**Done when:** raw JSON on disk for all four zones across your full available history,
with a documented coverage map.

---

## ⚠️ The structural break you must handle

From **1 October 2025** Sweden moved from hourly to quarter-hourly pricing. The API
returns **96 prices per day instead of 24** after that date; the data structure itself is
unchanged.

This is not an annoyance — it is one of the best things about this project. Handle it
deliberately:

1. **Normalise.** Aggregate post-October-2025 quarter-hourly prices to hourly means so
   you have one consistent series. Keep the raw quarter-hourly data too.
2. **Study the regime.** Did volatility change after the switch? Intraday shape? Spike
   frequency?
3. **Make it an experiment.** Train a model on pre-switch data only, evaluate on
   post-switch data. Report the degradation. That is a real, dated, documented concept
   drift event — most portfolio projects have to simulate drift artificially.

Write this up as its own README section. It is a differentiator.

---

## Phase 1 — Understand the data (Week 1–2)

In a notebook. Answer in writing:

- Price distribution per zone. Mean, median, tails, negative prices (they happen).
- How different are SE1/SE2 from SE3/SE4? Quantify the spread, don't just assert it.
- Seasonality: hour-of-day, day-of-week, month. Plot all three.
- Spikes: how often does price exceed 3× the rolling median? Are spikes clustered?
- Correlation between zones. Does SE3 lead SE4 or move with it?
- Weather relationship: scatter price vs temperature, price vs wind speed, per zone and
  season.
- Data quality: missing dates, currency/unit consistency, DST transitions (23- and
  25-hour days — these *will* break naive code).

**Deliverable:** `notebooks/01_data_understanding.ipynb` with plots and written answers.

Same muscle as your ECG sections 2.1/2.2/2.4. Do not skip the writing.

---

## Phase 2 — Build the dataset (Week 2–4)

**Row definition:** one row = one (zone, target hour, forecast horizon) triple.

```
features known at t₀  →  price at target hour t, where t - t₀ ∈ {2d … 7d}
```

**Feature groups**

| Group | Features |
|---|---|
| Calendar | hour, day of week, month, is_weekend, Swedish public holiday |
| Price history | price at same hour 1/2/3/7 days before t₀; rolling mean/std over 7 and 30 days; recent spike count |
| Weather forecast | temperature and wind at target hour from the forecast issued at t₀ (see caveat) |
| Cross-zone | price spread SE3−SE4, SE1−SE3 as of t₀ |
| Regime | binary flag for post-2025-10-01 quarter-hourly era |

**Weather caveat, and how to be honest about it:** you have historical *observations*,
not historical *forecasts*. Using observed weather at the target hour is technically
look-ahead — a real deployment would use a forecast, which carries error. Two options:
(a) use observed weather and state clearly in the README that this is an optimistic upper
bound, or (b) degrade observations with synthetic noise scaled to typical 3-day forecast
error. Either is fine. Silently using observations and not mentioning it is not.

**Splitting:** strictly by time. Earliest ~70% train, next ~15% validation, latest ~15%
test. Never random. Rolling aggregates must be computed causally — no window may reach
past `t₀`.

**Done when:** one command produces `train.parquet`, `val.parquet`, `test.parquet` with
documented row counts and date ranges, and the look-ahead unit test passes.

---

## Phase 3 — Baselines (Week 4–5)

Build these before any model. Write the numbers down so you cannot move the goalposts.

1. **Seasonal naive.** Price at the same hour, 7 days earlier.
2. **Persistence.** Last known price at that hour before `t₀`.
3. **Climatology.** Mean price for this zone × hour × month, computed on training data only.

Report MAE, RMSE, and MAPE (careful — MAPE breaks near zero and negative prices; consider
sMAPE or just drop it).

**Expect these to be strong.** Electricity prices are heavily seasonal and autocorrelated.
Beating seasonal naive by 10–15% at a 3-day horizon is a genuine result.

---

## Phase 4 — The model (Week 5–8)

**Choice:** gradient boosting (LightGBM or XGBoost). Not a neural net. Tabular data,
trains in seconds, gives feature importances you can discuss.

**Process**

1. Train with defaults. Compare to all three baselines, per horizon.
2. Feature importance → prune, and investigate anything surprising.
3. **Error analysis.** Which hours, months, zones, and price levels are worst? Plot
   predicted vs actual. Inspect the 20 worst predictions individually — they will almost
   all be spikes.
4. **Spike handling.** Decide explicitly: are you modelling the median case, or trying to
   catch spikes? Consider a secondary binary classifier for "will price exceed threshold X",
   reported separately with precision/recall. This is often more useful than squeezing MAE
   and makes a better interview story.
5. *Then* tune hyperparameters.

Train separate models per horizon (2d, 3d, … 7d), or one model with horizon as a feature.
Try both; report which wins.

**Track every run in MLflow from run one.** Config, features, metrics, seed.

**Done when:** your model beats seasonal naive on the validation set at every horizon,
with margins written down and error analysis complete.

---

## Phase 5 — The two headline experiments (Week 8–9)

This is what separates the project from a tutorial.

**A. Zone transfer**

| Train | Test |
|---|---|
| SE3 | SE3 |
| SE4 | SE4 |
| SE3 | SE4 |
| SE4 | SE3 |
| SE1+SE2 | SE3+SE4 |
| all four | all four |

Northern zones (SE1/SE2) are surplus and cheaper; southern zones (SE3/SE4) are deficit
and more expensive, especially on cold, low-wind days. Quantify the transfer loss and
explain it physically. Which features transfer, which are zone-specific?

**B. Regime shift across the quarter-hourly switch**

Train on pre-2025-10-01 only. Evaluate on post-switch data. Report degradation and
diagnose it: changed volatility, changed intraday shape, or just time passing? Control
for the latter by also testing a model trained on an equally distant pre-switch window.

Honest negative results read as maturity here. "Cross-zone transfer cost 28% MAE, driven
by wind-sensitivity differences" beats a suspiciously clean number.

---

## Phase 6 — Serve it (Week 9–11)

1. **Prediction API.** FastAPI. Given zone and target datetime, return predicted price
   plus a prediction interval. Load model at startup, not per request.
2. **Dockerize.** App + Postgres via docker-compose.
3. **Daily job.** Fetch yesterday's actual prices and today's weather, store, and produce
   fresh forecasts.
4. **Deploy** somewhere it stays up. Live beats runnable.
5. **Dashboard.** Forecast vs actual over time, per zone, with error tracked as days pass.
   Streamlit is fine — do not spend two weeks on frontend.
6. **Drift monitoring.** Rolling MAE over the last 30 days, alert when it exceeds a
   threshold. You have a real drift event in your history to justify why this matters.
7. **Scheduled retraining**, weekly, with metrics logged to MLflow.

**Done when:** you can share a URL and it shows live forecasts with accumulating accuracy
history.

---

## Phase 7 — Write it up (Week 11–12)

The README is what gets read. Budget real time.

1. **What and why** — two sentences.
2. **Why this isn't trivial** — the day-ahead publication leakage trap, and how your
   framing avoids it. This paragraph signals more than any metric.
3. **Architecture diagram** — fetch → store → features → model → API → dashboard.
4. **Data** — sources, attribution to elprisetjustnu.se, coverage, known quality issues.
5. **The quarter-hourly regime change** — what it is, how you handled it, what it cost.
6. **Results** — baselines vs model per horizon, plus both experiment matrices.
7. **What didn't work.** Specific. Disproportionately persuasive.
8. **Limitations** — weather-forecast caveat, spike modelling, history length.
9. **Reproduce it** — exact commands.

Prepare a **3-minute spoken version**: problem → the leakage trap → baseline → result →
what you'd do next. Rehearse it out loud.

---

## Repository layout

```
elpris-forecast/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── src/
│   ├── fetch_prices.py
│   ├── fetch_weather.py
│   ├── build_dataset.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── api.py
├── tests/
│   └── test_no_lookahead.py     # the most important file here
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_error_analysis.ipynb
├── configs/
├── dashboard/
└── results/
    ├── metrics.csv
    └── figures/
```

---

## Working rules for a first ML project

- **Commit daily**, with real messages.
- **Notebooks explore; scripts produce.** Anything run twice moves to `src/`.
- **Baseline numbers go in writing before modelling.**
- **Seeds everywhere.** Reproducibility is graded in your ECG project for a reason.
- **A great result means suspect leakage first.** Here that means: check whether a
  day-ahead price sneaked into your features.
- **Watch DST.** Sweden has 23- and 25-hour days twice a year. They break naive hour
  indexing and you will lose an evening to this if unprepared.
- **No deep learning.** It will not beat gradient boosting on this and costs three weeks.

---

## Milestone checkpoints

| Date | Must be true |
|---|---|
| ~22 Sep | Full price + weather history on disk, all four zones, coverage mapped |
| ~6 Oct | Data understanding notebook done and written up |
| ~20 Oct | Dataset builder produces splits; look-ahead test passes |
| ~3 Nov | Baselines measured; model beats seasonal naive at all horizons |
| ~17 Nov | Zone transfer + regime shift experiments complete; error analysis done |
| ~1 Dec | API + dashboard deployed and live |
| ~15 Dec | README complete; 3-minute demo rehearsed |

**Cut in this order if behind:** cross-zone features → spike classifier → retraining
automation → dashboard polish. **Never cut:** the leakage discipline, the baseline
comparison, the two headline experiments, or the README.

---

## Known risks

| Risk | Mitigation |
|---|---|
| Small gains over seasonal naive | Expected. Frame around the baseline comparison and honest error analysis, not absolute accuracy. |
| Weather look-ahead | State the caveat explicitly, or degrade observations with forecast-scale noise. |
| Quarter-hourly switch breaks parsing | Handle both schemas from day one; normalise to hourly, keep raw. |
| Spikes dominate error | Model them separately as classification; report both tracks. |
| ECG deadlines collide (14 Oct, Submission 2) | ECG wins — hard deadline, group depends on you. Phase 2 here can absorb a week's slip. |
