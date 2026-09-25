import { useEffect, useState } from "react";
import { getJson } from "./api.js";
import FanChart from "./FanChart.jsx";
import { formatDateTime, formatPercent, formatPrice } from "./format.js";

const ZONES = ["SE1", "SE2", "SE3", "SE4"];
const HISTORY_DAYS = 14;
const METRICS_DAYS = 30;
const TABLE_HOURS = 24;

function toPoints(history) {
  return history.points.map((p) => ({
    t: Date.parse(p.target_time),
    low: p.low,
    median: p.median,
    high: p.high,
    actual: p.actual,
    band: [p.low, p.high],
  }));
}

function ZoneSelector({ zone, onChange }) {
  return (
    <div className="zones" role="group" aria-label="Bidding zone">
      {ZONES.map((z) => (
        <button key={z} type="button" aria-pressed={z === zone} onClick={() => onChange(z)}>
          {z}
        </button>
      ))}
    </div>
  );
}

function MetricsPanel({ metrics }) {
  if (metrics.n === 0) {
    return (
      <section className="panel metrics-empty">
        No forecast hours in the last {metrics.days} days have an actual price yet, so there is
        nothing to score.
      </section>
    );
  }

  const gap = (metrics.coverage - 0.8) * 100;
  const gapText =
    Math.abs(gap) < 0.05
      ? "on target"
      : `${Math.abs(gap).toFixed(1)} pts ${gap < 0 ? "below" : "above"} target`;

  return (
    <section className="metrics">
      <div className="panel tile">
        <div className="tile-label">Mean absolute error</div>
        <div className="tile-value">
          {formatPrice(metrics.mae)} <span className="unit">EUR/kWh</span>
        </div>
        <div className="tile-note">Median forecast against the actual price</div>
      </div>
      <div className="panel tile">
        <div className="tile-label">80% interval coverage</div>
        <div className="tile-value">{formatPercent(metrics.coverage)}</div>
        <div className="tile-note">
          Target 80% · {gapText}
        </div>
      </div>
      <div className="panel tile">
        <div className="tile-label">Sample size</div>
        <div className="tile-value">{metrics.n.toLocaleString("en-GB")}</div>
        <div className="tile-note">Hours with an actual price, last {metrics.days} days</div>
      </div>
    </section>
  );
}

function RecentTable({ points, now }) {
  const past = points.filter((p) => p.t <= now).slice(-TABLE_HOURS);
  const rows = past.length > 0 ? past : points.slice(0, TABLE_HOURS);
  const caption =
    past.length > 0
      ? `Most recent ${rows.length} hours`
      : `No past hours yet. Showing the next ${rows.length} forecast hours`;

  return (
    <section className="panel">
      <h2>{caption}</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Time</th>
              <th scope="col" className="num">Low</th>
              <th scope="col" className="num">Median</th>
              <th scope="col" className="num">High</th>
              <th scope="col" className="num">Actual</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.t}>
                <td>{formatDateTime(p.t)}</td>
                <td className="num">{formatPrice(p.low)}</td>
                <td className="num">{formatPrice(p.median)}</td>
                <td className="num">{formatPrice(p.high)}</td>
                <td className="num">{formatPrice(p.actual)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Dashboard({ zone, history, metrics }) {
  const points = toPoints(history);
  const now = Date.now();

  if (points.length === 0) {
    return (
      <>
        <MetricsPanel metrics={metrics} />
        <section className="panel status">
          No forecasts for {zone} in the last {HISTORY_DAYS} days.
        </section>
      </>
    );
  }

  return (
    <>
      <MetricsPanel metrics={metrics} />
      <section className="panel">
        <h2>Forecast and actual price, {zone}</h2>
        <p className="subtitle">EUR/kWh · Stockholm time · last {HISTORY_DAYS} days and upcoming</p>
        <ul className="legend">
          <li>
            <span className="key key-band" />
            80% prediction interval
          </li>
          <li>
            <span className="key key-line key-median" />
            Median forecast
          </li>
          <li>
            <span className="key key-line key-actual" />
            Actual price
          </li>
        </ul>
        <FanChart points={points} now={now} />
      </section>
      <RecentTable points={points} now={now} />
    </>
  );
}

export default function App() {
  const [zone, setZone] = useState("SE3");
  const [reloadKey, setReloadKey] = useState(0);
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading" });

    Promise.all([
      getJson(`/history?zone=${zone}&days=${HISTORY_DAYS}`, controller.signal),
      getJson(`/metrics?zone=${zone}&days=${METRICS_DAYS}`, controller.signal),
    ])
      .then(([history, metrics]) => setState({ status: "ready", history, metrics }))
      .catch((err) => {
        if (err.name !== "AbortError") setState({ status: "error", message: err.message });
      });

    return () => controller.abort();
  }, [zone, reloadKey]);

  return (
    <main>
      <header>
        <h1>Electricity price forecast</h1>
        <ZoneSelector zone={zone} onChange={setZone} />
      </header>

      {state.status === "loading" && <section className="panel status">Loading {zone}…</section>}

      {state.status === "error" && (
        <section className="panel status error" role="alert">
          <p>{state.message}</p>
          <button type="button" onClick={() => setReloadKey((k) => k + 1)}>
            Try again
          </button>
        </section>
      )}

      {state.status === "ready" && (
        <Dashboard zone={zone} history={state.history} metrics={state.metrics} />
      )}
    </main>
  );
}
