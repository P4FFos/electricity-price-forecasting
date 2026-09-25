import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatDateTime, formatDay, formatPrice, isStockholmMidnight } from "./format.js";

const axisTick = { fill: "var(--text-muted)", fontSize: 12 };

// Round y ticks (0.05, 0.10, ...) that always include zero
function priceTicks(points) {
  let min = 0;
  let max = 0;
  for (const p of points) {
    min = Math.min(min, p.low, p.actual ?? 0);
    max = Math.max(max, p.high, p.actual ?? 0);
  }
  const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1].find((s) => (max - min) / s <= 6) ?? 1;
  const ticks = [];
  for (let v = Math.floor(min / step) * step; v < max + step - 1e-9; v += step) {
    ticks.push(Number(v.toFixed(4)));
  }
  return ticks;
}

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;

  return (
    <div className="tooltip">
      <div className="tooltip-time">{formatDateTime(p.t)}</div>
      <div className="tooltip-row">
        <span className="key key-line key-actual" />
        <span>Actual</span>
        <span className="num">{formatPrice(p.actual)}</span>
      </div>
      <div className="tooltip-row">
        <span className="key key-line key-median" />
        <span>Median</span>
        <span className="num">{formatPrice(p.median)}</span>
      </div>
      <div className="tooltip-row">
        <span className="key key-band" />
        <span>80% interval</span>
        <span className="num">
          {formatPrice(p.low)} to {formatPrice(p.high)}
        </span>
      </div>
    </div>
  );
}

export default function FanChart({ points, now }) {
  const dayTicks = points.filter((p) => isStockholmMidnight(p.t)).map((p) => p.t);
  const first = points[0].t;
  const last = points[points.length - 1].t;
  const yTicks = priceTicks(points);

  return (
    <ResponsiveContainer width="100%" height={360}>
      <ComposedChart data={points} margin={{ top: 16, right: 28, bottom: 0, left: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--grid)" />
        <XAxis
          dataKey="t"
          type="number"
          scale="time"
          domain={[first, last]}
          ticks={dayTicks}
          tickFormatter={formatDay}
          tick={axisTick}
          stroke="var(--axis)"
          tickLine={false}
          padding={{ left: 12 }}
        />
        <YAxis
          width={52}
          domain={[yTicks[0], yTicks[yTicks.length - 1]]}
          ticks={yTicks}
          tickFormatter={(v) => formatPrice(v, 2)}
          tick={axisTick}
          stroke="var(--axis)"
          tickLine={false}
          axisLine={false}
        />
        {yTicks[0] < 0 && <ReferenceLine y={0} stroke="var(--axis)" />}
        {now > first && now < last && (
          <ReferenceLine
            x={now}
            stroke="var(--text-muted)"
            label={{ value: "now", position: "insideTopLeft", fill: "var(--text-muted)", fontSize: 12 }}
          />
        )}
        <Area
          dataKey="band"
          stroke="none"
          fill="var(--series-1)"
          fillOpacity={0.16}
          activeDot={false}
          isAnimationActive={false}
        />
        <Line
          dataKey="median"
          stroke="var(--series-1)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2 }}
          isAnimationActive={false}
        />
        <Line
          dataKey="actual"
          stroke="var(--series-2)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2 }}
          connectNulls={false}
          isAnimationActive={false}
        />
        <Tooltip
          content={<ChartTooltip />}
          cursor={{ stroke: "var(--axis)" }}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
