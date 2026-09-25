// Bidding zones are Swedish, so times are shown in Stockholm time regardless of
// the viewer's own time zone. The API may send any offset (it currently sends UTC).
const TZ = "Europe/Stockholm";

const dayFmt = new Intl.DateTimeFormat("en-GB", {
  timeZone: TZ,
  weekday: "short",
  day: "numeric",
  month: "short",
});

const dateTimeFmt = new Intl.DateTimeFormat("en-GB", {
  timeZone: TZ,
  weekday: "short",
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

const hourFmt = new Intl.DateTimeFormat("en-GB", {
  timeZone: TZ,
  hour: "2-digit",
  hourCycle: "h23",
});

export const formatDay = (ms) => dayFmt.format(ms);
export const formatDateTime = (ms) => dateTimeFmt.format(ms);
export const isStockholmMidnight = (ms) => hourFmt.format(ms) === "00";

export function formatPrice(value, digits = 4) {
  if (value == null) return "–";
  const text = value.toFixed(digits);
  // Tiny negatives round to "-0.0000"; show those as plain zero
  if (Number(text) === 0) return (0).toFixed(digits);
  return text.replace("-", "−");
}

export function formatPercent(value) {
  if (value == null) return "–";
  return `${(value * 100).toFixed(1)}%`;
}
