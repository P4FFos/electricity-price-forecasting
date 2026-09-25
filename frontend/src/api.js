// In development the Vite proxy forwards /api/* to http://localhost:8000/*.
// The production build is served by FastAPI itself, so routes are called directly.
const API_BASE = import.meta.env.DEV ? "/api" : "";

export async function getJson(path, signal) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { signal });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error("Could not reach the forecast API.");
  }

  if (!res.ok) {
    const body = await res.text();
    // The dev proxy answers with an empty 5xx when nothing listens on :8000
    if (!body && res.status >= 500) {
      throw new Error("Could not reach the forecast API. Is it running on localhost:8000?");
    }
    let detail = body;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {
      // plain-text error body, keep as is
    }
    throw new Error(`The API returned an error (HTTP ${res.status}): ${detail}`);
  }

  return res.json();
}
