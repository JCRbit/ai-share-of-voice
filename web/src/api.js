// Thin client for the FastAPI layer (src/ai_share_of_voice/api/server.py).
const KEY = "asov-demo-password";

export function getPassword() {
  try { return sessionStorage.getItem(KEY) || ""; } catch { return ""; }
}
export function setPassword(value) {
  try { sessionStorage.setItem(KEY, value); } catch { /* private mode: ask again next time */ }
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

async function call(method, path, body) {
  const headers = { "Content-Type": "application/json" };
  const pw = getPassword();
  if (pw) headers["X-Demo-Password"] = pw;
  const res = await fetch(path, { method, headers, body: body ? JSON.stringify(body) : undefined });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, data.detail ?? res.statusText);
  return data;
}

export const api = {
  meta: () => call("GET", "/api/meta"),
  access: () => call("POST", "/api/access"),
  parse: (text, lang) => call("POST", "/api/brief/parse", { text, lang }),
  questions: (brief) => call("POST", "/api/brief/questions", brief),
  start: (brief, questions) => call("POST", "/api/runs", { brief, questions }),
  history: () => call("GET", "/api/runs"),
  run: (id, lang) => call("GET", `/api/runs/${id}?lang=${lang}`),
  decide: (id, approved, comment) => call("POST", `/api/runs/${id}/decision`, { approved, comment }),
  results: (id, lang) => call("GET", `/api/runs/${id}/results?lang=${lang}`),
  answers: (id) => call("GET", `/api/runs/${id}/answers`),
  ask: (id, question, lang) => call("POST", `/api/runs/${id}/ask`, { question, lang }),
  file: (id, name) => `/api/runs/${id}/files/${encodeURIComponent(name)}`,
};

export const pct = (v) => (v == null ? "n/a" : `${Math.round(v * 100)}%`);
export const usd = (v) => `${(v ?? 0).toFixed(2).replace(".", ",")} $`;
export function duration(seconds, lang) {
  if (seconds < 60) return lang === "es" ? "unos segundos" : "a few seconds";
  const m = Math.round(seconds / 60);
  return lang === "es" ? `~${m} min` : `~${m} min`;
}
