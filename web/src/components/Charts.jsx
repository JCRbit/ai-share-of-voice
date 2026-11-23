import { pct } from "../api.js";
import { t } from "../i18n.js";

// The certainty funnel: the client's per-question interval after each round, with the target band.
export function Funnel({ points, target, maxRounds, stop, lang, height = 210 }) {
  const W = 460, H = height, L = 36, R = 12, T = 14, B = 30;
  const n = Math.max(maxRounds || 0, points.length, 2);
  const x = (i) => L + (i * (W - L - R)) / (n - 1);
  const y = (v) => T + (1 - v) * (H - T - B);
  const mid = points.length ? points[points.length - 1].estimate : 0.5;
  const upper = points.map((p, i) => `${x(i)},${y(p.high)}`);
  const lower = points.map((p, i) => `${x(i)},${y(p.low)}`).reverse();
  const area = points.length > 1 ? `M${upper.join(" L")} L${lower.join(" L")} Z` : "";
  const label = points.length
    ? `${t(lang, "margin")} ±${Math.round(points[points.length - 1].half_width * 100)} pts`
    : "";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${t(lang, "funnel")}. ${label}`}>
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="#ece5d8" />
          <text x={L - 6} y={y(v) + 4} fontSize="10" textAnchor="end" fill="#6b6358" fontFamily="IBM Plex Mono">{Math.round(v * 100)}</text>
        </g>
      ))}
      {target != null && (
        <g>
          <rect x={L} width={W - L - R} y={y(Math.min(1, mid + target))} height={Math.max(0, y(Math.max(0, mid - target)) - y(Math.min(1, mid + target)))} fill="#e3efe9" />
          <text x={W - R - 4} y={y(Math.min(1, mid + target)) - 4} fontSize="10" textAnchor="end" fill="#2f5d50" fontFamily="IBM Plex Mono">{t(lang, "target")} ±{Math.round(target * 100)}</text>
        </g>
      )}
      {area && <path d={area} fill="#fde8dc" stroke="#f0b59a" />}
      {points.map((p, i) => (
        <g key={i}>
          <line x1={x(i)} x2={x(i)} y1={y(p.high)} y2={y(p.low)} stroke="#1c1a17" strokeWidth="1.5" />
          <line x1={x(i) - 5} x2={x(i) + 5} y1={y(p.high)} y2={y(p.high)} stroke="#1c1a17" strokeWidth="1.5" />
          <line x1={x(i) - 5} x2={x(i) + 5} y1={y(p.low)} y2={y(p.low)} stroke="#1c1a17" strokeWidth="1.5" />
          <circle className="pop" style={{ animationDelay: `${0.15 + i * 0.12}s` }} cx={x(i)} cy={y(p.estimate)} r="5" fill="#c2410c" />
        </g>
      ))}
      {points.length > 1 && (
        <polyline key={points.length} className="draw" pathLength="1" points={points.map((p, i) => `${x(i)},${y(p.estimate)}`).join(" ")} fill="none" stroke="#c2410c" strokeWidth="2" />
      )}
      {Array.from({ length: n }).map((_, i) => (
        <text key={i} x={x(i)} y={H - 10} fontSize="10" textAnchor={i === n - 1 && n > 1 ? "end" : i === 0 ? "start" : "middle"} fill="#6b6358" fontFamily="IBM Plex Mono">
          {t(lang, "round")} {i + 1}
        </text>
      ))}
      {stop && stop.stop && points.length > 0 && (
        <text x={x(points.length - 1)} y={T + 10} fontSize="11" textAnchor={points.length - 1 > n / 2 ? "end" : "start"} fill="#1c1a17" fontFamily="IBM Plex Sans" fontWeight="600">■ {lang === "es" ? "paré aquí" : "stopped here"}</text>
      )}
    </svg>
  );
}

// Two bars per brand (mentions, recommends) with the per-question interval drawn on each.
export function BrandBars({ brands, lang }) {
  const bar = (m, color) =>
    m ? (
      <div className="track">
        <div className="fill" style={{ width: `${(m.estimate || 0) * 100}%`, background: color }} />
        {m.ci_low != null && (
          <div className="ci" style={{ left: `${m.ci_low * 100}%`, width: `${Math.max(0.5, (m.ci_high - m.ci_low) * 100)}%` }} />
        )}
      </div>
    ) : (
      <div className="track" />
    );
  const verdict = (b) => {
    if (b.is_client) return null;
    const v = b.recommend_vs_client;
    if (!v || v === "unknown") return null;
    return (
      <div className="verdict">
        {t(lang, "recommended").toLowerCase()}: {v === "not_conclusive" ? t(lang, "notConclusive") : t(lang, v)}
      </div>
    );
  };
  return (
    <div className="bars">
      <div className="legend">
        <span><i style={{ background: "#1c1a17" }} />{t(lang, "named")}</span>
        <span><i style={{ background: "#c2410c" }} />{t(lang, "recommended")}</span>
        <span>├─┤ {lang === "es" ? "margen de error (95%, por pregunta)" : "margin of error (95%, per question)"}</span>
      </div>
      {brands.map((b) => (
        <div key={b.name} className={`bar-row ${b.is_client ? "client" : ""}`}>
          <div>
            <strong style={{ fontWeight: b.is_client ? 600 : 400 }}>{b.name}</strong>
            {verdict(b)}
          </div>
          <div>
            {bar(b.mention, b.is_client ? "#1c1a17" : "#8a8174")}
            {bar(b.recommend, b.is_client ? "#c2410c" : "#f0b59a")}
          </div>
          <div className="vals">{pct(b.mention?.estimate)}<br />{pct(b.recommend?.estimate)}</div>
        </div>
      ))}
    </div>
  );
}

// Mention rate across comparable approved runs (same brand, sector, city and model).
export function Trend({ points, lang }) {
  if (!points || points.length < 2) return null;
  const W = 460, H = 160, L = 34, R = 12, T = 12, B = 28;
  const x = (i) => L + (i * (W - L - R)) / (points.length - 1);
  const y = (v) => T + (1 - v) * (H - T - B);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={t(lang, "trend")}>
      {[0, 0.5, 1].map((v) => (
        <line key={v} x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="#ece5d8" />
      ))}
      {points.map((p, i) => (
        <g key={p.run_id}>
          <line x1={x(i)} x2={x(i)} y1={y(p.ci_high)} y2={y(p.ci_low)} stroke="#1c1a17" />
          <circle className="pop" style={{ animationDelay: `${0.15 + i * 0.1}s` }} cx={x(i)} cy={y(p.mention)} r="4" fill="#c2410c" />
          <text x={x(i)} y={H - 8} fontSize="10" textAnchor="middle" fill="#6b6358" fontFamily="IBM Plex Mono">{p.date.slice(5)}</text>
        </g>
      ))}
    </svg>
  );
}
