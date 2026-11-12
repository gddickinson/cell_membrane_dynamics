// Small SVG charts (no library): line chart with several series and a scatter plot coloured by class.
import { esc } from './core.js';
import { f } from './units.js';

const W = 380, H = 240, M = { l: 52, r: 12, t: 12, b: 40 };
let clipId = 0;

function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}
function scale(d0, d1, r0, r1) {
  const s = v => r0 + (v - d0) / ((d1 - d0) || 1) * (r1 - r0);
  s.domain = [d0, d1];
  return s;
}
function extent(vals, pad = 0.05) {
  const v = vals.filter(Number.isFinite);
  if (!v.length) return [0, 1];
  let lo = Math.min(...v), hi = Math.max(...v);
  if (hi === lo) { lo -= Math.abs(lo) * 0.1 || 0.5; hi += Math.abs(hi) * 0.1 || 0.5; }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}
function axes(xs, ys, xl, yl) {
  const [x0, x1] = xs.domain, [y0, y1] = ys.domain;
  let s = `<line class="axis" x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}"/><line class="axis" x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${H - M.b}"/>`;
  for (const v of ticks(x0, x1)) s += `<text x="${xs(v)}" y="${H - M.b + 14}" text-anchor="middle">${f(v)}</text>`;
  for (const v of ticks(y0, y1, 4)) s += `<text x="${M.l - 5}" y="${ys(v) + 4}" text-anchor="end">${f(v)}</text>`;
  if (y0 < 0 && y1 > 0) s += `<line class="zero" x1="${M.l}" x2="${W - M.r}" y1="${ys(0)}" y2="${ys(0)}"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(xl)}</text>`;
  s += `<text transform="translate(11 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(yl)}</text>`;
  return s;
}

// series: [{ x: [], y: [], color, label, dots }]; mark: optional x for a vertical marker
export function lineChart(series, { xlabel, ylabel, mark = null, title = 'chart' }) {
  const xs = scale(...extent(series.flatMap(s => s.x), 0), M.l, W - M.r);
  const ys = scale(...extent(series.flatMap(s => s.y)), H - M.b, M.t);
  let s = axes(xs, ys, xlabel, ylabel);
  if (mark !== null) s += `<line class="zero" x1="${xs(mark)}" x2="${xs(mark)}" y1="${M.t}" y2="${H - M.b}"/>`;
  for (const se of series) {
    const pts = se.x.map((x, i) => [x, se.y[i]]).filter(p => Number.isFinite(p[1]));
    s += `<polyline class="ln" stroke="${se.color}" points="${pts.map(([x, y]) => `${xs(x).toFixed(1)},${ys(y).toFixed(1)}`).join(' ')}"/>`;
    if (se.dots) for (const [x, y] of pts) s += `<circle cx="${xs(x).toFixed(1)}" cy="${ys(y).toFixed(1)}" r="2.6" fill="${se.color}"><title>${f(x)}, ${f(y)}</title></circle>`;
  }
  const legend = series.length > 1 ? `<div class="legend">${series.map(se => `<span class="cls" style="--c:${se.color}">${esc(se.label)}</span>`).join('')}</div>` : '';
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${s}</svg>${legend}`;
}

// points: [{ x, y, color }]; trend: draw the least-squares line
export function scatter(points, { xlabel, ylabel, trend = true, title = 'scatter plot' }) {
  const xs = scale(...extent(points.map(p => p.x)), M.l, W - M.r), ys = scale(...extent(points.map(p => p.y)), H - M.b, M.t);
  let s = axes(xs, ys, xlabel, ylabel);
  for (const p of points) s += `<circle cx="${xs(p.x).toFixed(1)}" cy="${ys(p.y).toFixed(1)}" r="2.2" fill="${p.color}" opacity=".75"/>`;
  if (trend && points.length > 2) {
    const n = points.length, mx = points.reduce((a, p) => a + p.x, 0) / n, my = points.reduce((a, p) => a + p.y, 0) / n;
    const sxx = points.reduce((a, p) => a + (p.x - mx) ** 2, 0), sxy = points.reduce((a, p) => a + (p.x - mx) * (p.y - my), 0);
    if (sxx > 0) {
      const b = sxy / sxx, [x0, x1] = xs.domain, id = `clip${++clipId}`;
      s += `<clipPath id="${id}"><rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}"/></clipPath>`;
      s += `<line clip-path="url(#${id})" x1="${xs(x0)}" y1="${ys(my + b * (x0 - mx))}" x2="${xs(x1)}" y2="${ys(my + b * (x1 - mx))}" stroke="#ffb02e" stroke-dasharray="5 3" stroke-width="1.5"/>`;
    }
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${s}</svg>`;
}
