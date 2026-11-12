// Membrane dynamics and the per-sequence pipeline: a pure-JS port of src/analysis/dynamics_tracker.py (contour
// tracking and movement classification) and src/analysis/pipeline.py (analyze_sequence), node-tested against Python.
//
// The browser runs the contour-tracking part of DynamicsTracker.analyze_frame (arc-length-aligned matching of
// consecutive edges). The Python tracker also blends in
// Lucas-Kanade optical flow measured on the PIEZO1 channel (70 % flow, 30 % contour wherever flow is found); that
// needs OpenCV and runs only on the local Python server. Without flow the Python result equals this one.
// Units follow the Python code: positions in px, velocities in px/s, curvature in nm^-1.

import { curvature, curvatureStats, detectEdge, contourMetrics, normals, npSum } from './contour_engine.js';
import { pearson } from './stats_engine.js';

export const CLASSES = ['expanding', 'retracting', 'flowing', 'stationary'];
export const STATIC_NM_PER_S = 10;       // below this speed a point is 'stationary' (dynamics_tracker.py)
export const DIRECTION_THRESHOLD = 0.7;  // |cos(angle to normal)| above this is normal (expanding/retracting) motion
export const MAX_OFFSETS = 512;          // candidate start-point offsets when matching consecutive contours

// dynamics_tracker.closed_arclength: cumulative arc length at each point, then the perimeter (n + 1 values)
export function closedArclength(c) {
  const n = c.length, s = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const a = c[i], b = c[(i + 1) % n], dx = b[0] - a[0], dy = b[1] - a[1];
    s[i + 1] = s[i] + Math.sqrt(dx * dx + dy * dy);
  }
  return s;
}

// np.interp(t, s, f) for increasing s (t inside [s[0], s[last]))
function interp(t, s, f) {
  let lo = 0, hi = s.length - 1;
  if (t >= s[hi]) return f[hi];
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (s[mid] <= t) lo = mid; else hi = mid; }
  const slope = (f[lo + 1] - f[lo]) / (s[lo + 1] - s[lo]);
  return slope * (t - s[lo]) + f[lo];
}

// dynamics_tracker.points_at_arclength: points on the closed contour at arc lengths t, wrapped onto [0, P) like np.mod
export function pointsAtArclength(c, s, t) {
  const P = s[s.length - 1], fx = [...c.map(p => p[0]), c[0][0]], fy = [...c.map(p => p[1]), c[0][1]];
  return t.map(v => { let w = v % P; if (w < 0) w += P; return [interp(w, s, fx), interp(w, s, fy)]; });
}

// dynamics_tracker.match_previous_points: arc-length-aligned correspondence with the start-point offset that
// minimises the summed squared distance (best of MAX_OFFSETS candidates, refined by a parabola)
export function matchPrevious(prev, curr) {
  const sp = closedArclength(prev), sc = closedArclength(curr), Pp = sp[sp.length - 1], Pc = sc[sc.length - 1];
  const u = Array.from(sc.subarray(0, curr.length), v => v / Pc * Pp);
  const m = Math.min(prev.length, MAX_OFFSETS), step = Pp / m, cost = new Float64Array(m), sq = new Float64Array(2 * curr.length);
  for (let j = 0; j < m; j++) {
    const q = pointsAtArclength(prev, sp, u.map(v => v + j * step));
    for (let i = 0; i < curr.length; i++) {
      const dx = curr[i][0] - q[i][0], dy = curr[i][1] - q[i][1];
      sq[2 * i] = dx * dx; sq[2 * i + 1] = dy * dy;
    }
    cost[j] = npSum(sq);
  }
  let k = 0;
  for (let j = 1; j < m; j++) if (cost[j] < cost[k]) k = j;
  const e0 = cost[(k - 1 + m) % m], e1 = cost[k], e2 = cost[(k + 1) % m], den = e0 - 2 * e1 + e2;
  const delta = den > 0 ? Math.min(0.5, Math.max(-0.5, 0.5 * (e0 - e2) / den)) : 0;
  return pointsAtArclength(prev, sp, u.map(v => v + (k + delta) * step));
}

// DynamicsTracker._track_contour_movement: one velocity (px/s) per current contour point
export function trackContour(prev, curr, dt) {
  const q = matchPrevious(prev, curr);
  return curr.map((p, i) => [(p[0] - q[i][0]) / dt, (p[1] - q[i][1]) / dt]);
}

const norm = v => Math.sqrt(v[0] * v[0] + v[1] * v[1]);

// DynamicsTracker._classify_movements: velocities (px/s) against the inward normals of the current contour
export function classify(vel, contour, pixelSizeNm) {
  const nrm = normals(contour), thr = STATIC_NM_PER_S / pixelSizeNm;
  return vel.map((v, i) => {
    const m = norm(v);
    if (m < thr) return 'stationary';
    const d = (v[0] / m) * nrm[i][0] + (v[1] / m) * nrm[i][1];
    if (Math.abs(d) > DIRECTION_THRESHOLD) return d > 0 ? 'retracting' : 'expanding';
    return 'flowing';
  });
}

// pipeline.correlation_inputs + CorrelationAnalyzer.analyze_correlation (normal velocity > 0 = inward)
export function correlate(contour, curv, vel, classes) {
  const n = Math.min(curv.length, vel.length), nrm = normals(contour);
  const normal = [], tangential = [];
  for (let i = 0; i < n; i++) {
    const v = vel[i], m = nrm[i];
    normal.push(v[0] * m[0] + v[1] * m[1]);
    tangential.push(v[0] * -m[1] + v[1] * m[0]);
  }
  const k = Array.from(curv).slice(0, n), { r, p } = pearson(k, normal);
  return { curvatures: k, normal, tangential, classes: classes.slice(0, n), r, p };
}

export function motionStats(vel, classes) {
  const mags = vel.map(norm), n = mags.length, s = [...mags].sort((a, b) => a - b);
  const mean = mags.reduce((a, b) => a + b, 0) / n;
  const counts = Object.fromEntries(CLASSES.map(c => [c, classes.filter(x => x === c).length]));
  return { mean_speed: mean, max_speed: s[n - 1], median_speed: n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2,
    std_speed: Math.sqrt(mags.reduce((a, b) => a + (b - mean) ** 2, 0) / n), counts,
    percent: Object.fromEntries(CLASSES.map(c => [c, 100 * counts[c] / n])) };
}

// pipeline.analyze_sequence without optical flow. masks: array of Uint8Array/Float32Array (w x h).
// opts: pixelSizeNm, dt (s), sigma (edge smoothing), start, end (inclusive frame indices).
export function analyseSequence(masks, w, h, { pixelSizeNm = 100, dt = 1, sigma = 1, start = 0, end = null, onFrame = null } = {}) {
  const last = end ?? masks.length - 1, frames = [], failed = [];
  let prev = null;
  for (let i = start; i <= last; i++) {
    if (onFrame) onFrame(i);
    const { contour } = detectEdge(masks[i], w, h, { sigma });
    if (!contour || contour.length < 5) { failed.push(i); continue; }
    const curv = curvature(contour, pixelSizeNm);
    const fr = { frame: i, contour, curvature: curv, metrics: contourMetrics(contour), curvature_stats: curvatureStats(curv),
      motion: null, correlation: null };
    if (prev && i !== start) {
      const vel = trackContour(prev, contour, dt), classes = classify(vel, contour, pixelSizeNm);
      fr.motion = { velocities: vel, magnitudes: vel.map(norm), classes, stats: motionStats(vel, classes) };
      if (curv.length === contour.length) fr.correlation = correlate(contour, curv, vel, classes);
    }
    prev = contour;
    frames.push(fr);
  }
  return { frames, failed };
}

// DynamicsTracker.create_kymograph: rows = frames with motion, columns = contour points pointIndex +- window.
// Each cell is [r, g, b] in 0..1: class colour scaled by min(1, speed / 50 px/s), as in the desktop app.
export function kymograph(frames, pointIndex, window = 10) {
  const rows = [];
  for (const fr of frames) {
    if (!fr.motion) continue;
    // indices wrap over the contour's points, as in Python; velocities may be fewer (then those cells stay black)
    const m = fr.motion, n = fr.contour.length, row = [];
    for (let wv = 0; wv < 2 * window + 1; wv++) {
      const idx = (((pointIndex + wv - window) % n) + n) % n;
      if (pointIndex >= n || idx >= m.magnitudes.length) { row.push([0, 0, 0]); continue; }
      const a = Math.min(1, m.magnitudes[idx] / 50), c = m.classes[idx];
      row.push(c === 'expanding' ? [a, 0, 0] : c === 'retracting' ? [0, 0, a] : c === 'flowing' ? [0, a, 0] : [a, a, a]);
    }
    rows.push(row);
  }
  return rows;
}
