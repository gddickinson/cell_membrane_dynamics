// Membrane dynamics and the per-sequence pipeline: a pure-JS port of src/analysis/dynamics_tracker.py (contour
// tracking and movement classification) and src/analysis/pipeline.py (analyze_sequence), node-tested against Python.
//
// The browser runs the contour-tracking part of DynamicsTracker.analyze_frame. The Python tracker also blends in
// Lucas-Kanade optical flow measured on the PIEZO1 channel (70 % flow, 30 % contour wherever flow is found); that
// needs OpenCV and runs only on the local Python server. Without flow the Python result equals this one.
// Units follow the Python code: positions in px, velocities in px/s, curvature in nm^-1.

import { arcLength, curvature, curvatureStats, detectEdge, contourMetrics, normals } from './contour_engine.js';
import { pearson } from './stats_engine.js';

const f32 = Math.fround;
export const CLASSES = ['expanding', 'retracting', 'flowing', 'stationary'];
export const STATIC_NM_PER_S = 10;       // below this speed a point is 'stationary' (dynamics_tracker.py)
export const DIRECTION_THRESHOLD = 0.7;  // |cos(angle to normal)| above this is normal (expanding/retracting) motion

// DynamicsTracker._find_point_at_distance
function pointAtDistance(c, dist) {
  let cur = 0;
  for (let i = 0; i < c.length; i++) {
    const a = c[i], b = c[(i + 1) % c.length], seg = Math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2);
    if (cur + seg >= dist) return [i, (dist - cur) / seg];
    cur += seg;
  }
  return [c.length - 1, 0];
}

// DynamicsTracker._resample_contour: n points at equal arc-length steps from the first point (float32 output)
export function resample(c, n) {
  const per = arcLength(c), out = [];
  for (let i = 0; i < n; i++) {
    const [k, f] = pointAtDistance(c, i * per / n), a = c[k], b = c[(k + 1) % c.length];
    out.push([f32(a[0] * (1 - f) + b[0] * f), f32(a[1] * (1 - f) + b[1] * f)]);
  }
  return out;
}

// Contour-tracking velocities (px/s) between consecutive contours, paired by index after resampling both to
// min(len(prev), len(curr)) points (at least 100, never more than the current contour has).
export function trackContour(prev, curr, dt) {
  const n = Math.min(Math.max(Math.min(prev.length, curr.length), 100), curr.length);
  const a = resample(prev, n), b = resample(curr, n), d = f32(dt);
  return b.map((p, i) => [f32(f32(p[0] - a[i][0]) / d), f32(f32(p[1] - a[i][1]) / d)]);
}

const norm32 = v => f32(Math.sqrt(f32(f32(v[0] * v[0]) + f32(v[1] * v[1]))));

// DynamicsTracker._classify_movements: velocities (px/s) against the inward normals of the current contour
export function classify(vel, contour, pixelSizeNm) {
  const nrm = normals(contour), thr = f32(STATIC_NM_PER_S / pixelSizeNm);
  return vel.map((v, i) => {
    const m = norm32(v);
    if (m < thr) return 'stationary';
    const d = f32(v[0] / m) * nrm[i][0] + f32(v[1] / m) * nrm[i][1];
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
  const mags = vel.map(norm32), n = mags.length, s = [...mags].sort((a, b) => a - b);
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
      fr.motion = { velocities: vel, magnitudes: vel.map(norm32), classes, stats: motionStats(vel, classes) };
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
