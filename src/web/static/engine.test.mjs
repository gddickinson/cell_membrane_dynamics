// Node test: the JS engines reproduce the repo's Python analysis on tests/fixtures/engine_cases.json
// (written by scripts/make_engine_fixture.py). Run: node src/web/static/engine.test.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { contourMetrics, curvature, curvatureStats, detectEdge, normals } from './contour_engine.js';
import { analyseSequence, kymograph } from './dynamics_engine.js';
import { pearson } from './stats_engine.js';
import { rleDecode } from './rle.js';

const fx = JSON.parse(readFileSync(fileURLToPath(new URL('../../../tests/fixtures/engine_cases.json', import.meta.url))));
let checks = 0, fails = 0;
const e2e = { cases: 0, area: 0, zeros: 0 };
const fail = msg => { fails++; console.error('FAIL', msg); };
function close(a, b, what, rtol = 1e-9, atol = 1e-9) {
  checks++;
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  if (!(Math.abs(a - b) <= atol + rtol * Math.abs(b))) { fail(`${what}: js ${a} vs py ${b}`); return false; }
  return true;
}
function closeArr(a, b, what, rtol, atol) {
  checks++;
  if (a.length !== b.length) return fail(`${what}: length js ${a.length} vs py ${b.length}`);
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (Array.isArray(y)) { for (let k = 0; k < y.length; k++) if (!close(x[k], y[k], `${what}[${i}][${k}]`, rtol, atol)) return; }
    else if (!close(x, y, `${what}[${i}]`, rtol, atol)) return;
  }
}

fx.edges.forEach((c, ci) => {
  const mask = rleDecode(c.rle, c.w * c.h);
  const raw = detectEdge(mask, c.w, c.h, { sigma: 0 }).contour;
  if (c.raw === null) { checks++; if (raw !== null) fail(`edge ${ci}: js found a contour, py none`); return; }
  closeArr(raw, c.raw, `edge ${ci} raw contour`, 0, 0);
  for (const [sigma, key] of [[1.0, "s1.0"], [2.5, "s2.5"]]) {
    const py = c[key], { contour } = detectEdge(mask, c.w, c.h, { sigma });
    const tag = `edge ${ci} sigma ${sigma}`;
    closeArr(contour, py.contour, `${tag} contour`, 1e-12, 1e-10);
    const k = curvature(contour, fx.pixel_nm);
    closeArr(Array.from(k), py.curvature, `${tag} curvature`, 1e-7, 1e-12);
    closeArr(normals(contour), py.normals, `${tag} normals`, 1e-9, 1e-9);
    // metrics and statistics on Python's own contour / curvature: they truncate coordinates and drop exact zeros,
    // so a last-bit difference in the smoothed contour (FMA in SciPy's compiled filter) can move a point across
    // an integer or away from 0; end to end such points are counted below rather than failed
    const m = contourMetrics(py.contour);
    for (const key of ['area', 'perimeter', 'circularity', 'aspect_ratio', 'n_points']) close(m[key], py.metrics[key], `${tag} ${key}`);
    closeArr(m.bounding_box, py.metrics.bounding_box, `${tag} bbox`, 0, 0);
    const s = curvatureStats(py.curvature);
    for (const key of Object.keys(py.stats)) close(s[key], py.stats[key], `${tag} stats.${key}`, 1e-9, 1e-15);
    const me = contourMetrics(contour);
    if (Math.abs(me.area - py.metrics.area) > 0) e2e.area++;
    e2e.zeros += Math.abs(Array.from(k).filter(v => v === 0).length - py.curvature.filter(v => v === 0).length);
    e2e.cases++;
  }
});

const sq = fx.sequence, [h, w] = sq.shape;
const masks = sq.rle.map(r => rleDecode(r, w * h));
const res = analyseSequence(masks, w, h, { pixelSizeNm: sq.pixel_nm, dt: sq.dt, sigma: 1.0 });
checks++;
if (res.frames.length !== sq.frames.length) fail(`sequence: ${res.frames.length} frames vs ${sq.frames.length}`);
let classDiff = 0, classTotal = 0;
sq.frames.forEach((py, i) => {
  const js = res.frames[i];
  close(js.frame, py.frame, `seq frame id ${i}`); close(js.contour.length, py.n, `seq frame ${i} n points`);
  if (!py.velocities) { checks++; if (js.motion) fail(`seq frame ${i}: js has motion, py none`); return; }
  // resampled positions are float32 (as in OpenCV/NumPy); one float32 ulp of a ~100 px coordinate is 7.6e-6 px
  closeArr(js.motion.velocities, py.velocities, `seq frame ${i} velocities`, 0, 2e-5 / sq.dt * 2);
  py.classes.forEach((c, k) => { classTotal++; if (js.motion.classes[k] !== c) classDiff++; });
  close(js.correlation.r, py.r, `seq frame ${i} r`, 1e-4, 1e-6);
  close(js.correlation.p, py.p, `seq frame ${i} p`, 1e-4, 1e-9);
});
checks++;
if (classDiff) fail(`movement classes differ at ${classDiff}/${classTotal} points`);
const ky = kymograph(res.frames, 5, 10);
closeArr(ky.flat(), sq.kymograph_p5_w10.flat(), 'kymograph', 1e-4, 1e-6);

fx.pearson.forEach((c, i) => {
  const { r, p } = pearson(c.x, c.y);
  close(r, c.r, `pearson ${i} r`, 1e-12, 1e-14);
  close(p, c.p, `pearson ${i} p`, 1e-9, 1e-300);
});

console.log(`${checks - fails}/${checks} checks passed (${classTotal} movement classes compared)`);
console.log(`end to end: ${e2e.area}/${e2e.cases} contours differ in integer-truncated area, ${e2e.zeros} curvature points differ in being exactly 0`);
process.exit(fails ? 1 : 0);
