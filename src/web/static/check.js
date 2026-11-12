// JS vs Python: the browser engine and the repo's Python pipeline (precomputed at build time, /api/demo/python) on
// the synthetic demo stack with the same settings, frame by frame.
import { api, app, route } from './core.js';
import { rleDecode } from './rle.js';
import { runBrowser } from './run_engine.js';
import { CLASSES, f, perUm, pfmt, um2, umPerS } from './units.js';

const rel = (a, b) => (a === b ? 0 : Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-300));

route('check', async () => {
  app().innerHTML = '<h1>JS vs Python</h1><p class="mute">Running the browser engine on the demo stack...</p>';
  const [demo, py] = await Promise.all([api('/demo'), api('/demo/python')]);
  const p = py.params, masks = demo.rle.map(r => rleDecode(r, demo.w * demo.h));
  const t0 = performance.now();
  const js = await runBrowser(masks, demo.w, demo.h, { pixelSizeNm: p.pixel_size_nm, dt: p.frame_s, sigma: p.sigma, start: p.start, end: p.end });
  const secs = (performance.now() - t0) / 1000, u = p.um_per_px;
  let worst = 0;
  const rows = py.frames.map((a, i) => {
    const b = js.frames[i];
    const kAll = Array.from(b.curvature).reduce((x, y) => x + y, 0) / b.curvature.length;
    const diffs = [rel(b.metrics.area, a.metrics.area), rel(kAll, a.curvature_mean_all)];
    if (a.motion) diffs.push(rel(b.motion.stats.mean_speed, a.motion.stats.mean_speed));
    if (a.correlation) diffs.push(rel(b.correlation.r, a.correlation.r));
    const sameClasses = !a.motion || CLASSES.every(c => b.motion.stats.counts[c] === a.motion.stats.counts[c]);
    const d = Math.max(...diffs); worst = Math.max(worst, d);
    return `<tr><td>${a.frame + 1}</td><td>${a.n_points} / ${b.contour.length}</td>
      <td>${f(um2(a.metrics.area, u), 6)} / ${f(um2(b.metrics.area, u), 6)}</td>
      <td>${f(perUm(a.curvature_mean_all), 6)} / ${f(perUm(kAll), 6)}</td>
      <td>${a.motion ? `${f(umPerS(a.motion.stats.mean_speed, u), 6)} / ${f(umPerS(b.motion.stats.mean_speed, u), 6)}` : '–'}</td>
      <td>${a.correlation ? `${f(a.correlation.r, 6)} / ${f(b.correlation.r, 6)}` : '–'}</td>
      <td>${a.correlation ? `${pfmt(a.correlation.p)} / ${pfmt(b.correlation.p)}` : '–'}</td>
      <td class="${sameClasses ? 'ok' : 'bad'}">${sameClasses ? 'identical' : 'differ'}</td><td>${d.toExponential(1)}</td></tr>`;
  }).join('');
  app().innerHTML = `<h1>JS vs Python</h1>
  <p class="lead">The same synthetic demo stack (${demo.n} frames, ${demo.w} × ${demo.h} px; <span class="badge syn">synthetic, not real data</span>)
    through the repo's Python pipeline (computed when this site was built) and through the browser engine just now (${secs.toFixed(2)} s).
    Settings: ${u} µm/px, ${p.frame_s} s per frame, edge smoothing σ ${p.sigma}, no optical flow.</p>
  <div class="scroll"><table class="tbl"><thead><tr><th>frame</th><th>edge points py / js</th><th>area µm²</th><th>mean curvature µm⁻¹ (all points)</th>
    <th>mean speed µm/s</th><th>r</th><th>p</th><th>class counts</th><th>max rel. diff</th></tr></thead><tbody>${rows}</tbody></table></div>
  <p>Largest relative difference over all frames and columns: <b>${worst.toExponential(2)}</b>.</p>
  <p class="small mute">Mean curvature here is over all edge points. The desktop app's curvature statistics (and the tiles on the Analyse page) leave
    out points whose curvature is exactly 0 (three collinear points); whether a nearly straight stretch rounds to exactly 0 can differ between
    the engines, so those statistics agree to about 1–2 % rather than to rounding.</p>
  <div class="panel small"><p>The engines agree to rounding, not bit for bit. OpenCV and NumPy do part of the work in single precision
    (contour resampling, velocities), which the port reproduces with <code>Math.fround</code>; SciPy's compiled Gaussian filter may use fused
    multiply-adds, so smoothed edge coordinates can differ in the last bit. The node test (<code>src/web/static/engine.test.mjs</code>, run by
    <code>pytest tests/test_web.py</code>) compares raw contours exactly and smoothed contours, curvature, normals, metrics, velocities,
    movement classes, correlation, p-values and the kymograph with tight tolerances on 14 masks (random blobs and demo frames)
    and the whole demo sequence.</p></div>`;
});
