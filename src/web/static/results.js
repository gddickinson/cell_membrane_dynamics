// Results of one analysis run: frame viewer (image + coloured edge + velocity arrows), per-frame numbers, charts,
// kymograph and CSV export. Works on the JS engine's result and the Python API's result (same keys and units:
// px, px/s, nm^-1); everything shown is converted to µm, µm/s and µm^-1 with result.params.um_per_px.
import { $, download, esc } from './core.js';
import { kymograph } from './dynamics_engine.js';
import { lineChart, scatter } from './charts.js';
import { arcPositions, CLASS_COLORS, CLASSES, f, perUm, pfmt, um, um2, umPerS } from './units.js';
import { drawFrame } from './frame_view.js';

export function renderResults(host, res, data, info) {
  const p = res.params, u = p.um_per_px, frames = res.frames;
  if (!frames.length) { host.innerHTML = `<p class="err">No cell edge was found in any frame (frames ${p.start + 1}–${p.end + 1}). Check that the masks are binary and the cell lies more than ${info.defaults.border_margin_px} px from the image border.</p>`; return; }
  const engine = res.engine === 'python' ? `<span class="badge py">Python pipeline${res.optical_flow ? ' + optical flow' : ', contour tracking only'}</span>` : '<span class="badge js">browser (JS port)</span>';
  host.innerHTML = `
  <h2>Results ${engine} ${data.synthetic ? '<span class="badge syn">synthetic demo data</span>' : ''}</h2>
  <dl class="kv panel">
    <dt>Data</dt><dd>${esc(data.label)}</dd>
    <dt>Pixel size</dt><dd>${u} µm/px (${p.pixel_size_nm.toFixed(1)} nm), ${esc(p.scale_source || '')}</dd>
    <dt>Frame interval</dt><dd>${p.frame_s} s</dd>
    <dt>Edge smoothing σ</dt><dd>${p.sigma} contour points</dd>
    <dt>Frames</dt><dd>${p.start + 1}–${p.end + 1} of ${p.n_frames}: edge found in ${frames.length}${res.failed.length ? `; none in ${res.failed.map(i => i + 1).join(', ')}` : ''}</dd>
  </dl>
  <div class="framebar"><button type="button" id="fprev" aria-label="Previous frame">◀</button>
    <label for="fsl" class="small">Frame <b id="flab"></b></label><input type="range" id="fsl" min="0" max="${frames.length - 1}" value="${Math.min(1, frames.length - 1)}">
    <button type="button" id="fnext" aria-label="Next frame">▶</button></div>
  <div class="view">
    <div>
      <div class="imgwrap"><canvas id="fc" role="img" aria-label="Frame with the detected cell edge"></canvas></div>
      <div class="framebar small">
        <label>Colour edge by <select id="cby"><option value="curv">curvature</option><option value="cls">movement class</option></select></label>
        <label><input type="checkbox" id="arr" checked> velocity arrows</label>
        <label>arrow = motion over <input type="number" id="asc" value="${Math.max(1, Math.round(3 * p.frame_s))}" min="0.1" step="any" style="width:5em"> s</label>
      </div>
      <div class="legend" id="leg"></div>
    </div>
    <div id="fstats"></div>
  </div>
  <div class="charts" id="charts"></div>
  <h2>Over time</h2>
  <div class="charts" id="tcharts"></div>
  <h2>Kymograph</h2>
  <div class="panel kymo">
    <div class="framebar small"><label>Contour point <input type="number" id="kp" min="0" value="0" style="width:6em"></label>
      <label>Window ± <input type="number" id="kw" min="1" max="200" value="10" style="width:5em"> points</label></div>
    <canvas id="kc" role="img" aria-label="Kymograph: movement class and speed around one contour point, frame by frame"></canvas>
    <p class="small mute">Rows are frames with motion (top = first), columns are contour points around the chosen one. Colour = movement class
      (red expanding, blue retracting, green flowing, grey stationary), brightness = speed / ${info.defaults.kymograph_full_scale_px_per_s} px/s
      (${f(info.defaults.kymograph_full_scale_px_per_s * u)} µm/s at this pixel size), as in the desktop app.</p>
  </div>
  <p><button type="button" id="csv">Download CSV</button> <button type="button" id="json" class="alt">Download JSON</button></p>`;
  const sl = $('#fsl', host), redraw = () => frameView(host, res, data, +sl.value);
  sl.addEventListener('input', redraw);
  $('#fprev', host).onclick = () => { sl.value = Math.max(0, +sl.value - 1); redraw(); };
  $('#fnext', host).onclick = () => { sl.value = Math.min(frames.length - 1, +sl.value + 1); redraw(); };
  for (const id of ['#cby', '#arr', '#asc']) $(id, host).addEventListener('input', redraw);
  redraw();
  timeCharts(host, res);
  const kdraw = () => drawKymo($('#kc', host), frames, +$('#kp', host).value, +$('#kw', host).value);
  $('#kp', host).addEventListener('input', kdraw); $('#kw', host).addEventListener('input', kdraw);
  kdraw();
  $('#csv', host).onclick = () => download(`${(data.label || 'cell').replace(/\.[^.]*$/, '').replace(/[^\w.-]+/g, '_')}_membrane.csv`, toCsv(res, data), 'text/csv');
  $('#json', host).onclick = () => download('membrane_dynamics.json', JSON.stringify(res), 'application/json');
}

function frameView(host, res, data, k) {
  const fr = res.frames[k], u = res.params.um_per_px;
  $('#flab', host).textContent = `${fr.frame + 1} (${k + 1} of ${res.frames.length} analysed)`;
  const by = $('#cby', host).value, kmax = drawFrame($('#fc', host), data, fr, { by, arrows: $('#arr', host).checked, seconds: +$('#asc', host).value });
  $('#leg', host).innerHTML = by === 'cls'
    ? CLASSES.map(c => `<span class="cls" style="--c:${CLASS_COLORS[c]}">${c}</span>`).join('') + (fr.motion ? '' : '<span class="small mute">first frame: no motion yet</span>')
    : `<span class="small mono">${f(-perUm(kmax))} µm⁻¹ <span class="ramp"></span> ${f(perUm(kmax))} µm⁻¹</span>`;
  const m = fr.metrics, cs = fr.curvature_stats, mo = fr.motion, co = fr.correlation;
  $('#fstats', host).innerHTML = `<div class="tiles small">
    <div class="tile"><b>${f(um2(m.area, u), 4)}</b><span>area (µm²)</span></div>
    <div class="tile"><b>${f(um(m.perimeter, u), 4)}</b><span>perimeter (µm)</span></div>
    <div class="tile"><b>${f(m.circularity)}</b><span>circularity</span></div>
    <div class="tile"><b>${f(perUm(cs.mean))}</b><span>mean curvature (µm⁻¹)</span></div>
    <div class="tile"><b>${mo ? f(umPerS(mo.stats.mean_speed, u)) : '–'}</b><span>mean edge speed (µm/s)</span></div>
    <div class="tile"><b>${co ? f(co.r) : '–'}</b><span>r, curvature vs normal velocity (p ${co ? pfmt(co.p) : '–'})</span></div>
  </div>
  <p class="small mute">${m.n_points} edge points · bounding box ${f(um(m.bounding_box[2], u))} × ${f(um(m.bounding_box[3], u))} µm ·
    curvature &gt; 0 at ${f(cs.positive_percent)} % of points${mo ? ` · ${CLASSES.map(c => `${c} ${f(mo.stats.percent[c])} %`).join(', ')}` : ''}</p>`;
  const arc = arcPositions(fr.contour).map(s => um(s, u));
  let html = `<div class="panel"><h3>Curvature along the edge</h3>${lineChart([{ x: arc, y: Array.from(fr.curvature, perUm), color: '#2ef2c4', label: 'curvature' }],
    { xlabel: 'distance along edge from the first point (µm)', ylabel: 'curvature (µm⁻¹)', title: 'Curvature along the edge' })}</div>`;
  if (co) {
    const pts = co.curvatures.map((c, i) => ({ x: perUm(c), y: umPerS(co.normal[i], u), color: CLASS_COLORS[co.classes[i]] || '#999' }));
    html += `<div class="panel"><h3>Curvature vs normal velocity</h3>${scatter(pts, { xlabel: 'curvature (µm⁻¹)', ylabel: 'normal velocity, + inward (µm/s)', title: 'Curvature vs normal velocity' })}
      <p class="small mute">r = ${f(co.r)}, p = ${pfmt(co.p)} (Pearson, ${co.curvatures.length} points); dashed line: least squares.</p></div>`;
  } else html += '<div class="panel"><h3>Curvature vs normal velocity</h3><p class="mute small">Motion needs a previous frame: pick a later frame.</p></div>';
  $('#charts', host).innerHTML = html;
}

function timeCharts(host, res) {
  const u = res.params.um_per_px, fs = res.frames.filter(fr => fr.correlation), x = fs.map(fr => fr.frame + 1);
  if (fs.length < 1) { $('#tcharts', host).innerHTML = '<p class="mute small">Needs at least two analysed frames.</p>'; return; }
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  $('#tcharts', host).innerHTML = `
    <div class="panel"><h3>Curvature–motion correlation</h3>${lineChart([{ x, y: fs.map(fr => fr.correlation.r ?? NaN), color: '#2ef2c4', label: 'r', dots: true }], { xlabel: 'frame', ylabel: 'Pearson r', title: 'Correlation per frame' })}</div>
    <div class="panel"><h3>Mean curvature</h3>${lineChart([{ x, y: fs.map(fr => perUm(mean(fr.correlation.curvatures))), color: '#ff4fd8', label: 'curvature', dots: true }], { xlabel: 'frame', ylabel: 'mean curvature (µm⁻¹)', title: 'Mean curvature per frame' })}</div>
    <div class="panel"><h3>Mean |normal velocity|</h3>${lineChart([{ x, y: fs.map(fr => umPerS(mean(fr.correlation.normal.map(Math.abs)), u)), color: '#ffb02e', label: 'speed', dots: true }], { xlabel: 'frame', ylabel: 'mean |normal velocity| (µm/s)', title: 'Mean normal speed per frame' })}</div>
    <div class="panel"><h3>Movement classes</h3>${lineChart(CLASSES.map(c => ({ x, y: fs.map(fr => 100 * fr.correlation.classes.filter(k => k === c).length / fr.correlation.classes.length), color: CLASS_COLORS[c], label: c, dots: true })), { xlabel: 'frame', ylabel: 'edge points (%)', title: 'Movement classes per frame' })}</div>`;
}

function drawKymo(canvas, frames, point, win) {
  const rows = kymograph(frames, Math.max(0, point | 0), Math.max(1, Math.min(200, win | 0)));
  const W = 2 * Math.max(1, Math.min(200, win | 0)) + 1, H = Math.max(1, rows.length), sx = Math.max(1, Math.floor(600 / W)), sy = Math.max(4, Math.floor(240 / H));
  canvas.width = W * sx; canvas.height = H * sy;
  const g = canvas.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, canvas.width, canvas.height);
  rows.forEach((row, t) => row.forEach(([r, gg, b], w) => { g.fillStyle = `rgb(${r * 255 | 0},${gg * 255 | 0},${b * 255 | 0})`; g.fillRect(w * sx, t * sy, sx, sy); }));
  g.strokeStyle = 'rgba(255,255,255,.5)'; g.strokeRect(((W - 1) / 2) * sx + 0.5, 0.5, sx - 1, canvas.height - 1);
}

function toCsv(res, data) {
  const p = res.params, u = p.um_per_px;
  const head = [`# PIEZO1 Membrane Dynamics Tracker web app; engine: ${res.engine}${res.optical_flow ? ' + optical flow' : ''}`,
    `# data: ${data.label}${data.synthetic ? ' (synthetic demo, not real data)' : ''}`,
    `# pixel_size_um_per_px: ${u} (${p.scale_source || ''}); frame_interval_s: ${p.frame_s}; edge_smoothing_sigma: ${p.sigma}`,
    '# frame numbers start at 1; x/y: pixel-centre coordinates (OpenCV convention); curvature sign as in the repo code; normal velocity > 0 = inward',
    'frame,point_idx,x_px,y_px,x_um,y_um,curvature_per_um,velocity_x_um_s,velocity_y_um_s,speed_um_s,normal_velocity_um_s,classification'];
  const rows = [];
  for (const fr of res.frames) {
    fr.contour.forEach((pt, i) => {
      const v = fr.motion && i < fr.motion.velocities.length ? fr.motion.velocities[i] : null;
      const nv = fr.correlation && i < fr.correlation.normal.length ? umPerS(fr.correlation.normal[i], u) : '';
      rows.push([fr.frame + 1, i, pt[0], pt[1], um(pt[0], u), um(pt[1], u), perUm(fr.curvature[i]),
        v ? umPerS(v[0], u) : '', v ? umPerS(v[1], u) : '', v ? umPerS(fr.motion.magnitudes[i], u) : '', nv,
        v ? fr.motion.classes[i] : ''].join(','));
    });
  }
  return head.concat(rows).join('\n') + '\n';
}
