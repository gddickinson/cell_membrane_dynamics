// Cell-edge detection and contour geometry: a pure-JS port of src/analysis/edge_detection.py (EdgeDetector) and the
// finite-difference curvature the desktop app uses (src/analysis/pipeline.py). Node-tested against the Python code
// (contour_engine.test.mjs, fixture from scripts/make_engine_fixture.py).
//
// Conventions: images are row-major (y, x); contour points are [x, y] in pixel units with the origin at the centre of
// the top-left pixel (OpenCV's convention); curvature is returned in nm^-1 like the Python code (the pages convert to
// µm^-1). Float32 arithmetic that OpenCV/NumPy does in single precision is reproduced with Math.fround.

export const BORDER_MARGIN = 30;   // px zeroed around the image before edge detection (edge_detection.py)
export const MIN_SIZE = 100;       // px; smaller objects are removed (remove_small_objects, 4-connectivity)

const f32 = Math.fround;

// Label connected components of a 0/1 Uint8Array (w x h). conn 4 or 8. Returns { labels: Int32Array, sizes: [] }
// where label k (1-based) has sizes[k] pixels; labels are assigned in raster order of each component's first pixel.
export function labelComponents(bin, w, h, conn = 8) {
  const labels = new Int32Array(w * h), sizes = [0], stack = [];
  const nb = conn === 4 ? [[1, 0], [-1, 0], [0, 1], [0, -1]]
    : [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  let next = 0;
  for (let i = 0; i < w * h; i++) {
    if (!bin[i] || labels[i]) continue;
    next++; labels[i] = next; stack.push(i); let n = 0;
    while (stack.length) {
      const p = stack.pop(), x = p % w, y = (p - x) / w; n++;
      for (const [dx, dy] of nb) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const q = yy * w + xx;
        if (bin[q] && !labels[q]) { labels[q] = next; stack.push(q); }
      }
    }
    sizes.push(n);
  }
  return { labels, sizes };
}

// The mask EdgeDetector.detect_edge actually traces: > 0, border margin zeroed, objects < minSize px removed.
export function prepareMask(mask, w, h, { margin = BORDER_MARGIN, minSize = MIN_SIZE } = {}) {
  const out = new Uint8Array(w * h);
  for (let y = margin; y < h - margin; y++) for (let x = margin; x < w - margin; x++) out[y * w + x] = mask[y * w + x] > 0 ? 1 : 0;
  const { labels, sizes } = labelComponents(out, w, h, 4);
  for (let i = 0; i < out.length; i++) if (out[i] && sizes[labels[i]] < minSize) out[i] = 0;
  return out;
}

// OpenCV chain-code directions (x, y), y pointing down
const DX = [1, 1, 0, -1, -1, -1, 0, 1], DY = [0, -1, -1, -1, 0, 1, 1, 1];

// Outer border of the component whose raster-first pixel is (x0, y0): Suzuki-Abe border following as in OpenCV's
// icvFetchContour with CHAIN_APPROX_NONE. Returns [[x, y], ...] starting at (x0, y0).
export function traceOuterBorder(bin, w, h, x0, y0) {
  const val = (x, y) => (x >= 0 && y >= 0 && x < w && y < h ? bin[y * w + x] : 0);
  let s = 4; const sEnd = 4;
  let x1, y1;
  do { s = (s - 1) & 7; x1 = x0 + DX[s]; y1 = y0 + DY[s]; } while (!val(x1, y1) && s !== sEnd);
  if (s === sEnd) return [[x0, y0]];
  const pts = [];
  let x3 = x0, y3 = y0;
  for (let guard = 0; guard < 8 * w * h + 8; guard++) {
    let x4 = x3, y4 = y3;
    while (s < 15) { s++; x4 = x3 + DX[s & 7]; y4 = y3 + DY[s & 7]; if (val(x4, y4)) break; }
    s &= 7;
    pts.push([x3, y3]);
    if (x4 === x0 && y4 === y0 && x3 === x1 && y3 === y1) break;
    x3 = x4; y3 = y4; s = (s + 4) & 7;
  }
  return pts;
}

// cv2.contourArea (absolute shoelace area, px^2)
export function contourArea(pts) {
  let a = 0;
  for (let i = 0, n = pts.length; i < n; i++) {
    const p = pts[i === 0 ? n - 1 : i - 1], q = pts[i];
    a += p[0] * q[1] - p[1] * q[0];
  }
  return Math.abs(a) / 2;
}

// cv2.arcLength(points, closed=True): float32 segment lengths summed in double precision, starting with last -> first
export function arcLength(pts) {
  const n = pts.length; if (n < 2) return 0;
  let per = 0, px = f32(pts[n - 1][0]), py = f32(pts[n - 1][1]);
  for (let i = 0; i < n; i++) {
    const x = f32(pts[i][0]), y = f32(pts[i][1]), dx = f32(x - px), dy = f32(y - py);
    per += f32(Math.sqrt(f32(f32(dx * dx) + f32(dy * dy))));
    px = x; py = y;
  }
  return per;
}

// NumPy's pairwise summation (np.sum of a contiguous float64 array), reproduced so results match bit for bit
export function npSum(a, lo = 0, n = a.length) {
  if (n < 8) { let r = 0; for (let i = lo; i < lo + n; i++) r += a[i]; return r; }
  if (n <= 128) {
    const r = a.slice(lo, lo + 8);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = Math.floor(n / 2); n2 -= n2 % 8;
  return npSum(a, lo, n2) + npSum(a, lo + n2, n - n2);
}

// scipy.ndimage.gaussian_filter1d(points, sigma, axis=0, mode='wrap') (truncate 4.0), with scipy's kernel
// normalisation and its symmetric-kernel summation order (outermost pair first), so results match bit for bit
export function smoothContour(pts, sigma) {
  const n = pts.length;
  if (n < 3) return pts.map(p => [p[0], p[1]]);
  const r = Math.floor(4 * sigma + 0.5), c = -0.5 / (sigma * sigma), phi = [];
  for (let k = -r; k <= r; k++) phi.push(Math.exp(c * (k * k)));
  const sum = npSum(phi), w = phi.map(v => v / sum).slice(r);   // w[0] centre, w[j] = weight at +-j
  const at = i => pts[((i % n) + n) % n];
  return pts.map((p, i) => {
    let x = p[0] * w[0], y = p[1] * w[0];
    for (let j = r; j >= 1; j--) { const a = at(i - j), b = at(i + j); x += (a[0] + b[0]) * w[j]; y += (a[1] + b[1]) * w[j]; }
    return [x, y];
  });
}

// EdgeDetector.detect_edge: largest outer contour of the prepared mask, Gaussian-smoothed when sigma > 0.
// Returns { contour, raw, prepared } (contour null when nothing is found).
export function detectEdge(mask, w, h, { sigma = 1.0, margin = BORDER_MARGIN, minSize = MIN_SIZE } = {}) {
  const prepared = prepareMask(mask, w, h, { margin, minSize });
  const { labels, sizes } = labelComponents(prepared, w, h, 8);
  const seen = new Uint8Array(sizes.length);
  let best = null, bestArea = -1;
  for (let i = 0; i < w * h; i++) {
    const k = labels[i];
    if (!k || seen[k]) continue;
    seen[k] = 1;
    const x = i % w, raw = traceOuterBorder(prepared, w, h, x, (i - x) / w), a = contourArea(raw);
    if (a > bestArea) { best = raw; bestArea = a; }
  }
  if (!best || best.length < 2) return { contour: null, raw: best, prepared };
  return { contour: sigma > 0 ? smoothContour(best, sigma) : best.map(p => [p[0], p[1]]), raw: best, prepared };
}

// Finite-difference curvature (nm^-1; pixelSizeNm = nm per pixel). Sign as in the Python code: -(x'y'' - y'x'')/|r'|^3
export function curvature(pts, pixelSizeNm) {
  const n = pts.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = pts[(i - 1 + n) % n], c = pts[i], q = pts[(i + 1) % n];
    const dx1 = c[0] - p[0], dy1 = c[1] - p[1], dx2 = q[0] - c[0], dy2 = q[1] - c[1];
    const dx = (dx1 + dx2) / 2, dy = (dy1 + dy2) / 2, d2x = dx2 - dx1, d2y = dy2 - dy1;
    const num = dx * d2y - dy * d2x, den = (dx ** 2 + dy ** 2) ** 1.5;
    out[i] = Math.abs(den) > 1e-10 ? -num / den / pixelSizeNm : 0;
  }
  return out;
}

export function centroid(pts) {
  let x = 0, y = 0;
  for (const p of pts) { x += p[0]; y += p[1]; }
  return [x / pts.length, y / pts.length];
}

// Unit normals (central-difference tangent rotated 90 degrees), flipped to point towards the centroid
export function normals(pts) {
  const n = pts.length, [cx, cy] = centroid(pts), out = [];
  for (let i = 0; i < n; i++) {
    const a = pts[(i - 1 + n) % n], b = pts[(i + 1) % n];
    let tx = b[0] - a[0], ty = b[1] - a[1];
    const tn = Math.hypot(tx, ty);
    if (!(tn > 0)) { out.push([0, 0]); continue; }
    tx /= tn; ty /= tn;
    let nx = -ty, ny = tx;
    const ux = cx - pts[i][0], uy = cy - pts[i][1], un = Math.hypot(ux, uy);
    if (un > 0 && nx * (ux / un) + ny * (uy / un) < 0) { nx = -nx; ny = -ny; }
    out.push([nx, ny]);
  }
  return out;
}

// EdgeDetector.analyze_contour (pixel units): computed on the contour truncated to integers, as in the Python code
export function contourMetrics(pts) {
  const ip = pts.map(p => [Math.trunc(p[0]), Math.trunc(p[1])]);
  const area = contourArea(ip), perimeter = arcLength(ip);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of ip) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  return { area, perimeter, circularity: perimeter > 0 ? 4 * Math.PI * area / (perimeter * perimeter) : 0,
    bounding_box: [x0, y0, bw, bh], aspect_ratio: bh > 0 ? bw / bh : 0, n_points: pts.length };
}

// CurvatureAnalyzer.get_curvature_statistics (zeros are treated as failed points and left out)
export function curvatureStats(k) {
  const v = Array.from(k).filter(x => x !== 0), n = v.length;
  if (!n) return { mean: 0, std: 0, min: 0, max: 0, median: 0, positive_percent: 0, negative_percent: 0, n_valid: 0, n_total: k.length };
  const mean = v.reduce((a, b) => a + b, 0) / n, s = [...v].sort((a, b) => a - b);
  return { mean, std: Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / n), min: s[0], max: s[n - 1],
    median: n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2,
    positive_percent: 100 * v.filter(x => x > 0).length / n, negative_percent: 100 * v.filter(x => x < 0).length / n,
    n_valid: n, n_total: k.length };
}
