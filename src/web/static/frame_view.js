// Draw one analysed frame: the PIEZO1 image (if loaded, auto-contrast 0.5-99.5 %) or the mask as background, the
// detected edge coloured by curvature (diverging, symmetric about 0) or movement class, and velocity arrows.
import { CLASS_COLORS } from './units.js';

function background(data, fi) {
  const n = data.w * data.h, img = new Uint8ClampedArray(n * 4);
  if (data.fluor) {
    const v = data.fluor[fi], s = Array.from(v).filter((_, i) => i % 7 === 0).sort((a, b) => a - b);
    const lo = s[Math.floor(s.length * 0.005)], hi = s[Math.floor(s.length * 0.995)] || lo + 1;
    for (let i = 0; i < n; i++) { const g = Math.max(0, Math.min(255, 255 * (v[i] - lo) / ((hi - lo) || 1))); img.set([g, g, g, 255], i * 4); }
  } else {
    const m = data.masks[fi];
    for (let i = 0; i < n; i++) { const g = m[i] > 0 ? 52 : 8; img.set([g, g, g + 6, 255], i * 4); }
  }
  return new ImageData(img, data.w, data.h);
}

// diverging colour for t in [-1, 1]: cyan (negative) - dark grey - magenta (positive)
function diverge(t) {
  const a = Math.min(1, Math.abs(t)), base = [58, 65, 82], end = t < 0 ? [46, 230, 255] : [255, 79, 216];
  return `rgb(${base.map((b, i) => Math.round(b + (end[i] - b) * a)).join(',')})`;
}

export function drawFrame(canvas, data, fr, { by = 'curv', arrows = true, seconds = 3 } = {}) {
  const sc = Math.max(1, Math.min(4, Math.floor(900 / Math.max(data.w, data.h))));
  canvas.width = data.w * sc; canvas.height = data.h * sc;
  const g = canvas.getContext('2d'), tmp = document.createElement('canvas');
  tmp.width = data.w; tmp.height = data.h;
  tmp.getContext('2d').putImageData(background(data, fr.frame), 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  const P = ([x, y]) => [(x + 0.5) * sc, (y + 0.5) * sc];   // pixel-centre coordinates -> canvas
  const c = fr.contour, n = c.length, k = fr.curvature;
  const sorted = Array.from(k, Math.abs).sort((a, b) => a - b), kmax = sorted[Math.floor(0.98 * (n - 1))] || 1e-12;
  g.lineWidth = Math.max(2, sc * 0.9); g.lineCap = 'round';
  for (let i = 0; i < n; i++) {
    const a = P(c[i]), b = P(c[(i + 1) % n]);
    g.strokeStyle = by === 'cls'
      ? (fr.motion && i < fr.motion.classes.length ? CLASS_COLORS[fr.motion.classes[i]] : '#5b6478')
      : diverge(k[i] / kmax);
    g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
  }
  if (arrows && fr.motion) {
    const v = fr.motion.velocities, step = Math.max(1, Math.round(v.length / 90));
    g.lineWidth = Math.max(1, sc * 0.5);
    for (let i = 0; i < v.length; i += step) {
      const [x, y] = P(c[i]), dx = v[i][0] * seconds * sc, dy = v[i][1] * seconds * sc, L = Math.hypot(dx, dy);
      if (L < 0.5) continue;
      g.strokeStyle = CLASS_COLORS[fr.motion.classes[i]] || '#ccc';
      const ex = x + dx, ey = y + dy, h = Math.min(6, L * 0.35), ang = Math.atan2(dy, dx);
      g.beginPath(); g.moveTo(x, y); g.lineTo(ex, ey);
      g.lineTo(ex - h * Math.cos(ang - 0.45), ey - h * Math.sin(ang - 0.45));
      g.moveTo(ex, ey); g.lineTo(ex - h * Math.cos(ang + 0.45), ey - h * Math.sin(ang + 0.45));
      g.stroke();
    }
  }
  return kmax;
}
