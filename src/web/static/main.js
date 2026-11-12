// Entry point: lazy page modules, the mode label, a subtle "breathing membrane" background, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = { 'home.js': ['home'], 'analyse.js': ['analyse'], 'docs.js': ['methods', 'about'], 'check.js': ['check'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: the analysis runs in your browser (JS port of the Python code); nothing is uploaded'
  : 'local app: browser engine or the Python pipeline on this computer';

// Background: one faint closed outline whose radius wobbles slowly. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 50) return;
    last = t;
    const W = canvas.width, H = canvas.height, R = Math.min(W, H) * 0.32, s = t / 4000;
    g.clearRect(0, 0, W, H);
    g.lineWidth = 2;
    for (const [k, col] of [[0, 'rgba(46,242,196,.10)'], [1, 'rgba(255,79,216,.07)']]) {
      g.strokeStyle = col; g.beginPath();
      for (let i = 0; i <= 180; i++) {
        const a = i / 180 * 2 * Math.PI;
        const r = R * (1 + 0.06 * Math.sin(3 * a - s - k) + 0.04 * Math.sin(5 * a + 0.7 * s) + 0.02 * k);
        const x = W * 0.72 + r * Math.cos(a), y = H * 0.55 + r * Math.sin(a);
        if (i) g.lineTo(x, y); else g.moveTo(x, y);
      }
      g.stroke();
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
