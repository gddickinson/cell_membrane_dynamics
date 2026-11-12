// Home: hero, numbers read from the code (/api/info), cards to the pages.
import { api, app, isStatic, route } from './core.js';

route('home', async () => {
  const info = await api('/info'), d = info.defaults;
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">TIRF microscopy · cell edge · curvature · membrane motion</p>
    <h1 class="glow">PIEZO1 Membrane Dynamics Tracker</h1>
    <p class="lead">Follow a cell's edge through a time-lapse of binary cell masks: membrane curvature at every edge point,
      how the edge moves between frames (expanding, retracting, flowing along the edge or stationary) and whether
      curvature and motion are correlated. The desktop app does this for PIEZO1 TIRF recordings; here the same analysis
      runs on your own mask stack, ${isStatic() ? 'in your browser' : 'in your browser or in Python on this computer'}.</p>
    <a class="btn" href="#/analyse">Analyse a mask stack</a><a class="btn alt" href="#/methods">How it works</a><a class="btn alt" href="#/check">JS vs Python</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${d.border_margin_px} px</b><span>image border ignored when finding the edge</span></div>
    <div class="tile"><b>${d.min_size_px} px</b><span>objects smaller than this are dropped</span></div>
    <div class="tile"><b>${d.stationary_nm_per_s} nm/s</b><span>below this an edge point is stationary</span></div>
    <div class="tile"><b>${d.direction_threshold}</b><span>|cos| to the normal above which motion counts as expanding / retracting</span></div>
    <div class="tile"><b>${info.demo.frames}</b><span>frames in the synthetic demo stack</span></div>
  </div>
  ${isStatic() ? `<div class="callout">This is the static site. Your files are read and analysed in this browser by a JavaScript port of the
    repo's Python code (node-tested against it, see <a href="#/check">JS vs Python</a>); nothing is uploaded. The Python pipeline's
    optional optical-flow step (Lucas–Kanade on the PIEZO1 channel, via OpenCV) needs the local app: see <a href="#/about">About</a>.</div>` : ''}
  <h2>Pages</h2>
  <div class="grid">
    <a class="card" href="#/analyse"><b>Analyse</b><span>The synthetic demo or your own mask TIFF (plus an optional PIEZO1 stack as background): edge, curvature, motion, correlation, kymograph, CSV export. Pixel size in µm, frame interval in s.</span></a>
    <a class="card" href="#/methods"><b>Methods</b><span>Each step of the pipeline as the code does it, with its constants, units and known caveats.</span></a>
    <a class="card" href="#/check"><b>JS vs Python</b><span>The browser engine and the repo's Python pipeline on the same demo stack, frame by frame.</span></a>
    <a class="card" href="#/about"><b>About</b><span>What the project is, how to run the desktop app and the web app, and credits.</span></a>
  </div>`;
});
