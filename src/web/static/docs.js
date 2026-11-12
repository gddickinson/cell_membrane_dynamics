// Methods (the pipeline as the code does it, with constants from /api/info) and About.
import { api, app, route } from './core.js';

route('methods', async () => {
  const d = (await api('/info')).defaults;
  app().innerHTML = `<h1>Methods</h1>
  <p class="lead">What happens to each frame, in the order of <code>src/analysis/pipeline.py</code> (the loop behind the desktop app's
    "Run Analysis"). The browser engine is a JavaScript port of these steps; numbers in the code's own units (px, px/s, nm⁻¹) are converted
    to µm, µm/s and µm⁻¹ with the pixel size you give.</p>
  <div class="panel"><h2>1 · Cell edge</h2>
    <p><code>EdgeDetector.detect_edge</code>: any mask value &gt; 0 is cell; a ${d.border_margin_px}-px frame around the image is cleared (cells touching
      the border are cut there); objects under ${d.min_size_px} px (4-connected) are removed; OpenCV's outer-contour tracing (every boundary
      pixel, 8-connected) gives one contour per object and the largest by enclosed area is kept. Its points are smoothed with a periodic
      Gaussian of σ contour points (default ${d.sigma}; 0 = no smoothing). Points are pixel centres, origin at the centre of the top-left
      pixel (OpenCV convention).</p></div>
  <div class="panel"><h2>2 · Curvature</h2>
    <p>Finite differences over each point and its two neighbours: κ = −(x′y″ − y′x″) / (x′² + y′²)^{3/2}, divided by the pixel size, so in
      nm⁻¹ (shown here ×1000 as µm⁻¹). A circle of radius R µm gives κ ≈ 1/R µm⁻¹. Statistics leave out points where κ is exactly 0.</p>
    <p><b>Sign.</b> Positive where the membrane is convex (bulging outward: a round cell is positive everywhere), negative where it is
      concave (indented towards the cell interior), whatever the order of the contour points.</p></div>
  <div class="panel"><h2>3 · Edge motion</h2>
    <p><code>DynamicsTracker</code>: each point of the current edge is matched to the previous edge at the same fraction of the perimeter,
      shifted by the start-point offset that minimises the summed squared distance between matched points (best of up to 512 evenly spaced
      offsets, refined by a parabola). Velocity = (current point − matched point) / frame interval, one per current edge point. The match does
      not depend on where each edge starts, so a still or uniformly expanding outline shows no tangential motion. (Before 2026-11-12 the
      code paired point i with point i after resampling both edges from their own start points, which made a large share of points
      "flowing"; on the demo stack the flowing fraction drops from 66 % to 19 % with the new matching.)</p>
    <p>The Python pipeline then measures Lucas–Kanade optical flow (OpenCV) on the PIEZO1 frames at the edge points and, wherever flow is
      found, uses ${d.flow_weight} × flow + ${d.contour_weight} × contour velocity. That step needs OpenCV and runs only in the local app; without a
      PIEZO1 stack the result equals the browser's. 16-bit frames are scaled to 8 bit by the pair's joint intensity range before the flow is measured.</p>
    <p>Classification against the inward normal (perpendicular to the edge, flipped towards the edge's centroid): slower than
      ${d.stationary_nm_per_s} nm/s → <span class="cls" style="--c:var(--c-stat)">stationary</span>; |cos(angle to normal)| &gt; ${d.direction_threshold}
      → <span class="cls" style="--c:var(--c-exp)">expanding</span> (outward) or <span class="cls" style="--c:var(--c-ret)">retracting</span>
      (inward); otherwise <span class="cls" style="--c:var(--c-flow)">flowing</span> along the edge.</p></div>
  <div class="panel"><h2>4 · Curvature vs motion</h2>
    <p>Velocity split into normal (positive = inward) and tangential components; Pearson r between curvature and normal velocity at each
      edge point, with SciPy's two-sided p-value. Points along an edge are not independent, so p is optimistic.</p></div>
  <div class="panel"><h2>5 · Kymograph</h2>
    <p><code>DynamicsTracker.create_kymograph</code>: for one contour point index and ± a window of neighbours, each frame's movement class as
      colour and speed / ${d.kymograph_full_scale_px_per_s} px/s as brightness. The point index is not tracked between frames: index k is the k-th
      point from each frame's start point.</p></div>
  <div class="panel"><h2>Not in the code yet</h2>
    <p>The README also lists PIEZO1 intensity along the edge and its correlation with curvature and movement class. The current analysis
      code does not compute these; the PIEZO1 stack is used for optical flow (Python) and as the background image here.</p></div>`;
});

route('about', () => {
  app().innerHTML = `<h1>About</h1>
  <p class="lead">PIEZO1 Membrane Dynamics Tracker is a Python application for analysing PIEZO1 protein distribution and membrane dynamics
    in TIRF microscopy data: a PyQt6 desktop app that loads a binary cell-mask sequence and a PIEZO1 fluorescence sequence and maps
    membrane movement, curvature and their relationship along the cell edge, with correlation plots, time series and kymographs.</p>
  <div class="two">
    <div class="panel"><h2>Run the desktop app</h2>
      <pre><code>git clone https://github.com/gddickinson/cell_membrane_dynamics
cd cell_membrane_dynamics
pip install -r requirements.txt
python main.py</code></pre>
      <p class="small">Load the cell-mask sequence and the PIEZO1 sequence (multi-page TIFF or numbered single TIFFs), set the pixel size
        (nm) and frame interval (s), run the analysis and export CSV or figures.</p></div>
    <div class="panel"><h2>Run the web app</h2>
      <pre><code>pip install -r requirements.txt -r requirements-web.txt
python run_web.py      # http://127.0.0.1:8148</code></pre>
      <p class="small">The local web app adds the Python engine (the repo's own pipeline), including Lucas–Kanade optical flow on the
        PIEZO1 stack and compressed TIFFs. The static site (GitHub Pages) runs the JavaScript port only.</p></div>
  </div>
  <div class="panel"><h2>Credits</h2>
    <p>Developed for analysing PIEZO1 mechanosensitive ion channel distribution in relation to membrane dynamics and curvature in TIRF
      microscopy data. © 2025 George Dickinson, MIT licence. Built with AI assistance from Claude (Anthropic). The demo stack on this site
      is synthetic (<code>src/web/demo_data.py</code>), not a recording.</p></div>`;
});
