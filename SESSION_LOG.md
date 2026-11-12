# Session log

## 2026-11-11/12 -- Web interface, static site and bug fixes (Claude, branch `web-interface`)

- Fixed (each with a regression test in `tests/test_regressions.py`, separate commits):
  - `DynamicsTracker`: IndexError for contours shorter than 100 points (small cells); resampled length now capped at
    the current contour's length (results unchanged for >= 100 points).
  - Desktop app loaded only the first page of a multi-page TIFF (cv2.imread); now reads every plane with tifffile
    (`image_utils.load_tiff_stack`); `load_tiff_sequence` also miscounted stacks stored as one series.
  - "Edge Smoothing" slider did not change the edge smoothing (only the optical-flow window); it now sets
    `EdgeDetector.smoothing_sigma` too (default 1.0 unchanged).
  - Plot labels said nm/s for velocities that are in px/s; `create_curvature_motion_map` lacked `import cv2`.
  - README: border margin is 30 px (not 40); intensity correlation not implemented; broken screenshot link replaced.
- Refactor: the Run Analysis loop moved to `src/analysis/pipeline.py` (Qt-free); old and new GUI give identical
  results on a 12-frame synthetic stack.
- Web interface `src/web/` (FastAPI, port 8148) + static GitHub Pages site: Analyse (demo or own TIFF, µm/px and s,
  browser JS engine or Python engine with optical flow), Methods, JS vs Python, About. JS ports node-tested against
  the Python code (`tests/fixtures/engine_cases.json`): raw contours identical; smoothed contours, curvature, velocities,
  classes, r, p, kymograph within float32/rounding tolerances.
- Needs George's decision (not changed): curvature sign (convex gives positive, docs say negative); 16-bit frames
  cast to uint8 with wrap-around before optical flow; index-based edge correspondence between frames; circle-fit
  curvature formula in `CurvatureAnalyzer`.
