# Cell Membrane Dynamics Tracker -- Interface Map

## Entry Points
- `main.py` -- Main application: `MainWindow` (parameters, frame navigation, `run_analysis`) + `main()`; UI building,
  I/O, frame view and plots come from the mixins in `src/gui/window_*.py`
- `run.py` -- Deprecated convenience wrapper, calls `main.main()`
- `run_web.py` -- Starts the web interface (uvicorn, http://127.0.0.1:8148)

## src/ Package
- `__init__.py` -- Package metadata (version)

### src/analysis/
- `__init__.py` -- Re-exports: EdgeDetector, CurvatureAnalyzer, DynamicsTracker, OpticalFlowAnalyzer, CorrelationAnalyzer, CorrelationData
- `edge_detection.py` -- `EdgeDetector`: cell boundary detection from masks
- `curvature_analyzer.py` -- `CurvatureAnalyzer`: membrane curvature (finite difference, circle fit, spline; positive = convex), `orientation_sign()`
- `optical_flow.py` -- `OpticalFlowAnalyzer`: Farneback dense optical flow analysis
- `dynamics_tracker.py` -- `DynamicsTracker`, `MembraneMotion`: tracks membrane movement over time; `match_previous_points()`
  (arc-length-aligned edge correspondence), `closed_arclength()`, `points_at_arclength()`, `frames_to_uint8()` (8-bit for optical flow)
- `correlation_analyzer.py` -- `CorrelationAnalyzer`, `CorrelationData`: curvature / normal-velocity correlation
- `pipeline.py` -- Qt-free "Run Analysis" loop: `analyze_sequence()` (edge, curvature, dynamics, correlation per frame),
  `finite_difference_curvature()`, `contour_normals()`, `correlation_inputs()`. Used by `main.py` and the web app.

### src/gui/
- `__init__.py` -- Subpackage docstring
- `window_layout.py` -- `LayoutMixin`: `_create_control_panel`, `_create_visualization_panel`
- `window_io.py` -- `IOMixin`: `load_mask_sequence`, `load_fluorescence_sequence`, `_load_image_sequence`, `export_csv`, `export_figures`
- `window_view.py` -- `ViewMixin`: `update_visualization` (display modes)
- `window_plots.py` -- `PlotsMixin`: `update_correlation_plot`, `update_time_series`, `update_kymograph`
- `dynamics_panel.py` -- Dynamics analysis display panel
- `time_sequence_panel.py` -- Temporal sequence visualization panel
- `visualization_panel.py` -- Main visualization panel with curvature and overlay

### src/utils/
- `__init__.py` -- Re-exports data structures, image utils, FrameStack, VectorField
- `data_structures.py` -- Dataclasses: ImageData, EdgeData, CurvatureData, MotionData, FlowData, CorrelationData, AnalysisParameters
- `image_utils.py` -- Image manipulation: normalize, segment, enhance, composite, overlay; `load_tiff_stack()` (all planes of a multi-page TIFF), `load_tiff_sequence()`
- `frame_stack.py` -- `FrameStack`: TIFF stack loading and frame management
- `vector_field.py` -- `VectorField`: vector field operations

### src/web/ (web interface; see README "Web interface")
- `app.py` -- FastAPI app: `/api/info`, `/api/demo`, `/api/demo/python`, `POST /api/analyse`; `STATIC_ENDPOINTS`
- `analysis.py` -- `read_stack()` (TIFF + pixel size / frame interval from metadata), `analyse()` (runs `pipeline.analyze_sequence`, JSON in px, px/s, nm^-1), `summarise()`
- `demo_data.py` -- synthetic demo stack (`demo_masks()`, `demo_fluorescence()`; not real data), `rle_encode/decode`
- `static/` -- front end (no build step): `index.html`, `main.js` (router), pages `home.js`, `analyse.js`, `results.js`,
  `frame_view.js`, `charts.js`, `check.js`, `docs.js` (Methods, About); `units.js` (µm / µm/s / µm^-1 conversion);
  JS ports `contour_engine.js` (EdgeDetector, curvature, normals, metrics), `dynamics_engine.js` (contour tracking,
  classification, correlation, kymograph, `analyseSequence`), `stats_engine.js` (Pearson r and p), `tiff_io.js`
  (TIFF read/write), `rle.js`; `engine_worker.js` / `run_engine.js` (runs the engine in a worker); `static_api.js`
  (static site data layer); node tests `engine.test.mjs`, `tiff_io.test.mjs`; `core.js`, `style.css` (shared kit)

## scripts/
- `build_static.py` -- exports the static GitHub Pages site to `site/`
- `screenshots.py` -- browser smoke test and README screenshots (`docs/img/`)
- `make_engine_fixture.py` -- writes `tests/fixtures/engine_cases.json` (Python results the JS ports must match)
- `make_tiff_fixtures.py` -- writes `tests/fixtures/tiff/` (TIFF layouts for the browser reader test)

## docs/
- `membrane-dynamics-project.md` -- Project design document

## tests/
- `test_analysis.py` -- Tests for CurvatureAnalyzer, OpticalFlowAnalyzer, EdgeDetector, DynamicsTracker, and imports
- `test_regressions.py` -- Regression tests for the 2026-11 bug fixes (small cells, multi-page TIFF, smoothing slider, units, cv2 import)
- `test_fixes_round2.py` -- curvature sign, 16-bit optical flow, edge correspondence, circle fit
- `test_gui_smoke.py` -- desktop app run + every view on the demo stack (Qt offscreen)
- `test_web.py` -- Web API, static build, fixture currency, node tests of the JS ports

## Key Class Relationships
MainWindow instantiates: EdgeDetector, DynamicsTracker, OpticalFlowAnalyzer, CorrelationAnalyzer, CurvatureAnalyzer.
MainWindow.run_analysis and src/web/analysis.py both call src/analysis/pipeline.analyze_sequence.
All analyzers use pixel_size parameter for physical unit conversions.
