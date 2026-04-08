# Cell Membrane Dynamics Tracker -- Roadmap

## Current State
A PyQt6 GUI application for analyzing PIEZO1 protein distribution and membrane dynamics from TIRF microscopy. Well-structured with `src/analysis/` (5 modules: edge detection, curvature, optical flow, dynamics tracking, correlation), `src/gui/` (3 panels), and `src/utils/` (4 modules). Has `requirements.txt`, dual entry points (`main.py`, `run.py`), and a project design document (`docs/membrane-dynamics-project.md`).

## Short-term Improvements
- [x] Add unit tests for `src/analysis/curvature_analyzer.py` and `src/analysis/optical_flow.py`
- [x] Rename `gitignore.txt` to `.gitignore` and activate it
- [x] Move `.tif` files from project root to a `data/` directory
- [ ] Add input validation in `src/utils/frame_stack.py` for corrupt TIFF files
- [ ] Add type hints to `src/analysis/dynamics_tracker.py`
- [x] Consolidate `main.py` and `run.py` into a single entry point

## Feature Enhancements
- [ ] Add CLI mode for headless batch analysis of multiple time-series datasets
- [ ] Implement automated movement classification thresholds based on data statistics
- [ ] Add heatmap visualization of curvature changes over time in `src/gui/visualization_panel.py`
- [ ] Support exporting kymographs as publication-quality figures
- [ ] Add comparison view for multiple datasets (treated vs control)
- [ ] Implement real-time parameter preview when adjusting edge smoothing or pixel size

## Long-term Vision
- [ ] Add 3D membrane tracking using z-stack data
- [ ] Integrate machine learning for automated movement pattern classification
- [ ] Support multi-cell tracking with identity persistence across frames
- [ ] Create a Jupyter widget version for notebook-based analysis
- [ ] Publish as a pip package with comprehensive API documentation

## Technical Debt
- [x] `membrane-dynamics-project.md` is a design doc that should be in a `docs/` directory
- [ ] `src/gui/dynamics_panel.py` and `src/gui/time_sequence_panel.py` may share plotting code -- extract shared visualization utilities
- [ ] `src/utils/vector_field.py` and `src/analysis/optical_flow.py` may have overlapping vector math -- consolidate
- [ ] No CI/CD pipeline -- add GitHub Actions for testing and linting
- [x] `src/__init__.py` files may be empty -- add subpackage docstrings
- [ ] Border handling (40px exclusion) is hardcoded -- make configurable via settings
