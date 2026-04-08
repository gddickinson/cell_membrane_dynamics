# Cell Membrane Dynamics Tracker -- Interface Map

## Entry Points
- `main.py` -- Main application: `MainWindow(QMainWindow)` GUI + `main()` function
- `run.py` -- Deprecated convenience wrapper, calls `main.main()`

## src/ Package
- `__init__.py` -- Package metadata (version)

### src/analysis/
- `__init__.py` -- Re-exports: EdgeDetector, CurvatureAnalyzer, DynamicsTracker, OpticalFlowAnalyzer, CorrelationAnalyzer, CorrelationData
- `edge_detection.py` -- `EdgeDetector`: cell boundary detection from masks
- `curvature_analyzer.py` -- `CurvatureAnalyzer`: membrane curvature calculation (finite difference, circle fit, spline methods)
- `optical_flow.py` -- `OpticalFlowAnalyzer`: Farneback dense optical flow analysis
- `dynamics_tracker.py` -- `DynamicsTracker`, `MembraneMotion`: tracks membrane movement over time
- `correlation_analyzer.py` -- `CorrelationAnalyzer`, `CorrelationData`: movement-intensity correlation

### src/gui/
- `__init__.py` -- Subpackage docstring
- `dynamics_panel.py` -- Dynamics analysis display panel
- `time_sequence_panel.py` -- Temporal sequence visualization panel
- `visualization_panel.py` -- Main visualization panel with curvature and overlay

### src/utils/
- `__init__.py` -- Re-exports data structures, image utils, FrameStack, VectorField
- `data_structures.py` -- Dataclasses: ImageData, EdgeData, CurvatureData, MotionData, FlowData, CorrelationData, AnalysisParameters
- `image_utils.py` -- Image manipulation: normalize, segment, enhance, composite, overlay
- `frame_stack.py` -- `FrameStack`: TIFF stack loading and frame management
- `vector_field.py` -- `VectorField`: vector field operations

## docs/
- `membrane-dynamics-project.md` -- Project design document

## tests/
- `test_analysis.py` -- Tests for CurvatureAnalyzer, OpticalFlowAnalyzer, EdgeDetector, DynamicsTracker, and imports

## Key Class Relationships
MainWindow instantiates: EdgeDetector, DynamicsTracker, OpticalFlowAnalyzer, CorrelationAnalyzer, CurvatureAnalyzer.
All analyzers use pixel_size parameter for physical unit conversions.
