# PIEZO1 Membrane Dynamics Tracking

A Python-based tool for analyzing membrane dynamics and correlating curvature with movement in TIRF microscopy data.

## Project Overview

This tool extends curvature analysis by incorporating temporal information to distinguish between expanding protrusions (e.g., blebs) and retracting regions of cell membranes. By correlating curvature measurements with membrane movement, it provides context for interpreting membrane-protein interactions.

## Project Structure

```
membrane_dynamics/
├── README.md
├── main.py                 # Application entry point
├── requirements.txt        # Package dependencies
└── src/
    ├── __init__.py
    ├── analysis/
    │   ├── __init__.py
    │   ├── edge_detection.py
    │   ├── curvature_analyzer.py
    │   ├── dynamics_tracker.py      # New module for tracking movement
    │   ├── optical_flow.py          # Implements motion tracking
    │   └── correlation_analyzer.py  # Correlates motion with curvature
    ├── gui/
    │   ├── __init__.py
    │   ├── main_window.py
    │   ├── dynamics_panel.py        # Controls for dynamics analysis
    │   ├── visualization_panel.py   
    │   └── time_sequence_panel.py   # For viewing sequential frames
    └── utils/
        ├── __init__.py
        ├── data_structures.py
        ├── frame_stack.py           # For handling time-series data
        └── vector_field.py          # For representing motion fields
```

## Core Concepts

### 1. Motion Tracking

Tracks membrane movement between consecutive frames using:
- Contour point correspondence
- Local optical flow calculation
- Registration of membrane segments

### 2. Movement Classification

Classifies membrane regions as:
- **Expanding**: Moving outward from cell (e.g., blebs)
- **Retracting**: Moving inward (e.g., membrane retraction)
- **Stationary**: Minimal movement
- **Flowing**: Lateral movement along the membrane

### 3. Curvature-Motion Correlation

Correlates curvature measurements with movement direction to provide biological context:
- Positive curvature + outward motion = Bleb formation
- Positive curvature + inward motion = Retraction completion
- Negative curvature + outward motion = Retraction initiation
- Negative curvature + inward motion = Bleb retraction

### 4. Temporal Visualization

Provides visualization tools to represent dynamics:
- Motion vectors overlaid on membrane
- Color-coded movement classification
- Time sequence animations
- Kymographs for tracking specific regions over time

## Technical Implementation

The core technical components include:

1. **Contour Point Tracking Algorithm**
   - Implements point correspondence between frames
   - Handles topological changes in the membrane

2. **Optical Flow Analysis**
   - Applies Lucas-Kanade optical flow to membrane regions
   - Calculates velocity vectors for membrane segments

3. **Curvature-Motion Integration**
   - Combines geometric (curvature) and dynamic (motion) information
   - Creates unified representation of membrane activity

4. **Temporal Data Structures**
   - Efficiently stores and processes time-series data
   - Enables multi-frame analysis

## Analysis Workflow

1. **Input**
   - Time-series of cell mask images
   - Time-series of fluorescence images

2. **Preprocessing**
   - Edge detection on each frame
   - Registration to correct for cell movement/drift

3. **Dynamics Analysis**
   - Tracking of membrane points across frames
   - Calculation of movement vectors
   - Classification of movement patterns

4. **Correlation Analysis**
   - Integration of curvature and movement data
   - Identification of biologically significant patterns

5. **Visualization**
   - Multi-frame visualization
   - Interactive temporal analysis
   - Motion-enhanced curvature maps

## Dependencies

- OpenCV for image processing and optical flow
- NumPy for numerical computing
- SciPy for scientific algorithms
- scikit-image for image analysis
- PyQt6 for GUI
- matplotlib for visualization
