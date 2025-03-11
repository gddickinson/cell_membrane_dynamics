#!/usr/bin/env python3
# src/utils/data_structures.py

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

@dataclass
class ImageData:
    """Container for image data and metadata."""
    data: np.ndarray
    filename: str
    is_stack: bool = False
    current_frame: int = 0

    @property
    def shape(self):
        return self.data.shape

@dataclass
class EdgeData:
    """Container for cell edge detection results."""
    contour: np.ndarray  # Nx2 array of (x,y) coordinates
    edge_image: np.ndarray  # Binary edge image
    smoothed_contour: Optional[np.ndarray] = None  # Smoothed contour if available

@dataclass
class CurvatureData:
    """Container for curvature analysis results."""
    points: np.ndarray  # Points on the contour
    curvatures: np.ndarray  # Curvature values
    pixel_size: float  # Size of pixel in nanometers
    high_curvature_regions: Optional[List[Dict]] = None  # Regions of high curvature

@dataclass
class MotionData:
    """Container for membrane motion data."""
    points: np.ndarray  # Points on the membrane
    velocities: np.ndarray  # Velocity vectors
    magnitudes: np.ndarray  # Movement magnitudes
    directions: np.ndarray  # Movement directions
    classifications: List[str]  # Movement classifications
    frame_index: int  # Frame index
    time_delta: float  # Time between frames (seconds)

@dataclass
class FlowData:
    """Container for optical flow data."""
    flow_field: np.ndarray  # Dense flow field
    contour_flow: Optional[np.ndarray] = None  # Flow vectors along contour
    tangential_components: Optional[np.ndarray] = None  # Tangential component of flow
    normal_components: Optional[np.ndarray] = None  # Normal component of flow

@dataclass
class CorrelationData:
    """Container for curvature-motion correlation data."""
    curvatures: np.ndarray  # Curvature values
    velocities: np.ndarray  # Velocity vectors
    normal_velocities: np.ndarray  # Normal component of velocity
    tangential_velocities: np.ndarray  # Tangential component of velocity
    classifications: List[str]  # Movement classifications
    correlation: float  # Correlation coefficient
    p_value: float  # Statistical significance
    frame_index: int  # Frame index
    points: Optional[np.ndarray] = None  # Points on the contour

@dataclass
class AnalysisParameters:
    """Container for analysis parameters."""
    # Common parameters
    pixel_size: float = 100.0  # Size of pixel in nanometers
    time_delta: float = 1.0  # Time between frames (seconds)
    
    # Edge detection parameters
    smoothing_sigma: float = 1.0  # Gaussian smoothing sigma for contours
    min_size: int = 100  # Minimum object size to consider
    
    # Curvature analysis parameters
    segment_length: int = 9  # Length of segment for curvature calculation
    curvature_method: str = "finite_difference"  # Method for curvature calculation
    
    # Optical flow parameters
    window_size: int = 15  # Size of search window for optical flow
    max_level: int = 2  # Number of pyramid levels for optical flow
    
    # Visualization parameters
    show_vectors: bool = True  # Show velocity vectors
    show_contour: bool = True  # Show contour
    display_mode: str = "Original"  # Display mode
    
    # Kymograph parameters
    kymo_point_idx: int = 0  # Point index for kymograph
    kymo_window_size: int = 10  # Window size for kymograph