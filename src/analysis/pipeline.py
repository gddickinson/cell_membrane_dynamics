#!/usr/bin/env python3
# src/analysis/pipeline.py
"""Qt-free analysis pipeline: the frame loop of the desktop app's "Run Analysis" (MainWindow.run_analysis).

For each frame: detect the cell edge (EdgeDetector), finite-difference curvature (nm^-1), membrane dynamics against
the previous frame (DynamicsTracker; velocities in pixels/s), and the curvature / normal-velocity correlation
(CorrelationAnalyzer). main.py and the web interface both call analyze_sequence, so they give the same numbers.
"""
from typing import Callable, Dict, List, Optional

import numpy as np

from .correlation_analyzer import CorrelationAnalyzer
from .curvature_analyzer import CurvatureAnalyzer
from .dynamics_tracker import DynamicsTracker
from .edge_detection import EdgeDetector


def finite_difference_curvature(contour: np.ndarray, pixel_size: float) -> np.ndarray:
    """Signed curvature (nm^-1) at every contour point; pixel_size in nm per pixel."""
    return CurvatureAnalyzer(pixel_size=pixel_size)._finite_difference_curvature(contour)


def contour_normals(contour: np.ndarray) -> np.ndarray:
    """Unit normals at every contour point, oriented towards the contour's centroid (approximately inward)."""
    return EdgeDetector().calculate_normals(contour)


def binarize(mask: np.ndarray) -> np.ndarray:
    """The app's mask preparation: any value above 1 becomes 255 (masks stored as 0/1 are passed through)."""
    mask = mask.copy()
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8) * 255
    return mask


def correlation_inputs(contour: np.ndarray, curvature: np.ndarray, motion) -> Dict:
    """Velocity components paired with curvature, truncated to the shorter of the two (as in the desktop app)."""
    n = min(len(curvature), len(motion.velocities))
    normals = contour_normals(contour)[:n]
    vel = motion.velocities[:n]
    return {
        'velocities': vel,
        'normal_components': np.array([np.dot(v, nv) for v, nv in zip(vel, normals)]),
        'tangential_components': np.array([np.dot(v, [-nv[1], nv[0]]) for v, nv in zip(vel, normals)]),
        'classifications': motion.classifications[:n],
        'points': contour[:n],
    }


def analyze_sequence(masks: List[np.ndarray], frames: List[np.ndarray], pixel_size: float = 100.0,
                     time_delta: float = 1.0, smoothing: float = 1.0, start: int = 0, end: Optional[int] = None,
                     progress: Optional[Callable[[int, int, int], None]] = None,
                     edge_detector: Optional[EdgeDetector] = None) -> Dict:
    """Run the full analysis on frames start..end (inclusive).

    Args:
        masks: binary cell masks, one per frame
        frames: PIEZO1 fluorescence frames (used for Lucas-Kanade optical flow at the edge)
        pixel_size: nm per pixel; time_delta: seconds between frames
        smoothing: optical-flow window parameter of DynamicsTracker (the app's Edge Smoothing value)
        progress: optional callback(frame_index, start, end) called before each frame
        edge_detector: EdgeDetector to use (default: smoothing sigma = smoothing)

    Returns:
        dict with contours, curvatures, motion_data (None for the first frame), correlation_data, failed (frame
        indices with no detectable edge), frame_ids (frame index of each contour) and the DynamicsTracker.
    """
    if end is None:
        end = min(len(masks), len(frames)) - 1
    detector = edge_detector or EdgeDetector(smoothing_sigma=smoothing)
    tracker = DynamicsTracker(contour_smoothing=smoothing, pixel_size=pixel_size)
    tracker.set_time_delta(time_delta)
    correlator = CorrelationAnalyzer(pixel_size=pixel_size)
    out = {'contours': [], 'curvatures': [], 'motion_data': [], 'correlation_data': [], 'failed': [],
           'frame_ids': [], 'tracker': tracker}
    for i in range(start, end + 1):
        if progress:
            progress(i, start, end)
        mask = binarize(masks[i])
        frame = frames[i].copy()
        contour = detector.detect_edge(mask)
        if contour is None or len(contour) < 5:
            out['failed'].append(i)
            continue
        out['contours'].append(contour)
        out['frame_ids'].append(i)
        curvature = finite_difference_curvature(contour, pixel_size)
        out['curvatures'].append(curvature)
        motion = tracker.analyze_frame(frame, contour)
        if i == start:
            out['motion_data'].append(None)
            continue
        out['motion_data'].append(motion)
        if motion is not None and len(curvature) == len(contour):
            data = correlation_inputs(contour, curvature, motion)
            n = len(data['velocities'])
            out['correlation_data'].append(correlator.analyze_correlation(curvature[:n], data, i))
    return out
