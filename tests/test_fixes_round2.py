"""Regression tests for the second round of fixes (2026-11-12): curvature sign, 16-bit optical flow,
frame-to-frame edge correspondence, circle-fit curvature.

Run: python -m pytest -q tests/test_fixes_round2.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.curvature_analyzer import CurvatureAnalyzer  # noqa: E402
from src.analysis.edge_detection import EdgeDetector  # noqa: E402


def circle(r, n=200, cx=100.0, cy=100.0, clockwise=False, start=0.0):
    t = start + np.linspace(0, 2 * np.pi, n, endpoint=False) * (-1 if clockwise else 1)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


# --- 1. curvature sign: positive on a convex outline, whatever the point order ----------------------------------

@pytest.mark.parametrize("clockwise", [False, True])
def test_circle_curvature_positive_and_one_over_r(clockwise):
    k = CurvatureAnalyzer(pixel_size=100.0).calculate_curvature(circle(30.0, clockwise=clockwise))
    assert np.all(k > 0)
    assert np.mean(k) == pytest.approx(1 / (30.0 * 100.0), rel=1e-3)   # nm^-1


def test_detected_disk_is_positive_and_dent_is_negative():
    yy, xx = np.mgrid[:200, :200]
    disk = ((xx - 100) ** 2 + (yy - 100) ** 2 < 50 ** 2) & ~((xx - 150) ** 2 + (yy - 100) ** 2 < 20 ** 2)
    c = EdgeDetector(smoothing_sigma=2.0).detect_edge(disk.astype(np.uint8))
    k = CurvatureAnalyzer(pixel_size=100.0).calculate_curvature(c)
    far = np.hypot(c[:, 0] - 150, c[:, 1] - 100) > 35          # the round part, away from the dent
    dent = np.hypot(c[:, 0] - 150, c[:, 1] - 100) < 21
    assert np.median(k[far]) > 0 and np.median(k[dent]) < 0


# --- 2. 16-bit PIEZO1 frames for optical flow ---------------------------------------------------------------------

from src.analysis.dynamics_tracker import DynamicsTracker, frames_to_uint8  # noqa: E402


def _blob16(cx, size=120):
    """A textured 16-bit blob (values well above 255) centred at (cx, 60)."""
    yy, xx = np.mgrid[:size, :size].astype(float)
    texture = 1 + 0.3 * np.sin(xx / 3.0 - cx / 3.0) * np.cos(yy / 4.0)
    return (2000 + 30000 * np.exp(-((xx - cx) ** 2 + (yy - 60) ** 2) / (2 * 12.0 ** 2)) * texture).astype(np.uint16)


def test_frames_to_uint8_scales_without_wrapping():
    a = np.array([[300, 1000], [5000, 65535]], np.uint16)
    b = np.array([[256, 512], [1024, 2048]], np.uint16)
    pa, pb = frames_to_uint8(a, b)
    assert pa.dtype == np.uint8 and pb.dtype == np.uint8
    assert pa.max() == 255 and pb.min() == 0                    # joint range of the pair
    assert pb[0, 0] < pb[0, 1] < pb[1, 0] < pb[1, 1]            # order kept (the old cast mapped 256 and 512 to 0)
    u8 = np.array([[1, 2]], np.uint8)
    assert frames_to_uint8(u8, u8)[0] is u8                       # 8-bit frames pass through unchanged


def test_optical_flow_on_16bit_frames_follows_the_shift():
    tr = DynamicsTracker(pixel_size=100.0)
    prev, curr = _blob16(55.0), _blob16(58.0)                    # moved 3 px in x
    pts = np.array([[55.0 + 12 * np.cos(t), 60 + 12 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
    v = tr._calculate_optical_flow_at_contour(prev, curr, pts, pts)
    found = np.linalg.norm(v, axis=1) > 0
    assert found.mean() > 0.8
    assert np.median(v[found, 0]) == pytest.approx(3.0, abs=0.5) and abs(np.median(v[found, 1])) < 0.5
