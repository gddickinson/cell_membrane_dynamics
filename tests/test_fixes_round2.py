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


# --- 3. edge correspondence between frames ------------------------------------------------------------------------

def _motion(prev, curr, dt=1.0):
    tr = DynamicsTracker(pixel_size=100.0)
    tr.set_time_delta(dt)
    frame = np.zeros((220, 220), np.uint8)
    tr.analyze_frame(frame, prev)
    return tr.analyze_frame(frame, curr)


def test_unchanged_outline_with_a_different_start_point_is_stationary():
    """Same circle, start point rotated by 37 degrees and a different point count: no motion."""
    m = _motion(circle(40.0, n=250), circle(40.0, n=260, start=np.deg2rad(37)))
    assert len(m.velocities) == 260
    assert np.max(np.linalg.norm(m.velocities, axis=1)) < 0.05      # px/s
    assert set(m.classifications) == {"stationary"}


def test_expanding_circle_with_rotated_start_is_expanding():
    """Radius 40 -> 42 px in 2 s (1 px/s outward) with the start point rotated: radial motion, not 'flowing'."""
    m = _motion(circle(40.0, n=250), circle(42.0, n=264, start=np.deg2rad(123)), dt=2.0)
    radial = (m.points - 100.0) / np.linalg.norm(m.points - 100.0, axis=1)[:, None]
    vr = np.sum(m.velocities * radial, axis=1)
    vt = np.abs(m.velocities[:, 0] * -radial[:, 1] + m.velocities[:, 1] * radial[:, 0])
    assert np.allclose(vr, 1.0, atol=0.02) and vt.max() < 0.05
    assert set(m.classifications) == {"expanding"}


def test_velocities_pair_with_every_current_point():
    ed = EdgeDetector()
    yy, xx = np.mgrid[:200, :200]
    a = ed.detect_edge((((xx - 100) ** 2 + (yy - 100) ** 2) < 45 ** 2).astype(np.uint8))
    b = ed.detect_edge((((xx - 102) / 47.0) ** 2 + ((yy - 99) / 44.0) ** 2 < 1).astype(np.uint8))
    m = _motion(a, b)
    assert len(m.velocities) == len(m.classifications) == len(b)


# --- 4. circle-fit curvature ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("r", [8.0, 25.0, 60.0])
@pytest.mark.parametrize("clockwise", [False, True])
def test_circle_fit_curvature_is_one_over_r(r, clockwise):
    an = CurvatureAnalyzer(pixel_size=100.0, segment_length=9)
    k = an.calculate_curvature(circle(r, n=int(2 * np.pi * r), clockwise=clockwise), method="circle_fit")
    assert np.all(k > 0)
    assert np.allclose(k, 1 / (r * 100.0), rtol=1e-6)


def test_circle_fit_sign_on_a_dent():
    yy, xx = np.mgrid[:200, :200]
    shape = ((xx - 100) ** 2 + (yy - 100) ** 2 < 50 ** 2) & ~((xx - 150) ** 2 + (yy - 100) ** 2 < 20 ** 2)
    c = EdgeDetector(smoothing_sigma=2.0).detect_edge(shape.astype(np.uint8))
    k = CurvatureAnalyzer(pixel_size=100.0, segment_length=9).calculate_curvature(c, method="circle_fit")
    dent = np.hypot(c[:, 0] - 150, c[:, 1] - 100) < 21
    far = np.hypot(c[:, 0] - 150, c[:, 1] - 100) > 35
    assert np.median(k[far]) > 0 and np.median(k[dent]) < 0
    assert np.median(k[far]) == pytest.approx(1 / (50 * 100.0), rel=0.15)
