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
