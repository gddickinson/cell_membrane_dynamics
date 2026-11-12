"""Regression tests for bugs fixed in the existing analysis code and desktop app (2026-11-11).

Run: python -m pytest -q tests/test_regressions.py
The GUI tests run Qt offscreen and are skipped when PyQt6 is not installed.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.correlation_analyzer import CorrelationAnalyzer, CorrelationData  # noqa: E402
from src.analysis.dynamics_tracker import DynamicsTracker  # noqa: E402
from src.analysis.edge_detection import EdgeDetector  # noqa: E402


def _disk(r, size=120):
    yy, xx = np.mgrid[:size, :size]
    return (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < r * r).astype(np.uint8)


def test_small_cell_dynamics_does_not_crash():
    """Contours shorter than 100 points used to raise IndexError (velocities resampled to 100 points, normals
    and optical flow sized to the real contour)."""
    ed = EdgeDetector()
    c1, c2 = ed.detect_edge(_disk(10)), ed.detect_edge(_disk(11))
    assert len(c1) < 100 and len(c2) < 100
    frame = np.zeros((120, 120), np.uint8)
    tr = DynamicsTracker(pixel_size=100.0)
    tr.analyze_frame(frame, c1)
    motion = tr.analyze_frame(frame, c2)
    assert len(motion.velocities) == len(motion.classifications) <= len(c2)


def test_large_cell_dynamics_unchanged():
    """The fix leaves contours with >= 100 points exactly as before (resampled to min(len(prev), len(curr)))."""
    ed = EdgeDetector()
    c1, c2 = ed.detect_edge(_disk(40, 200)), ed.detect_edge(_disk(42, 200))
    frame = np.zeros((200, 200), np.uint8)
    tr = DynamicsTracker(pixel_size=100.0)
    tr.analyze_frame(frame, c1)
    motion = tr.analyze_frame(frame, c2)
    assert len(motion.velocities) == min(len(c1), len(c2))


def test_curvature_motion_map_runs():
    """create_curvature_motion_map used cv2 without importing it (NameError)."""
    pts = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], float)
    data = CorrelationData(curvatures=np.array([1e-3, -1e-3, 2e-3, 0.0]), velocities=np.ones((4, 2)),
                           normal_velocities=np.zeros(4), tangential_velocities=np.zeros(4),
                           classifications=["expanding", "retracting", "flowing", "stationary"],
                           correlation=0.0, p_value=1.0, frame_index=1, points=pts)
    img = CorrelationAnalyzer().create_curvature_motion_map(data, np.zeros((60, 60), np.uint8))
    assert img.shape == (60, 60, 3)


def test_velocity_axis_labels_are_pixels_per_second():
    """Velocities are computed in pixels/s (the static threshold is 10 nm/s / pixel_size), so plots must not
    label them nm/s."""
    for f in ["main.py", "src/analysis/correlation_analyzer.py"]:
        assert "Velocity (nm/s)" not in (ROOT / f).read_text(), f


@pytest.fixture(scope="module")
def window():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication([])
    import main
    win = main.MainWindow()
    yield win
    win.close()
    del qapp


def test_gui_loads_every_page_of_a_multipage_tiff(window, tmp_path):
    """The README promises multi-page TIFFs; the GUI loaded only the first page with cv2.imread."""
    tifffile = pytest.importorskip("tifffile")
    stack = np.stack([_disk(20 + i, 100) * 255 for i in range(4)]).astype(np.uint8)
    f = tmp_path / "cell_1_masks.tif"
    tifffile.imwrite(f, stack)
    window._load_image_sequence(str(f), is_mask=True)
    assert len(window.masks) == 4
    assert np.array_equal(window.masks[2], stack[2])


def test_edge_smoothing_slider_reaches_edge_detector(window):
    """The 'Edge Smoothing' slider only changed the optical-flow window, never the edge smoothing."""
    window.smoothing_slider.setValue(30)
    assert window.edge_detector.smoothing_sigma == pytest.approx(3.0)
    window.smoothing_slider.setValue(10)
    assert window.edge_detector.smoothing_sigma == pytest.approx(1.0)
