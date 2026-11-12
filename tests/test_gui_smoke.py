"""Desktop app smoke test (Qt offscreen): run the analysis on the synthetic demo stack and draw every view.

Run: python -m pytest -q tests/test_gui_smoke.py   (skipped without PyQt6)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_main_window_runs_analysis_and_draws_every_view():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication([])
    import main
    from src.analysis.pipeline import analyze_sequence
    from src.web.demo_data import demo_fluorescence, demo_masks
    m = demo_masks()[:5]
    f = demo_fluorescence(m)
    w = main.MainWindow()
    w.masks, w.frames, w.time_delta = list(m), list(f), 2.0
    w.run_analysis()
    ref = analyze_sequence(list(m), list(f), pixel_size=100.0, time_delta=2.0)
    assert len(w.contours) == 5
    assert all(np.array_equal(a.velocities, b.velocities) for a, b in zip(w.correlation_data, ref["correlation_data"]))
    for mode in range(w.display_mode.count()):
        w.display_mode.setCurrentIndex(mode)
        w.update_visualization()
    w.change_frame(3)
    w.update_correlation_plot()
    w.update_time_series()
    w.update_kymograph()
    w.close()
    del qapp
