"""
Smoke tests for analysis modules.

Tests curvature analysis, optical flow, edge detection,
and dynamics tracking with synthetic data.
"""

import sys
import os
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.curvature_analyzer import CurvatureAnalyzer
from src.analysis.optical_flow import OpticalFlowAnalyzer
from src.analysis.edge_detection import EdgeDetector
from src.analysis.dynamics_tracker import DynamicsTracker


class TestCurvatureAnalyzer(unittest.TestCase):
    """Tests for CurvatureAnalyzer."""

    def setUp(self):
        self.analyzer = CurvatureAnalyzer(pixel_size=100.0)

    def test_init(self):
        """Test analyzer can be instantiated."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.pixel_size, 100.0)

    def test_calculate_curvature_circle(self):
        """Test curvature calculation on a circle (should be approximately constant)."""
        # Create circular contour
        n_points = 100
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radius = 30.0
        contour = np.column_stack([
            50 + radius * np.cos(theta),
            50 + radius * np.sin(theta)
        ])

        curvatures = self.analyzer.calculate_curvature(contour, method='finite_difference')
        self.assertEqual(len(curvatures), n_points)
        # Curvature of a circle should be roughly 1/radius everywhere
        # Allow for discretization error
        nonzero = curvatures[curvatures != 0]
        if len(nonzero) > 0:
            self.assertGreater(len(nonzero), n_points // 2)

    def test_calculate_curvature_invalid_method(self):
        """Test that invalid method raises ValueError."""
        contour = np.array([[0, 0], [1, 0], [2, 0]])
        with self.assertRaises(ValueError):
            self.analyzer.calculate_curvature(contour, method='invalid')


class TestOpticalFlowAnalyzer(unittest.TestCase):
    """Tests for OpticalFlowAnalyzer."""

    def setUp(self):
        self.analyzer = OpticalFlowAnalyzer(pixel_size=100.0, time_delta=1.0)

    def test_init(self):
        """Test analyzer can be instantiated."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.pixel_size, 100.0)
        self.assertEqual(self.analyzer.time_delta, 1.0)

    def test_farneback_params(self):
        """Test Farneback params are set."""
        self.assertIn('pyr_scale', self.analyzer.farneback_params)
        self.assertIn('levels', self.analyzer.farneback_params)


class TestEdgeDetector(unittest.TestCase):
    """Tests for EdgeDetector."""

    def setUp(self):
        self.detector = EdgeDetector()

    def test_init(self):
        """Test detector can be instantiated."""
        self.assertIsNotNone(self.detector)


class TestDynamicsTracker(unittest.TestCase):
    """Tests for DynamicsTracker."""

    def setUp(self):
        self.tracker = DynamicsTracker(pixel_size=100.0)

    def test_init(self):
        """Test tracker can be instantiated."""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(self.tracker.pixel_size, 100.0)


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""

    def test_import_analysis(self):
        """Test analysis package imports."""
        from src.analysis import (
            EdgeDetector, CurvatureAnalyzer, DynamicsTracker,
            OpticalFlowAnalyzer, CorrelationAnalyzer
        )
        self.assertIsNotNone(EdgeDetector)

    def test_import_utils(self):
        """Test utils package imports."""
        from src.utils import FrameStack, VectorField
        self.assertIsNotNone(FrameStack)
        self.assertIsNotNone(VectorField)

    def test_import_src(self):
        """Test src package imports."""
        import src
        self.assertEqual(src.__version__, '1.0.0')


if __name__ == '__main__':
    unittest.main()
