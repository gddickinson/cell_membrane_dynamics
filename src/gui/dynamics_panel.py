#!/usr/bin/env python3
# src/gui/dynamics_panel.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSlider, QDoubleSpinBox, QComboBox,
    QCheckBox, QPushButton, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Optional
from ..utils.data_structures import AnalysisParameters

class DynamicsPanel(QWidget):
    """Panel for controlling dynamics analysis parameters."""
    
    # Signal emitted when parameters change
    parameters_changed = pyqtSignal(AnalysisParameters)
    
    def __init__(self, params: Optional[AnalysisParameters] = None):
        """Initialize dynamics panel.
        
        Args:
            params: Initial parameters (optional)
        """
        super().__init__()
        
        # Create default parameters if not provided
        self.params = params if params is not None else AnalysisParameters()
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Create parameter groups
        layout.addWidget(self._create_common_group())
        layout.addWidget(self._create_edge_group())
        layout.addWidget(self._create_curvature_group())
        layout.addWidget(self._create_flow_group())
        
        # Add stretch at the bottom
        layout.addStretch()
        
    def _create_common_group(self) -> QGroupBox:
        """Create group for common parameters."""
        group = QGroupBox("Common Parameters")
        layout = QVBoxLayout()
        
        # Pixel size control
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("Pixel Size (nm):"))
        self.pixel_size_input = QDoubleSpinBox()
        self.pixel_size_input.setMinimum(1.0)
        self.pixel_size_input.setMaximum(1000.0)
        self.pixel_size_input.setValue(self.params.pixel_size)
        self.pixel_size_input.valueChanged.connect(self._on_pixel_size_changed)
        pixel_layout.addWidget(self.pixel_size_input)
        layout.addLayout(pixel_layout)
        
        # Time delta control
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time Between Frames (s):"))
        self.time_delta_input = QDoubleSpinBox()
        self.time_delta_input.setMinimum(0.01)
        self.time_delta_input.setMaximum(60.0)
        self.time_delta_input.setValue(self.params.time_delta)
        self.time_delta_input.valueChanged.connect(self._on_time_delta_changed)
        time_layout.addWidget(self.time_delta_input)
        layout.addLayout(time_layout)
        
        group.setLayout(layout)
        return group
        
    def _create_edge_group(self) -> QGroupBox:
        """Create group for edge detection parameters."""
        group = QGroupBox("Edge Detection")
        layout = QVBoxLayout()
        
        # Edge smoothing slider
        smooth_layout = QHBoxLayout()
        self.smoothing_label = QLabel(f"Edge Smoothing: {self.params.smoothing_sigma:.1f}")
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setMinimum(0)
        self.smoothing_slider.setMaximum(50)
        self.smoothing_slider.setValue(int(self.params.smoothing_sigma * 10))
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        smooth_layout.addWidget(self.smoothing_label)
        smooth_layout.addWidget(self.smoothing_slider)
        layout.addLayout(smooth_layout)
        
        # Minimum size slider
        min_size_layout = QHBoxLayout()
        self.min_size_label = QLabel(f"Minimum Size: {self.params.min_size}")
        self.min_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_size_slider.setMinimum(10)
        self.min_size_slider.setMaximum(1000)
        self.min_size_slider.setValue(self.params.min_size)
        self.min_size_slider.valueChanged.connect(self._on_min_size_changed)
        min_size_layout.addWidget(self.min_size_label)
        min_size_layout.addWidget(self.min_size_slider)
        layout.addLayout(min_size_layout)
        
        group.setLayout(layout)
        return group
        
    def _create_curvature_group(self) -> QGroupBox:
        """Create group for curvature analysis parameters."""
        group = QGroupBox("Curvature Analysis")
        layout = QVBoxLayout()
        
        # Curvature method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.curvature_method = QComboBox()
        self.curvature_method.addItems(["Finite Difference", "Circle Fit", "Spline"])
        
        # Set current selection based on parameter
        if self.params.curvature_method == "finite_difference":
            current_index = 0
        elif self.params.curvature_method == "circle_fit":
            current_index = 1
        else:  # spline
            current_index = 2
        self.curvature_method.setCurrentIndex(current_index)
        
        self.curvature_method.currentIndexChanged.connect(self._on_curvature_method_changed)
        method_layout.addWidget(self.curvature_method)
        layout.addLayout(method_layout)
        
        # Segment length slider
        segment_layout = QHBoxLayout()
        self.segment_label = QLabel(f"Segment Length: {self.params.segment_length}")
        self.segment_slider = QSlider(Qt.Orientation.Horizontal)
        self.segment_slider.setMinimum(3)
        self.segment_slider.setMaximum(41)
        self.segment_slider.setSingleStep(2)
        self.segment_slider.setValue(self.params.segment_length)
        self.segment_slider.valueChanged.connect(self._on_segment_length_changed)
        segment_layout.addWidget(self.segment_label)
        segment_layout.addWidget(self.segment_slider)
        layout.addLayout(segment_layout)
        
        group.setLayout(layout)
        return group
        
    def _create_flow_group(self) -> QGroupBox:
        """Create group for optical flow parameters."""
        group = QGroupBox("Optical Flow")
        layout = QVBoxLayout()
        
        # Window size slider
        window_layout = QHBoxLayout()
        self.window_label = QLabel(f"Window Size: {self.params.window_size}")
        self.window_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_slider.setMinimum(3)
        self.window_slider.setMaximum(31)
        self.window_slider.setSingleStep(2)
        self.window_slider.setValue(self.params.window_size)
        self.window_slider.valueChanged.connect(self._on_window_size_changed)
        window_layout.addWidget(self.window_label)
        window_layout.addWidget(self.window_slider)
        layout.addLayout(window_layout)
        
        # Max level slider
        level_layout = QHBoxLayout()
        self.level_label = QLabel(f"Pyramid Levels: {self.params.max_level}")
        self.level_slider = QSlider(Qt.Orientation.Horizontal)
        self.level_slider.setMinimum(0)
        self.level_slider.setMaximum(5)
        self.level_slider.setValue(self.params.max_level)
        self.level_slider.valueChanged.connect(self._on_max_level_changed)
        level_layout.addWidget(self.level_label)
        level_layout.addWidget(self.level_slider)
        layout.addLayout(level_layout)
        
        # Flow visualization options
        viz_layout = QHBoxLayout()
        self.show_vectors = QCheckBox("Show Vectors")
        self.show_vectors.setChecked(self.params.show_vectors)
        self.show_vectors.stateChanged.connect(self._on_show_vectors_changed)
        
        self.show_contour = QCheckBox("Show Contour")
        self.show_contour.setChecked(self.params.show_contour)
        self.show_contour.stateChanged.connect(self._on_show_contour_changed)
        
        viz_layout.addWidget(self.show_vectors)
        viz_layout.addWidget(self.show_contour)
        layout.addLayout(viz_layout)
        
        group.setLayout(layout)
        return group
        
    # Parameter update handlers
    def _on_pixel_size_changed(self, value):
        """Update pixel size parameter."""
        self.params.pixel_size = value
        self._emit_parameters_changed()
        
    def _on_time_delta_changed(self, value):
        """Update time delta parameter."""
        self.params.time_delta = value
        self._emit_parameters_changed()
        
    def _on_smoothing_changed(self, value):
        """Update smoothing parameter."""
        self.params.smoothing_sigma = value / 10.0
        self.smoothing_label.setText(f"Edge Smoothing: {self.params.smoothing_sigma:.1f}")
        self._emit_parameters_changed()
        
    def _on_min_size_changed(self, value):
        """Update minimum size parameter."""
        self.params.min_size = value
        self.min_size_label.setText(f"Minimum Size: {value}")
        self._emit_parameters_changed()
        
    def _on_curvature_method_changed(self, index):
        """Update curvature method parameter."""
        if index == 0:
            self.params.curvature_method = "finite_difference"
        elif index == 1:
            self.params.curvature_method = "circle_fit"
        else:
            self.params.curvature_method = "spline"
        self._emit_parameters_changed()
        
    def _on_segment_length_changed(self, value):
        """Update segment length parameter."""
        self.params.segment_length = value
        self.segment_label.setText(f"Segment Length: {value}")
        self._emit_parameters_changed()
        
    def _on_window_size_changed(self, value):
        """Update window size parameter."""
        self.params.window_size = value
        self.window_label.setText(f"Window Size: {value}")
        self._emit_parameters_changed()
        
    def _on_max_level_changed(self, value):
        """Update max level parameter."""
        self.params.max_level = value
        self.level_label.setText(f"Pyramid Levels: {value}")
        self._emit_parameters_changed()
        
    def _on_show_vectors_changed(self, state):
        """Update show vectors parameter."""
        self.params.show_vectors = state == Qt.CheckState.Checked
        self._emit_parameters_changed()
        
    def _on_show_contour_changed(self, state):
        """Update show contour parameter."""
        self.params.show_contour = state == Qt.CheckState.Checked
        self._emit_parameters_changed()
        
    def _emit_parameters_changed(self):
        """Emit parameters changed signal."""
        self.parameters_changed.emit(self.params)
        
    def get_parameters(self) -> AnalysisParameters:
        """Get current parameters.
        
        Returns:
            Analysis parameters
        """
        return self.params
        
    def set_parameters(self, params: AnalysisParameters):
        """Set parameters.
        
        Args:
            params: New parameters
        """
        self.params = params
        
        # Update UI controls
        self.pixel_size_input.setValue(params.pixel_size)
        self.time_delta_input.setValue(params.time_delta)
        self.smoothing_slider.setValue(int(params.smoothing_sigma * 10))
        self.min_size_slider.setValue(params.min_size)
        
        # Set curvature method
        if params.curvature_method == "finite_difference":
            self.curvature_method.setCurrentIndex(0)
        elif params.curvature_method == "circle_fit":
            self.curvature_method.setCurrentIndex(1)
        else:  # spline
            self.curvature_method.setCurrentIndex(2)
            
        self.segment_slider.setValue(params.segment_length)
        self.window_slider.setValue(params.window_size)
        self.level_slider.setValue(params.max_level)
        self.show_vectors.setChecked(params.show_vectors)
        self.show_contour.setChecked(params.show_contour)