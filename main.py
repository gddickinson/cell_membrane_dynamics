#!/usr/bin/env python3
# main.py

import sys
import os
import numpy as np
import cv2
from PyQt6.QtWidgets import (
QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QComboBox,
    QTabWidget, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

from src.analysis.edge_detection import EdgeDetector
from src.analysis.dynamics_tracker import DynamicsTracker
from src.analysis.optical_flow import OpticalFlowAnalyzer
from src.analysis.correlation_analyzer import CorrelationAnalyzer, CorrelationData
from src.analysis.curvature_analyzer import CurvatureAnalyzer

class MainWindow(QMainWindow):
    """Main window for membrane dynamics analysis application."""

    def __init__(self):
        super().__init__()

        # Set application title
        self.setWindowTitle("PIEZO1 Membrane Dynamics Tracker")

        # Initialize analysis parameters
        self.pixel_size = 100.0  # nm per pixel
        self.time_delta = 1.0  # seconds between frames

        # Initialize analyzers
        self.edge_detector = EdgeDetector()
        self.dynamics_tracker = DynamicsTracker(pixel_size=self.pixel_size)
        self.optical_flow = OpticalFlowAnalyzer(pixel_size=self.pixel_size, time_delta=self.time_delta)
        self.correlation_analyzer = CorrelationAnalyzer(pixel_size=self.pixel_size)
        self.curvature_analyzer = CurvatureAnalyzer()

        # Data storage
        self.frames = []
        self.masks = []
        self.contours = []
        self.curvatures = []
        self.motion_data = []
        self.correlation_data = []
        self.current_frame_idx = 0

        # Initialize UI
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)

        # Create horizontal layout for controls and visualization
        h_layout = QHBoxLayout()

        # Add control panel on the left
        control_panel = self._create_control_panel()
        h_layout.addWidget(control_panel, 1)

        # Add visualization panel on the right
        visualization_panel = self._create_visualization_panel()
        h_layout.addWidget(visualization_panel, 3)

        main_layout.addLayout(h_layout)

        # Add status bar for messages
        self.statusBar().showMessage("Ready")

        # Set default window size
        self.resize(1600, 1000)

    def _create_control_panel(self):
        """Create the control panel with file loading and parameters."""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Create file loading section
        file_group = QGroupBox("File Loading")
        file_layout = QVBoxLayout()

        # Load Cell Mask button
        self.load_mask_button = QPushButton("Load Cell Mask Sequence")
        self.load_mask_button.clicked.connect(self.load_mask_sequence)
        file_layout.addWidget(self.load_mask_button)

        # Cell mask info label
        self.mask_info_label = QLabel("No cell mask loaded")
        file_layout.addWidget(self.mask_info_label)

        # Load PIEZO1 Fluorescence button
        self.load_fluor_button = QPushButton("Load PIEZO1 Sequence")
        self.load_fluor_button.clicked.connect(self.load_fluorescence_sequence)
        file_layout.addWidget(self.load_fluor_button)

        # Fluorescence info label
        self.fluor_info_label = QLabel("No fluorescence data loaded")
        file_layout.addWidget(self.fluor_info_label)

        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        # Create parameters section
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QVBoxLayout()

        # Pixel Size
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("Pixel Size (nm):"))
        self.pixel_size_input = QDoubleSpinBox()
        self.pixel_size_input.setRange(1, 1000)
        self.pixel_size_input.setValue(self.pixel_size)
        self.pixel_size_input.valueChanged.connect(self.update_parameters)
        pixel_layout.addWidget(self.pixel_size_input)
        params_layout.addLayout(pixel_layout)

        # Time delta
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time Between Frames (s):"))
        self.time_delta_input = QDoubleSpinBox()
        self.time_delta_input.setRange(0.01, 60)
        self.time_delta_input.setValue(self.time_delta)
        self.time_delta_input.valueChanged.connect(self.update_parameters)
        time_layout.addWidget(self.time_delta_input)
        params_layout.addLayout(time_layout)

        # Edge smoothing
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Edge Smoothing:"))
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 50)
        self.smoothing_slider.setValue(10)
        self.smoothing_slider.valueChanged.connect(self.update_parameters)
        smooth_layout.addWidget(self.smoothing_slider)
        self.smoothing_value_label = QLabel("1.0")
        smooth_layout.addWidget(self.smoothing_value_label)
        params_layout.addLayout(smooth_layout)

        params_group.setLayout(params_layout)
        control_layout.addWidget(params_group)

        # Frame navigation
        frame_group = QGroupBox("Frame Navigation")
        frame_layout = QVBoxLayout()

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.change_frame)
        frame_layout.addWidget(self.frame_slider)

        # Frame navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)

        self.frame_label = QLabel("Frame: 0/0")
        nav_layout.addWidget(self.frame_label)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)

        frame_layout.addLayout(nav_layout)

        # Add play animation button
        self.play_button = QPushButton("Play Animation")
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_animation)
        frame_layout.addWidget(self.play_button)

        frame_group.setLayout(frame_layout)
        control_layout.addWidget(frame_group)


        # Analysis controls
        analysis_group = QGroupBox("Analysis Controls")
        analysis_layout = QVBoxLayout()

        # Run Analysis button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setEnabled(False)
        analysis_layout.addWidget(self.run_button)

        # Frame range selection for batch analysis
        frame_range_group = QGroupBox("Frame Range")
        frame_range_layout = QHBoxLayout()

        # Start frame selector
        frame_range_layout.addWidget(QLabel("Start:"))
        self.start_frame_spinner = QSpinBox()
        self.start_frame_spinner.setMinimum(0)
        self.start_frame_spinner.setMaximum(0)
        self.start_frame_spinner.setValue(0)
        frame_range_layout.addWidget(self.start_frame_spinner)

        # End frame selector
        frame_range_layout.addWidget(QLabel("End:"))
        self.end_frame_spinner = QSpinBox()
        self.end_frame_spinner.setMinimum(0)
        self.end_frame_spinner.setMaximum(0)
        self.end_frame_spinner.setValue(0)
        frame_range_layout.addWidget(self.end_frame_spinner)

        # Full range checkbox
        self.full_range_checkbox = QCheckBox("Analyze All Frames")
        self.full_range_checkbox.setChecked(True)
        self.full_range_checkbox.stateChanged.connect(self.toggle_frame_range)
        frame_range_layout.addWidget(self.full_range_checkbox)

        frame_range_group.setLayout(frame_range_layout)
        analysis_layout.addWidget(frame_range_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        analysis_layout.addWidget(self.progress_bar)

        # Visualization options
        vis_layout = QHBoxLayout()

        # Display mode selection
        vis_layout.addWidget(QLabel("Display:"))
        self.display_mode = QComboBox()
        self.display_mode.addItem("Original")
        self.display_mode.addItem("Cell Mask")
        self.display_mode.addItem("Optical Flow")
        self.display_mode.addItem("Curvature-Motion")
        self.display_mode.addItem("Movement Classification")
        self.display_mode.currentIndexChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.display_mode)

        analysis_layout.addLayout(vis_layout)

        # Add checkboxes for overlay options
        overlay_layout = QHBoxLayout()
        self.show_vectors = QCheckBox("Show Vectors")
        self.show_vectors.setChecked(True)
        self.show_vectors.stateChanged.connect(self.update_visualization)
        overlay_layout.addWidget(self.show_vectors)

        self.show_contour = QCheckBox("Show Contour")
        self.show_contour.setChecked(True)
        self.show_contour.stateChanged.connect(self.update_visualization)
        overlay_layout.addWidget(self.show_contour)

        analysis_layout.addLayout(overlay_layout)

        analysis_group.setLayout(analysis_layout)
        control_layout.addWidget(analysis_group)

        # Export controls
        export_group = QGroupBox("Export Results")
        export_layout = QVBoxLayout()

        # Export buttons
        self.export_csv_button = QPushButton("Export Data to CSV")
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_csv_button.setEnabled(False)
        export_layout.addWidget(self.export_csv_button)

        self.export_figures_button = QPushButton("Export Figures")
        self.export_figures_button.clicked.connect(self.export_figures)
        self.export_figures_button.setEnabled(False)
        export_layout.addWidget(self.export_figures_button)

        export_group.setLayout(export_layout)
        control_layout.addWidget(export_group)

        # Add stretch to keep controls at the top
        control_layout.addStretch()

        return control_widget

    def _create_visualization_panel(self):
        """Create the visualization panel with multiple tabs."""
        vis_widget = QTabWidget()

        # Main visualization tab
        self.main_vis_tab = QWidget()
        main_vis_layout = QVBoxLayout(self.main_vis_tab)

        # Create matplotlib figure for main visualization
        self.main_fig = Figure(figsize=(8, 8), dpi=100)
        self.main_canvas = FigureCanvas(self.main_fig)
        main_vis_layout.addWidget(self.main_canvas)

        vis_widget.addTab(self.main_vis_tab, "Main View")

        # Correlation analysis tab
        self.corr_tab = QWidget()
        corr_layout = QVBoxLayout(self.corr_tab)

        # Create matplotlib figure for correlation plots
        self.corr_fig = Figure(figsize=(8, 8), dpi=100)
        self.corr_canvas = FigureCanvas(self.corr_fig)
        corr_layout.addWidget(self.corr_canvas)

        vis_widget.addTab(self.corr_tab, "Correlation Analysis")

        # Time series analysis tab
        self.time_tab = QWidget()
        time_layout = QVBoxLayout(self.time_tab)

        # Create matplotlib figure for time series
        self.time_fig = Figure(figsize=(8, 8), dpi=100)
        self.time_canvas = FigureCanvas(self.time_fig)
        time_layout.addWidget(self.time_canvas)

        vis_widget.addTab(self.time_tab, "Time Series")

        # Kymograph tab
        self.kymo_tab = QWidget()
        kymo_layout = QVBoxLayout(self.kymo_tab)

        # Controls for kymograph point selection
        kymo_control_layout = QHBoxLayout()
        kymo_control_layout.addWidget(QLabel("Point Index:"))

        self.kymo_point_spinner = QSpinBox()
        self.kymo_point_spinner.setMinimum(0)
        self.kymo_point_spinner.setMaximum(0)
        self.kymo_point_spinner.valueChanged.connect(self.update_kymograph)
        kymo_control_layout.addWidget(self.kymo_point_spinner)

        kymo_control_layout.addWidget(QLabel("Window Size:"))
        self.kymo_window_spinner = QSpinBox()
        self.kymo_window_spinner.setMinimum(1)
        self.kymo_window_spinner.setMaximum(50)
        self.kymo_window_spinner.setValue(10)
        self.kymo_window_spinner.valueChanged.connect(self.update_kymograph)
        kymo_control_layout.addWidget(self.kymo_window_spinner)

        self.update_kymo_button = QPushButton("Update Kymograph")
        self.update_kymo_button.clicked.connect(self.update_kymograph)
        self.update_kymo_button.setEnabled(False)
        kymo_control_layout.addWidget(self.update_kymo_button)

        kymo_layout.addLayout(kymo_control_layout)

        # Create matplotlib figure for kymograph
        self.kymo_fig = Figure(figsize=(8, 8), dpi=100)
        self.kymo_canvas = FigureCanvas(self.kymo_fig)
        kymo_layout.addWidget(self.kymo_canvas)

        vis_widget.addTab(self.kymo_tab, "Kymograph")

        return vis_widget

    def load_mask_sequence(self):
        """Load a sequence of cell mask images."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load First Cell Mask Image", "", "TIFF Files (*.tif *.tiff);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Determine if this is a multi-page TIFF
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)

            # Try to load as multi-page TIFF first
            try:
                import tifffile
                with tifffile.TiffFile(file_path) as tif:
                    num_pages = len(tif.pages)
                    if num_pages > 1:
                        # Load multi-page TIFF
                        self.statusBar().showMessage(f"Loading {num_pages} frames from multi-page TIFF...")
                        self.masks = []
                        for i in range(num_pages):
                            page = tif.pages[i]
                            self.masks.append(page.asarray())
                        self.mask_info_label.setText(f"Loaded: {filename} ({num_pages} frames)")
                    else:
                        # Single-page TIFF, search for sequence
                        self._load_image_sequence(file_path, is_mask=True)

            except (ImportError, Exception) as e:
                # Fall back to sequence loading
                self._load_image_sequence(file_path, is_mask=True)

            # Update UI for frame navigation
            self._update_frame_controls()

            # Enable run button if both masks and frames are loaded
            self.run_button.setEnabled(len(self.masks) > 0 and len(self.frames) > 0)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load mask sequence: {str(e)}")

    def load_fluorescence_sequence(self):
        """Load a sequence of fluorescence images."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load First Fluorescence Image", "", "TIFF Files (*.tif *.tiff);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Determine if this is a multi-page TIFF
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)

            # Try to load as multi-page TIFF first
            try:
                import tifffile
                with tifffile.TiffFile(file_path) as tif:
                    num_pages = len(tif.pages)
                    if num_pages > 1:
                        # Load multi-page TIFF
                        self.statusBar().showMessage(f"Loading {num_pages} frames from multi-page TIFF...")
                        self.frames = []
                        for i in range(num_pages):
                            page = tif.pages[i]
                            self.frames.append(page.asarray())
                        self.fluor_info_label.setText(f"Loaded: {filename} ({num_pages} frames)")
                    else:
                        # Single-page TIFF, search for sequence
                        self._load_image_sequence(file_path, is_mask=False)

            except (ImportError, Exception) as e:
                # Fall back to sequence loading
                self._load_image_sequence(file_path, is_mask=False)

            # Update UI for frame navigation
            self._update_frame_controls()

            # Enable run button if both masks and frames are loaded
            self.run_button.setEnabled(len(self.masks) > 0 and len(self.frames) > 0)

            # Update visualization
            self.update_visualization()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load fluorescence sequence: {str(e)}")

    def _load_image_sequence(self, first_file_path, is_mask=True):
        """Load a sequence of images starting with the given file."""
        directory = os.path.dirname(first_file_path)
        filename = os.path.basename(first_file_path)

        # Extract the numbering pattern from the filename
        import re
        match = re.search(r'(\d+)', filename)

        if match:
            # Get the numerical part and its position
            num_str = match.group(1)
            num_pos = match.start(1)
            prefix = filename[:num_pos]
            suffix = filename[num_pos + len(num_str):]
            num_digits = len(num_str)
            num_value = int(num_str)

            # Search for files with the same pattern
            sequence = []
            i = 0
            while True:
                curr_num = num_value + i
                curr_filename = f"{prefix}{curr_num:0{num_digits}d}{suffix}"
                curr_path = os.path.join(directory, curr_filename)

                if os.path.exists(curr_path):
                    img = cv2.imread(curr_path, cv2.IMREAD_ANYDEPTH)
                    sequence.append(img)
                    i += 1
                else:
                    break

            if is_mask:
                self.masks = sequence
                self.mask_info_label.setText(f"Loaded: {prefix}*{suffix} ({len(sequence)} frames)")
            else:
                self.frames = sequence
                self.fluor_info_label.setText(f"Loaded: {prefix}*{suffix} ({len(sequence)} frames)")
        else:
            # No number in filename, just load the single file
            img = cv2.imread(first_file_path, cv2.IMREAD_ANYDEPTH)

            if is_mask:
                self.masks = [img]
                self.mask_info_label.setText(f"Loaded: {filename} (1 frame)")
            else:
                self.frames = [img]
                self.fluor_info_label.setText(f"Loaded: {filename} (1 frame)")

    def _update_frame_controls(self):
        """Update the frame navigation controls based on loaded data."""
        num_frames = max(len(self.masks), len(self.frames))

        if num_frames > 0:
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(num_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)

            self.kymo_point_spinner.setEnabled(False)
            self.kymo_point_spinner.setValue(0)

            self.prev_button.setEnabled(False)  # Disabled at frame 0
            self.next_button.setEnabled(num_frames > 1)
            self.play_button.setEnabled(num_frames > 1)

            self.frame_label.setText(f"Frame: 1/{num_frames}")
            self.current_frame_idx = 0

            # Update main visualization
            self.update_visualization()
        else:
            self.frame_slider.setEnabled(False)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.frame_label.setText("Frame: 0/0")

    def update_parameters(self):
        """Update analysis parameters from UI controls."""
        self.pixel_size = self.pixel_size_input.value()
        self.time_delta = self.time_delta_input.value()

        # Update smoothing value display
        smoothing_value = self.smoothing_slider.value() / 10.0
        self.smoothing_value_label.setText(f"{smoothing_value:.1f}")

        # Update analyzers with new parameters
        self.dynamics_tracker = DynamicsTracker(
            contour_smoothing=smoothing_value,
            pixel_size=self.pixel_size
        )
        self.dynamics_tracker.set_time_delta(self.time_delta)

        self.optical_flow = OpticalFlowAnalyzer(
            pixel_size=self.pixel_size,
            time_delta=self.time_delta
        )

        self.correlation_analyzer = CorrelationAnalyzer(
            pixel_size=self.pixel_size
        )

        # If we already have analysis results, re-run with new parameters
        if len(self.motion_data) > 0:
            self.run_analysis()

    def change_frame(self, frame_idx):
        """Change the current frame being displayed."""
        if frame_idx == self.current_frame_idx:
            return

        self.current_frame_idx = frame_idx
        max_frames = max(len(self.masks), len(self.frames))

        if max_frames > 0:
            self.frame_label.setText(f"Frame: {frame_idx + 1}/{max_frames}")

            # Enable/disable navigation buttons
            self.prev_button.setEnabled(frame_idx > 0)
            self.next_button.setEnabled(frame_idx < max_frames - 1)

            # Update visualization
            self.update_visualization()

    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self.frame_slider.setValue(self.current_frame_idx - 1)

    def next_frame(self):
        """Go to next frame."""
        max_frames = max(len(self.masks), len(self.frames))
        if self.current_frame_idx < max_frames - 1:
            self.frame_slider.setValue(self.current_frame_idx + 1)

    def toggle_animation(self, checked):
        """Toggle animation playback."""
        if checked:
            # Start animation timer
            self.play_button.setText("Stop Animation")
            # TODO: Implement animation timer
        else:
            # Stop animation timer
            self.play_button.setText("Play Animation")
            # TODO: Stop animation timer

    def run_analysis(self):
        """Run the complete analysis pipeline on the image sequence."""
        if len(self.masks) == 0 or len(self.frames) == 0:
            QMessageBox.warning(self, "Warning", "Please load both mask and fluorescence sequences first.")
            return

        try:
            # Get frame range based on selection
            if self.full_range_checkbox.isChecked():
                start_frame = 0
                end_frame = min(len(self.masks), len(self.frames)) - 1
            else:
                start_frame = self.start_frame_spinner.value()
                end_frame = min(self.end_frame_spinner.value(),
                              min(len(self.masks), len(self.frames)) - 1)

                # Validate range
                if start_frame > end_frame:
                    QMessageBox.warning(self, "Warning", "Start frame cannot be greater than end frame.")
                    return

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Clear previous results
            self.contours = []
            self.curvatures = []
            self.motion_data = []
            self.correlation_data = []

            # Initialize dynamics tracker
            smoothing_value = self.smoothing_slider.value() / 10.0
            self.dynamics_tracker = DynamicsTracker(
                contour_smoothing=smoothing_value,
                pixel_size=self.pixel_size
            )
            self.dynamics_tracker.set_time_delta(self.time_delta)

            # Calculate total frames to process
            total_frames = end_frame - start_frame + 1

            # Process each frame in the selected range
            for i in range(start_frame, end_frame + 1):
                # Update progress based on frames in range
                progress = int(((i - start_frame) / total_frames) * 100)
                self.progress_bar.setValue(progress)
                self.statusBar().showMessage(f"Processing frame {i+1}/{end_frame+1}...")
                QApplication.processEvents()  # Keep UI responsive

                # Get current frame and mask
                mask = self.masks[i].copy()
                frame = self.frames[i].copy()

                # Make sure mask is binary
                if mask.max() > 1:
                    mask = (mask > 0).astype(np.uint8) * 255

                # Detect cell edge
                contour = self._detect_cell_edge(mask)
                if contour is None or len(contour) < 5:
                    self.statusBar().showMessage(f"Failed to detect edge in frame {i+1}")
                    continue

                self.contours.append(contour)

                # Calculate curvature
                curvature = self._calculate_curvature(contour)
                self.curvatures.append(curvature)

                # Track membrane dynamics
                if i == start_frame:
                    # First frame, just store it
                    motion = self.dynamics_tracker.analyze_frame(frame, contour)
                    self.motion_data.append(None)  # No motion data for first frame
                else:
                    # Calculate dynamics between previous and current frame
                    motion = self.dynamics_tracker.analyze_frame(frame, contour)
                    self.motion_data.append(motion)

                    # Calculate correlation between curvature and motion
                    if motion is not None and len(curvature) == len(contour):
                        # Make sure we're using the same number of points for both analyses
                        min_length = min(len(curvature), len(motion.velocities))

                        dynamics_data = {
                            'velocities': motion.velocities[:min_length],
                            'normal_components': np.array([np.dot(v, n) for v, n in zip(motion.velocities[:min_length], self._calculate_normals(contour)[:min_length])]),
                            'tangential_components': np.array([np.dot(v, [-n[1], n[0]]) for v, n in zip(motion.velocities[:min_length], self._calculate_normals(contour)[:min_length])]),
                            'classifications': motion.classifications[:min_length],
                            'points': contour[:min_length]
                        }

                        # Ensure curvature data is also truncated to match
                        curvature = curvature[:min_length]

                        corr_data = self.correlation_analyzer.analyze_correlation(
                            curvature, dynamics_data, i
                        )
                        self.correlation_data.append(corr_data)

            # Analysis complete
            self.progress_bar.setValue(100)
            self.statusBar().showMessage(f"Analysis complete. Processed frames {start_frame+1} to {end_frame+1} ({total_frames} frames total).")

            # Enable export buttons
            self.export_csv_button.setEnabled(True)
            self.export_figures_button.setEnabled(True)

            # Enable kymograph controls
            self.update_kymo_button.setEnabled(True)
            if len(self.contours) > 0 and len(self.contours[0]) > 0:
                self.kymo_point_spinner.setMaximum(len(self.contours[0]) - 1)
                self.kymo_point_spinner.setEnabled(True)

            # Update all visualizations
            self.update_visualization()
            self.update_correlation_plot()
            self.update_time_series()
            self.update_kymograph()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def _detect_cell_edge(self, mask):
        """Detect cell edge from binary mask."""
        # Use the EdgeDetector to find and process the edge
        edge_data = self.edge_detector.detect_edge(mask)

        # Return the contour (may be None if detection failed)
        return edge_data

    def _calculate_curvature(self, contour):
        """Calculate curvature along the contour."""
        n_points = len(contour)
        curvatures = np.zeros(n_points)

        for i in range(n_points):
            # Use points before and after for curvature calculation
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points

            # Get points
            prev = contour[prev_idx]
            curr = contour[i]
            next_pt = contour[next_idx]

            # Calculate curvature using finite differences
            try:
                dx1 = curr[0] - prev[0]
                dy1 = curr[1] - prev[1]
                dx2 = next_pt[0] - curr[0]
                dy2 = next_pt[1] - curr[1]

                # First derivatives
                dx = (dx1 + dx2) / 2
                dy = (dy1 + dy2) / 2

                # Second derivatives
                d2x = dx2 - dx1
                d2y = dy2 - dy1

                # Curvature formula
                num = dx * d2y - dy * d2x
                denom = (dx**2 + dy**2)**(1.5)

                if abs(denom) > 1e-10:
                    # Curvature with sign (negative = convex, positive = concave)
                    # Convert to nm^-1
                    curvatures[i] = -num / denom / self.pixel_size
                else:
                    curvatures[i] = 0
            except:
                curvatures[i] = 0

        return curvatures

    def _calculate_normals(self, contour):
        """Calculate normal vectors along contour."""
        n_points = len(contour)
        normals = np.zeros((n_points, 2))

        for i in range(n_points):
            # Use neighboring points
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points

            # Calculate tangent using central difference
            tangent = contour[next_idx] - contour[prev_idx]
            tangent_norm = np.linalg.norm(tangent)

            if tangent_norm > 0:
                tangent = tangent / tangent_norm

                # Normal is perpendicular to tangent
                normal = np.array([-tangent[1], tangent[0]])

                # Ensure normal points inward (approximate)
                center = np.mean(contour, axis=0)
                to_center = center - contour[i]
                to_center_norm = np.linalg.norm(to_center)

                if to_center_norm > 0:
                    to_center = to_center / to_center_norm
                    if np.dot(normal, to_center) < 0:
                        normal = -normal

                normals[i] = normal
            else:
                # Fallback for degenerate case
                normals[i] = np.array([0, 0])

        return normals

    def update_visualization(self):
        """Update the main visualization based on current display mode."""
        if len(self.frames) == 0:
            return

        # Get current frame
        frame_idx = min(self.current_frame_idx, len(self.frames) - 1)
        frame = self.frames[frame_idx]

        # Convert frame for display if needed
        if frame.dtype != np.uint8:
            # Normalize and convert to 8-bit
            frame_disp = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        else:
            frame_disp = frame.copy()

        # Convert to RGB if grayscale
        if len(frame_disp.shape) == 2:
            frame_disp = cv2.cvtColor(frame_disp, cv2.COLOR_GRAY2BGR)

        # Get current mask if available
        mask = None
        if len(self.masks) > frame_idx:
            mask = self.masks[frame_idx]

        # Get display mode
        mode = self.display_mode.currentText()

        # Clear figure
        self.main_fig.clear()
        ax = self.main_fig.add_subplot(111)

        # Display based on mode
        if mode == "Original":
            # Show original frame
            ax.imshow(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
            title = "Original Frame"

            # Overlay contour if available and option enabled
            if self.show_contour.isChecked() and frame_idx < len(self.contours):
                contour = self.contours[frame_idx]
                ax.plot(contour[:, 0], contour[:, 1], 'y-', linewidth=2, alpha=0.8)

        elif mode == "Cell Mask":
            # Show mask
            if mask is not None:
                ax.imshow(mask, cmap='gray')
                title = "Cell Mask"

                # Overlay contour
                if self.show_contour.isChecked() and frame_idx < len(self.contours):
                    contour = self.contours[frame_idx]
                    ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
            else:
                ax.imshow(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
                title = "No Mask Available"

        elif mode == "Optical Flow":
            # Show optical flow visualization
            if frame_idx > 0 and frame_idx < len(self.frames):
                # Get previous and current frames
                prev_frame = self.frames[frame_idx - 1]
                curr_frame = self.frames[frame_idx]

                # Show background image instead of flow visualization
                ax.imshow(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
                title = "Optical Flow"

                # Overlay contour
                if self.show_contour.isChecked() and frame_idx < len(self.contours):
                    contour = self.contours[frame_idx]
                    ax.plot(contour[:, 0], contour[:, 1], 'y-', linewidth=2, alpha=0.8)

                # ONLY show flow vectors along contour if requested
                if self.show_vectors.isChecked() and frame_idx < len(self.contours) and frame_idx < len(self.motion_data):
                    motion = self.motion_data[frame_idx]
                    if motion is not None:
                        # Draw motion vectors
                        contour = self.contours[frame_idx]
                        scale = 5.0  # Scale factor for vector display

                        # Define a border margin to exclude
                        border_margin = 40
                        height, width = frame_disp.shape[:2]

                        for i, point in enumerate(contour):
                            x, y = point.astype(int)

                            # Skip points near the image border
                            if (x < border_margin or x >= width - border_margin or
                                y < border_margin or y >= height - border_margin):
                                continue

                            # Only draw every few points for clarity
                            if i % 5 != 0:
                                continue

                            # Bounds checking
                            if i < len(motion.velocities):
                                vx, vy = motion.velocities[i]

                                # Determine color based on classification
                                class_color = 'y'  # Default color
                                if i < len(motion.classifications):
                                    class_color = {
                                        'expanding': 'r',
                                        'retracting': 'b',
                                        'stationary': 'gray',
                                        'flowing': 'g'
                                    }.get(motion.classifications[i], 'y')

                                # Only draw significant vectors
                                if np.sqrt(vx*vx + vy*vy) > 1e-6:
                                    ax.arrow(
                                        x, y, vx*scale, vy*scale,
                                        head_width=5, head_length=7,
                                        fc=class_color, ec=class_color,
                                        alpha=0.8
                                    )
            else:
                ax.imshow(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
                title = "No Optical Flow Available"

        elif mode == "Curvature-Motion":
            # Show curvature-motion correlation visualization
            ax.imshow(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
            title = "Curvature-Motion Analysis"

            if frame_idx < len(self.contours) and frame_idx < len(self.curvatures):
                contour = self.contours[frame_idx]
                curvatures = self.curvatures[frame_idx]

                # Create colormap for curvature
                from matplotlib.colors import LinearSegmentedColormap
                cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'white', 'red'])

                # Normalize curvature for coloring
                if len(curvatures) > 0 and np.max(np.abs(curvatures)) > 0:
                    norm = plt.Normalize(-np.max(np.abs(curvatures)), np.max(np.abs(curvatures)))

                    # Define a border margin to exclude
                    border_margin = 40
                    height, width = frame_disp.shape[:2]

                    # Draw contour segments with curvature coloring
                    for i in range(len(contour) - 1):
                        # Skip points near the image border
                        x1, y1 = contour[i].astype(int)
                        x2, y2 = contour[i+1].astype(int)

                        if (x1 < border_margin or x1 >= width - border_margin or
                            y1 < border_margin or y1 >= height - border_margin or
                            x2 < border_margin or x2 >= width - border_margin or
                            y2 < border_margin or y2 >= height - border_margin):
                            continue

                        color = cmap(norm(curvatures[i]))
                        ax.plot(
                            [contour[i, 0], contour[i+1, 0]],
                            [contour[i, 1], contour[i+1, 1]],
                            color=color, linewidth=3
                        )

                    # Add colorbar
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax)
                    cbar.set_label('Curvature (nm⁻¹)')

                    # Overlay motion vectors if requested
                    if self.show_vectors.isChecked() and frame_idx < len(self.motion_data):
                        motion = self.motion_data[frame_idx]
                        if motion is not None:
                            # Draw motion vectors
                            scale = 5.0
                            for i, point in enumerate(contour):
                                if i % 5 != 0:  # Display every 5th point for clarity
                                    continue

                                x, y = point.astype(int)

                                # Skip points near the image border
                                if (x < border_margin or x >= width - border_margin or
                                    y < border_margin or y >= height - border_margin):
                                    continue

                                # Bounds checking
                                if i < len(motion.velocities):
                                    vx, vy = motion.velocities[i]

                                    # Draw arrow
                                    if np.sqrt(vx*vx + vy*vy) > 1e-6:
                                        ax.arrow(
                                            x, y, vx*scale, vy*scale,
                                            head_width=5, head_length=7,
                                            fc='y', ec='y',
                                            alpha=0.8
                                        )

        elif mode == "Movement Classification":
            # Show movement classification visualization
            ax.imshow(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
            title = "Movement Classification"

            if frame_idx < len(self.contours) and frame_idx < len(self.motion_data) and self.motion_data[frame_idx] is not None:
                contour = self.contours[frame_idx]
                motion = self.motion_data[frame_idx]

                # Color based on classification
                class_colors = {
                    'expanding': 'red',
                    'retracting': 'blue',
                    'stationary': 'gray',
                    'flowing': 'green'
                }

                # Define a border margin to exclude
                border_margin = 40
                height, width = frame_disp.shape[:2]

                # Draw contour segments with classification coloring
                for i in range(len(contour) - 1):
                    # Skip points near the image border
                    x1, y1 = contour[i].astype(int)
                    x2, y2 = contour[i+1].astype(int)

                    if (x1 < border_margin or x1 >= width - border_margin or
                        y1 < border_margin or y1 >= height - border_margin or
                        x2 < border_margin or x2 >= width - border_margin or
                        y2 < border_margin or y2 >= height - border_margin):
                        continue

                    # Bounds checking
                    if i < len(motion.classifications):
                        color = class_colors.get(motion.classifications[i], 'yellow')
                        ax.plot(
                            [contour[i, 0], contour[i+1, 0]],
                            [contour[i, 1], contour[i+1, 1]],
                            color=color, linewidth=3
                        )

                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='red', lw=3, label='Expanding'),
                    Line2D([0], [0], color='blue', lw=3, label='Retracting'),
                    Line2D([0], [0], color='green', lw=3, label='Flowing'),
                    Line2D([0], [0], color='gray', lw=3, label='Stationary')
                ]
                ax.legend(handles=legend_elements, loc='upper right')

                # Overlay motion vectors if requested
                if self.show_vectors.isChecked():
                    scale = 5.0
                    for i, point in enumerate(contour):
                        if i % 5 != 0:  # Display every 5th point for clarity
                            continue

                        x, y = point.astype(int)

                        # Skip points near the image border
                        if (x < border_margin or x >= width - border_margin or
                            y < border_margin or y >= height - border_margin):
                            continue

                        # Bounds checking
                        if i < len(motion.velocities) and i < len(motion.classifications):
                            vx, vy = motion.velocities[i]
                            color = class_colors.get(motion.classifications[i], 'yellow')

                            # Draw arrow
                            if np.sqrt(vx*vx + vy*vy) > 1e-6:
                                ax.arrow(
                                    x, y, vx*scale, vy*scale,
                                    head_width=5, head_length=7,
                                    fc=color, ec=color,
                                    alpha=0.8
                                )

        # Set title and adjust display
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        self.main_fig.tight_layout()

        # Refresh canvas
        self.main_canvas.draw()

    def update_correlation_plot(self):
        """Update correlation analysis plot."""
        # Check if we have correlation data
        if len(self.correlation_data) == 0:
            return

        # Get correlation data for current frame
        frame_idx = min(self.current_frame_idx, len(self.correlation_data))

        # Use the most recent available correlation data if current frame doesn't have it
        corr_idx = frame_idx
        while corr_idx >= 0 and (corr_idx >= len(self.correlation_data) or self.correlation_data[corr_idx] is None):
            corr_idx -= 1

        if corr_idx < 0:
            return

        corr_data = self.correlation_data[corr_idx]

        # Create correlation visualization
        fig = self.correlation_analyzer.create_correlation_visualization(corr_data)

        # Update canvas
        self.corr_fig.clear()
        for i, ax in enumerate(fig.axes):
            # Copy each axis to our figure
            self.corr_fig.add_subplot(2, 2, i+1)
            self.corr_fig.axes[i].clear()
            self.corr_fig.axes[i].sharex(ax)
            self.corr_fig.axes[i].sharey(ax)

            # Copy content
            for line in ax.lines:
                self.corr_fig.axes[i].plot(line.get_xdata(), line.get_ydata(),
                                         linestyle=line.get_linestyle(),
                                         color=line.get_color(),
                                         linewidth=line.get_linewidth(),
                                         marker=line.get_marker())

            # For scatter plots in collections
            for collection in ax.collections:
                if hasattr(collection, 'get_offsets'):
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        # Try to get colors and sizes if available
                        try:
                            colors = collection.get_facecolors()
                            if len(colors) == 1:  # Single color for all points
                                self.corr_fig.axes[i].scatter(
                                    offsets[:, 0], offsets[:, 1],
                                    color=colors[0],
                                    alpha=collection.get_alpha() if hasattr(collection, 'get_alpha') else None
                                )
                            else:
                                self.corr_fig.axes[i].scatter(
                                    offsets[:, 0], offsets[:, 1],
                                    c=colors,
                                    alpha=collection.get_alpha() if hasattr(collection, 'get_alpha') else None
                                )
                        except:
                            # If we can't get colors, just plot with default color
                            self.corr_fig.axes[i].scatter(
                                offsets[:, 0], offsets[:, 1],
                                alpha=collection.get_alpha() if hasattr(collection, 'get_alpha') else None
                            )

            # Copy labels and title
            self.corr_fig.axes[i].set_xlabel(ax.get_xlabel())
            self.corr_fig.axes[i].set_ylabel(ax.get_ylabel())
            self.corr_fig.axes[i].set_title(ax.get_title())

            # Copy legend only if there are labeled artists
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                self.corr_fig.axes[i].legend()

            # Copy grid settings (correctly checking grid lines)
            # Check if either x or y grid lines are visible
            has_grid = False
            for grid_line in ax.xaxis.get_gridlines():
                if grid_line.get_visible():
                    has_grid = True
                    break
            if not has_grid:
                for grid_line in ax.yaxis.get_gridlines():
                    if grid_line.get_visible():
                        has_grid = True
                        break

            self.corr_fig.axes[i].grid(has_grid)

        self.corr_fig.tight_layout()
        self.corr_canvas.draw()

    def update_time_series(self):
        """Update time series analysis plot."""
        if len(self.correlation_data) < 2:
            return

        # Clear figure
        self.time_fig.clear()

        # Create subplots
        ax1 = self.time_fig.add_subplot(311)  # Correlation over time
        ax2 = self.time_fig.add_subplot(312)  # Mean curvature over time
        ax3 = self.time_fig.add_subplot(313)  # Movement classification over time

        # Get temporal data
        frames = []
        correlations = []
        mean_curvatures = []
        mean_velocities = []
        class_percentages = {'expanding': [], 'retracting': [], 'stationary': [], 'flowing': []}

        for i, corr_data in enumerate(self.correlation_data):
            if corr_data is not None:
                frames.append(i)
                correlations.append(corr_data.correlation)
                mean_curvatures.append(np.mean(corr_data.curvatures))
                mean_velocities.append(np.mean(np.abs(corr_data.normal_velocities)))

                # Calculate percentage of each movement classification
                for cls in class_percentages.keys():
                    count = corr_data.classifications.count(cls)
                    percentage = 100 * count / len(corr_data.classifications)
                    class_percentages[cls].append(percentage)

        # Plot correlation vs time
        ax1.plot(frames, correlations, 'b-o', linewidth=2)
        ax1.set_ylabel('Correlation Coefficient')
        ax1.set_title('Curvature-Motion Correlation vs Time')
        ax1.grid(True, alpha=0.3)

        # Add horizontal line at zero
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Plot mean curvature and velocity over time
        ax2.plot(frames, mean_curvatures, 'r-', linewidth=2, label='Curvature (nm⁻¹)')

        # Add second axis for velocity
        ax2_twin = ax2.twinx()
        ax2_twin.plot(frames, mean_velocities, 'g--', linewidth=2, label='Velocity (nm/s)')

        ax2.set_ylabel('Mean Curvature (nm⁻¹)', color='r')
        ax2_twin.set_ylabel('Mean Velocity (nm/s)', color='g')
        ax2.set_title('Mean Curvature and Velocity vs Time')
        ax2.grid(True, alpha=0.3)

        # Plot movement classification percentages
        # Fix: Use proper color names and separate line style
        colors = {
            'expanding': 'red',
            'retracting': 'blue',
            'stationary': 'black',  # Changed from 'gray' to 'black'
            'flowing': 'green'
        }

        linestyles = {
            'expanding': '-',
            'retracting': '-',
            'stationary': '-',
            'flowing': '-'
        }

        for cls, percentages in class_percentages.items():
            # Fix: Pass color and linestyle separately
            ax3.plot(frames, percentages, color=colors[cls], linestyle=linestyles[cls],
                    linewidth=2, label=cls.capitalize())

        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Movement Classification vs Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Highlight current frame
        if self.current_frame_idx in frames:
            idx = frames.index(self.current_frame_idx)
            ax1.axvline(x=frames[idx], color='k', linestyle=':', alpha=0.5)
            ax2.axvline(x=frames[idx], color='k', linestyle=':', alpha=0.5)
            ax3.axvline(x=frames[idx], color='k', linestyle=':', alpha=0.5)

        self.time_fig.tight_layout()
        self.time_canvas.draw()

    def update_kymograph(self):
        """Update kymograph visualization."""
        if len(self.motion_data) < 2 or self.dynamics_tracker.motion_history is None:
            return

        # Get point index and window size
        point_idx = self.kymo_point_spinner.value()
        window_size = self.kymo_window_spinner.value()

        # Create kymograph
        kymograph = self.dynamics_tracker.create_kymograph(point_idx, window_size)

        if kymograph is None:
            return

        # Clear figure
        self.kymo_fig.clear()

        # Create two subplots
        # First for kymograph, second for location reference
        gs = self.kymo_fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = self.kymo_fig.add_subplot(gs[0])
        ax2 = self.kymo_fig.add_subplot(gs[1])

        # Plot kymograph
        im = ax1.imshow(kymograph, aspect='auto', origin='upper',
                      extent=[0, kymograph.shape[1], kymograph.shape[0], 0])

        ax1.set_title('Membrane Movement Kymograph')
        ax1.set_ylabel('Frame')
        ax1.set_xlabel('Position along membrane')

        # Add colorbar legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='k', label='Expanding'),
            Patch(facecolor='blue', edgecolor='k', label='Retracting'),
            Patch(facecolor='green', edgecolor='k', label='Flowing'),
            Patch(facecolor='gray', edgecolor='k', label='Stationary')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Add reference line for current frame
        ax1.axhline(y=self.current_frame_idx, color='w', linestyle='--')

        # Show reference location on cell contour
        if len(self.contours) > 0 and self.current_frame_idx < len(self.contours):
            contour = self.contours[self.current_frame_idx]
            ax2.plot(contour[:, 0], contour[:, 1], 'b-', alpha=0.7)

            # Highlight selected region
            window_start = (point_idx - window_size) % len(contour)
            window_end = (point_idx + window_size) % len(contour)

            # Handle wrap-around
            if window_start < window_end:
                highlight_points = contour[window_start:window_end+1]
            else:
                highlight_points = np.vstack((contour[window_start:], contour[:window_end+1]))

            ax2.plot(highlight_points[:, 0], highlight_points[:, 1], 'r-', linewidth=3)

            # Mark center point
            center_point = contour[point_idx]
            ax2.plot(center_point[0], center_point[1], 'ro', markersize=8)

            ax2.set_title(f'Selected Region (Point {point_idx})')
            ax2.set_aspect('equal')
            ax2.set_xticks([])
            ax2.set_yticks([])

        self.kymo_fig.tight_layout()
        self.kymo_canvas.draw()

    def export_csv(self):
        """Export analysis data to CSV file."""
        try:
            import pandas as pd

            # Ask for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", "", "CSV Files (*.csv)"
            )

            if not file_path:
                return

            # Create DataFrame with all analysis data
            data = []

            for frame_idx in range(max(len(self.contours), len(self.curvatures))):
                # Get frame data if available
                contour = None if frame_idx >= len(self.contours) else self.contours[frame_idx]
                curvature = None if frame_idx >= len(self.curvatures) else self.curvatures[frame_idx]
                motion = None if frame_idx >= len(self.motion_data) else self.motion_data[frame_idx]

                if contour is not None and curvature is not None:
                    for i in range(len(contour)):
                        row = {
                            'frame': frame_idx,
                            'point_idx': i,
                            'x': contour[i, 0],
                            'y': contour[i, 1],
                            'curvature': curvature[i]
                        }

                        # Add motion data if available
                        if motion is not None and i < len(motion.velocities):
                            row.update({
                                'velocity_x': motion.velocities[i, 0],
                                'velocity_y': motion.velocities[i, 1],
                                'velocity_magnitude': motion.magnitudes[i],
                                'classification': motion.classifications[i]
                            })

                        data.append(row)

            # Convert to DataFrame and save
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)

            self.statusBar().showMessage(f"Data exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")

    def export_figures(self):
        """Export analysis figures."""
        try:
            # Ask for directory
            directory = QFileDialog.getExistingDirectory(
                self, "Select Export Directory"
            )

            if not directory:
                return

            # Export current main view
            self.main_fig.savefig(
                f"{directory}/main_view_{self.current_frame_idx}.png",
                dpi=300, bbox_inches='tight'
            )

            # Export correlation plot
            self.corr_fig.savefig(
                f"{directory}/correlation_{self.current_frame_idx}.png",
                dpi=300, bbox_inches='tight'
            )

            # Export time series plot
            self.time_fig.savefig(
                f"{directory}/time_series.png",
                dpi=300, bbox_inches='tight'
            )

            # Export kymograph
            self.kymo_fig.savefig(
                f"{directory}/kymograph.png",
                dpi=300, bbox_inches='tight'
            )

            self.statusBar().showMessage(f"Figures exported to {directory}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export figures: {str(e)}")


    def toggle_frame_range(self, state):
        """Toggle frame range controls based on checkbox state."""
        enabled = not bool(state)  # True if unchecked, False if checked
        self.start_frame_spinner.setEnabled(enabled)
        self.end_frame_spinner.setEnabled(enabled)

    # Update the _update_frame_controls method to set the range for start/end spinners
    def _update_frame_controls(self):
        """Update the frame navigation controls based on loaded data."""
        num_frames = max(len(self.masks), len(self.frames))

        if num_frames > 0:
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(num_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)

            # Update frame range spinners for batch processing
            self.start_frame_spinner.setMaximum(num_frames - 1)
            self.end_frame_spinner.setMaximum(num_frames - 1)
            self.end_frame_spinner.setValue(num_frames - 1)

            self.kymo_point_spinner.setEnabled(False)
            self.kymo_point_spinner.setValue(0)

            self.prev_button.setEnabled(False)  # Disabled at frame 0
            self.next_button.setEnabled(num_frames > 1)
            self.play_button.setEnabled(num_frames > 1)

            self.frame_label.setText(f"Frame: 1/{num_frames}")
            self.current_frame_idx = 0

            # Update main visualization
            self.update_visualization()
        else:
            self.frame_slider.setEnabled(False)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.frame_label.setText("Frame: 0/0")


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
