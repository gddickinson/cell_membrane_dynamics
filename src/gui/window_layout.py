#!/usr/bin/env python3
# src/gui/window_layout.py
"""Control and visualisation panel construction for MainWindow (main.py).

Mixin class; its methods use the attributes MainWindow.__init__ sets up.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox, QTabWidget, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox, QProgressBar
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



class LayoutMixin:
    """Control and visualisation panel construction for MainWindow (main.py)."""

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
