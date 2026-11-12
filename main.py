#!/usr/bin/env python3
# main.py

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox

from src.analysis.edge_detection import EdgeDetector
from src.analysis.dynamics_tracker import DynamicsTracker
from src.analysis.optical_flow import OpticalFlowAnalyzer
from src.analysis.correlation_analyzer import CorrelationAnalyzer
from src.analysis.curvature_analyzer import CurvatureAnalyzer
from src.analysis.pipeline import analyze_sequence, finite_difference_curvature, contour_normals
from src.gui.window_layout import LayoutMixin
from src.gui.window_io import IOMixin
from src.gui.window_view import ViewMixin
from src.gui.window_plots import PlotsMixin

class MainWindow(LayoutMixin, IOMixin, ViewMixin, PlotsMixin, QMainWindow):
    """Main window for membrane dynamics analysis application.

    UI construction, file I/O, the frame view and the plots live in the mixins in src/gui/window_*.py.
    """

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

    def update_parameters(self):
        """Update analysis parameters from UI controls."""
        self.pixel_size = self.pixel_size_input.value()
        self.time_delta = self.time_delta_input.value()

        # Update smoothing value display
        smoothing_value = self.smoothing_slider.value() / 10.0
        self.smoothing_value_label.setText(f"{smoothing_value:.1f}")
        # The slider sets the Gaussian sigma (in contour points) used to smooth the detected edge
        self.edge_detector.smoothing_sigma = smoothing_value

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

            # Run the shared, Qt-free pipeline (src/analysis/pipeline.py)
            smoothing_value = self.smoothing_slider.value() / 10.0
            total_frames = end_frame - start_frame + 1

            def report(i, first, last):
                self.progress_bar.setValue(int(((i - first) / total_frames) * 100))
                self.statusBar().showMessage(f"Processing frame {i+1}/{last+1}...")
                QApplication.processEvents()  # Keep UI responsive

            result = analyze_sequence(
                self.masks, self.frames, pixel_size=self.pixel_size, time_delta=self.time_delta,
                smoothing=smoothing_value, start=start_frame, end=end_frame, progress=report,
                edge_detector=self.edge_detector)
            self.contours = result['contours']
            self.curvatures = result['curvatures']
            self.motion_data = result['motion_data']
            self.correlation_data = result['correlation_data']
            self.dynamics_tracker = result['tracker']
            if result['failed']:
                print(f"No edge detected in frame(s): {[i + 1 for i in result['failed']]}")

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
        """Calculate curvature (nm^-1) along the contour."""
        return finite_difference_curvature(contour, self.pixel_size)

    def _calculate_normals(self, contour):
        """Calculate normal vectors along contour."""
        return contour_normals(contour)

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
