#!/usr/bin/env python3
# src/gui/window_io.py
"""Loading mask / PIEZO1 sequences and exporting CSV / figures for MainWindow (main.py).

Mixin class; its methods use the attributes MainWindow.__init__ sets up.
"""

import os
import cv2
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from src.utils.image_utils import load_tiff_stack


class IOMixin:
    """Loading mask / PIEZO1 sequences and exporting CSV / figures for MainWindow (main.py)."""

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

        # A multi-page TIFF (e.g. a Micro-Manager MMStack) holds the whole sequence: load every page
        # (cv2.imread below reads only the first page).
        stack = load_tiff_stack(first_file_path)
        if stack is not None and len(stack) > 1:
            if is_mask:
                self.masks = stack
                self.mask_info_label.setText(f"Loaded: {filename} ({len(stack)} frames)")
            else:
                self.frames = stack
                self.fluor_info_label.setText(f"Loaded: {filename} ({len(stack)} frames)")
            return

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
