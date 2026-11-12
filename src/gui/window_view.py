#!/usr/bin/env python3
# src/gui/window_view.py
"""The main frame view (display modes) for MainWindow (main.py).

Mixin class; its methods use the attributes MainWindow.__init__ sets up.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt



class ViewMixin:
    """The main frame view (display modes) for MainWindow (main.py)."""

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
