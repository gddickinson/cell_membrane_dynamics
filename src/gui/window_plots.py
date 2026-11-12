#!/usr/bin/env python3
# src/gui/window_plots.py
"""Correlation, time-series and kymograph plots for MainWindow (main.py).

Mixin class; its methods use the attributes MainWindow.__init__ sets up.
"""

import numpy as np



class PlotsMixin:
    """Correlation, time-series and kymograph plots for MainWindow (main.py)."""

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
        ax2_twin.plot(frames, mean_velocities, 'g--', linewidth=2, label='Velocity (px/s)')

        ax2.set_ylabel('Mean Curvature (nm⁻¹)', color='r')
        ax2_twin.set_ylabel('Mean Velocity (px/s)', color='g')
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
