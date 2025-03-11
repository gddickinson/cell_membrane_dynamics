#!/usr/bin/env python3
# src/gui/visualization_panel.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox
from typing import Optional, List, Dict, Tuple

class VisualizationPanel(QWidget):
    """Panel for data visualization."""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        
        # Create custom colormap for curvature
        self.curvature_cmap = LinearSegmentedColormap.from_list(
            'curvature', ['blue', 'white', 'red']
        )
        
        # Create colormap for movement classification
        self.movement_cmap = {
            'expanding': 'red',
            'retracting': 'blue',
            'flowing': 'green',
            'stationary': 'gray'
        }
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure for visualization
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        # Control panel for visualization options
        control_layout = QHBoxLayout()
        
        # Display mode selector
        control_layout.addWidget(QLabel("Display:"))
        self.display_mode = QComboBox()
        self.display_mode.addItems(["Original", "Cell Mask", "Optical Flow", 
                                  "Curvature-Motion", "Movement Classification"])
        self.display_mode.currentIndexChanged.connect(self.update_display)
        control_layout.addWidget(self.display_mode)
        
        # Checkbox for showing vectors
        self.show_vectors = QCheckBox("Show Vectors")
        self.show_vectors.setChecked(True)
        self.show_vectors.stateChanged.connect(self.update_display)
        control_layout.addWidget(self.show_vectors)
        
        # Checkbox for showing contour
        self.show_contour = QCheckBox("Show Contour")
        self.show_contour.setChecked(True)
        self.show_contour.stateChanged.connect(self.update_display)
        control_layout.addWidget(self.show_contour)
        
        layout.addLayout(control_layout)
        
    def plot_original(self, image: np.ndarray, contour: Optional[np.ndarray] = None, 
                     title: str = "Original Image"):
        """Plot original image with optional contour overlay.
        
        Args:
            image: Image to display
            contour: Optional contour points
            title: Plot title
        """
        # Clear figure
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Show image
        ax.imshow(image, cmap='gray')
        
        # Overlay contour if provided and option enabled
        if contour is not None and self.show_contour.isChecked():
            ax.plot(contour[:, 0], contour[:, 1], 'y-', linewidth=2, alpha=0.8)
            
        # Set title and hide axes
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Update canvas
        self.canvas.draw()
        
    def plot_curvature(self, image: np.ndarray, contour: np.ndarray, 
                      curvatures: np.ndarray, vectors: Optional[np.ndarray] = None,
                      title: str = "Curvature Analysis"):
        """Plot image with curvature-colored contour.
        
        Args:
            image: Background image
            contour: Contour points
            curvatures: Curvature values
            vectors: Optional velocity vectors
            title: Plot title
        """
        # Clear figure
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Show background image
        ax.imshow(image, cmap='gray')
        
        # Create colormap for curvature
        if len(curvatures) > 0 and np.max(np.abs(curvatures)) > 0:
            norm = plt.Normalize(
                -np.max(np.abs(curvatures)),
                np.max(np.abs(curvatures))
            )
            
            # Draw contour segments with curvature coloring
            for i in range(len(contour) - 1):
                color = self.curvature_cmap(norm(curvatures[i]))
                ax.plot(
                    [contour[i, 0], contour[i+1, 0]],
                    [contour[i, 1], contour[i+1, 1]],
                    color=color, linewidth=3
                )
            
            # Connect last and first point
            color = self.curvature_cmap(norm(curvatures[-1]))
            ax.plot(
                [contour[-1, 0], contour[0, 0]],
                [contour[-1, 1], contour[0, 1]],
                color=color, linewidth=3
            )
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=self.curvature_cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Curvature (nm⁻¹)')
        
        # Overlay vectors if provided and option enabled
        if vectors is not None and self.show_vectors.isChecked():
            self._overlay_vectors(ax, contour, vectors)
            
        # Set title and hide axes
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Update canvas
        self.canvas.draw()
        
    def plot_movement(self, image: np.ndarray, contour: np.ndarray, 
                     classifications: List[str], vectors: Optional[np.ndarray] = None,
                     title: str = "Movement Classification"):
        """Plot image with movement-classified contour.
        
        Args:
            image: Background image
            contour: Contour points
            classifications: Movement classifications
            vectors: Optional velocity vectors
            title: Plot title
        """
        # Clear figure
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Show background image
        ax.imshow(image, cmap='gray')
        
        # Draw contour segments with classification coloring
        for i in range(len(contour) - 1):
            color = self.movement_cmap.get(classifications[i], 'yellow')
            ax.plot(
                [contour[i, 0], contour[i+1, 0]],
                [contour[i, 1], contour[i+1, 1]],
                color=color, linewidth=3
            )
        
        # Connect last and first point
        color = self.movement_cmap.get(classifications[-1], 'yellow')
        ax.plot(
            [contour[-1, 0], contour[0, 0]],
            [contour[-1, 1], contour[0, 1]],
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
        
        # Overlay vectors if provided and option enabled
        if vectors is not None and self.show_vectors.isChecked():
            self._overlay_vectors(ax, contour, vectors, classifications)
            
        # Set title and hide axes
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Update canvas
        self.canvas.draw()
        
    def plot_flow(self, image: np.ndarray, flow: np.ndarray, 
                 contour: Optional[np.ndarray] = None,
                 title: str = "Optical Flow"):
        """Plot optical flow visualization.
        
        Args:
            image: Background image
            flow: Optical flow field [height, width, 2]
            contour: Optional contour points
            title: Plot title
        """
        # Clear figure
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Show background image
        ax.imshow(image, cmap='gray')
        
        # Create color-coded flow visualization
        flow_img = self._create_flow_visualization(flow)
        
        # Overlay flow visualization with transparency
        ax.imshow(flow_img, alpha=0.7)
        
        # Overlay contour if provided and option enabled
        if contour is not None and self.show_contour.isChecked():
            ax.plot(contour[:, 0], contour[:, 1], 'y-', linewidth=2, alpha=0.8)
            
        # Add arrows for better visualization
        if self.show_vectors.isChecked():
            # Subsample flow field for clearer visualization
            height, width = flow.shape[:2]
            step = max(1, min(height, width) // 20)
            
            y, x = np.mgrid[step//2:height:step, step//2:width:step]
            fx = flow[y, x, 0]
            fy = flow[y, x, 1]
            
            # Scale for visibility
            scale = 1.0
            
            # Draw arrows
            ax.quiver(x, y, fx, fy, 
                     color='white', scale_units='xy', scale=scale,
                     width=0.002, headwidth=5, alpha=0.8)
            
        # Set title and hide axes
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Update canvas
        self.canvas.draw()
        
    def _create_flow_visualization(self, flow: np.ndarray) -> np.ndarray:
        """Create color-coded flow visualization.
        
        Args:
            flow: Optical flow field [height, width, 2]
            
        Returns:
            RGB flow visualization
        """
        # Calculate flow magnitude and direction
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create HSV image
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        
        # Angle to hue (scaled to 0-179 for OpenCV)
        hsv[..., 0] = angle * 180 / np.pi / 2
        
        # Magnitude to saturation (normalized)
        if np.max(magnitude) > 0:
            hsv[..., 1] = np.minimum(magnitude * 255 / np.max(magnitude), 255)
        
        # Full value
        hsv[..., 2] = 255
        
        # Convert HSV to RGB
        import cv2
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return flow_vis
        
    def _overlay_vectors(self, ax, contour: np.ndarray, vectors: np.ndarray, 
                        classifications: Optional[List[str]] = None):
        """Overlay motion vectors on plot.
        
        Args:
            ax: Matplotlib axes
            contour: Contour points
            vectors: Velocity vectors
            classifications: Optional movement classifications for coloring
        """
        # Calculate scale factor for visibility
        scale = 5.0
        
        # Draw vectors for every 10th point to avoid overcrowding
        for i in range(0, len(contour), 10):
            x, y = contour[i]
            vx, vy = vectors[i]
            
            # Skip if vector is too small
            if vx**2 + vy**2 < 1e-6:
                continue
                
            # Determine color based on classification if provided
            if classifications is not None:
                color = self.movement_cmap.get(classifications[i], 'yellow')
            else:
                color = 'yellow'
                
            # Draw arrow
            ax.arrow(
                x, y, vx*scale, vy*scale,
                head_width=5, head_length=7,
                fc=color, ec=color, alpha=0.8
            )
            
    def update_display(self):
        """Update display based on current options."""
        # This method should be connected to main window to update visualization
        # when options change
        pass
        
    def plot_correlation(self, curvatures: np.ndarray, velocities: np.ndarray, 
                        classifications: List[str],
                        title: str = "Curvature-Motion Correlation"):
        """Plot correlation between curvature and normal velocity.
        
        Args:
            curvatures: Curvature values
            velocities: Normal velocity values
            classifications: Movement classifications
            title: Plot title
        """
        # Clear figure
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Scatter plot with classification colors
        for cls in set(classifications):
            # Get indices for this classification
            indices = [i for i, c in enumerate(classifications) if c == cls]
            
            if indices:
                # Extract data for this classification
                cls_curvatures = curvatures[indices]
                cls_velocities = velocities[indices]
                
                # Get color from classification mapping
                color = self.movement_cmap.get(cls, 'yellow')
                
                # Create scatter plot
                ax.scatter(
                    cls_curvatures, cls_velocities,
                    color=color, alpha=0.7, label=cls.capitalize()
                )
        
        # Add trend line
        if len(curvatures) > 1:
            z = np.polyfit(curvatures, velocities, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(curvatures), max(curvatures), 100)
            ax.plot(x_range, p(x_range), 'r--', alpha=0.7)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(curvatures, velocities)[0, 1]
            ax.set_title(f"{title}\nr = {correlation:.3f}")
        else:
            ax.set_title(title)
            
        # Add labels and legend
        ax.set_xlabel('Curvature (nm⁻¹)')
        ax.set_ylabel('Normal Velocity (nm/s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Update canvas
        self.canvas.draw()
        
    def plot_kymograph(self, kymograph: np.ndarray, contour: Optional[np.ndarray] = None,
                      point_idx: Optional[int] = None,
                      title: str = "Membrane Kymograph"):
        """Plot kymograph visualization.
        
        Args:
            kymograph: Kymograph data [frames, positions, 4]
            contour: Optional contour for reference
            point_idx: Point index for reference
            title: Plot title
        """
        # Clear figure
        self.fig.clear()
        
        # Create two subplots: kymograph and reference
        if contour is not None and point_idx is not None:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax1 = self.fig.add_subplot(gs[0])
            ax2 = self.fig.add_subplot(gs[1])
        else:
            ax1 = self.fig.add_subplot(111)
            
        # Plot kymograph
        im = ax1.imshow(kymograph, aspect='auto', origin='upper',
                      extent=[0, kymograph.shape[1], kymograph.shape[0], 0])
        
        ax1.set_title(title)
        ax1.set_ylabel('Frame')
        ax1.set_xlabel('Position along membrane')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='k', label='Expanding'),
            Patch(facecolor='blue', edgecolor='k', label='Retracting'),
            Patch(facecolor='green', edgecolor='k', label='Flowing'),
            Patch(facecolor='gray', edgecolor='k', label='Stationary')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot reference contour if provided
        if contour is not None and point_idx is not None:
            ax2.plot(contour[:, 0], contour[:, 1], 'b-', alpha=0.7)
            
            # Mark selected point
            ax2.plot(contour[point_idx, 0], contour[point_idx, 1], 'ro', markersize=8)
            
            ax2.set_title(f'Selected Point (Index: {point_idx})')
            ax2.set_aspect('equal')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
        # Update layout and canvas
        self.fig.tight_layout()
        self.canvas.draw()