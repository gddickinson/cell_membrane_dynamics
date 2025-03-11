#!/usr/bin/env python3
# src/gui/time_sequence_panel.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QSpinBox, QPushButton, QCheckBox
)
from typing import List, Optional, Dict, Tuple

class TimeSequencePanel(QWidget):
    """Panel for time sequence analysis and visualization."""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        
        # Movement classification colors
        self.movement_colors = {
            'expanding': 'red',
            'retracting': 'blue',
            'flowing': 'green',
            'stationary': 'gray'
        }
        
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure for time series plots
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        # Control panel for visualization options
        control_layout = QHBoxLayout()
        
        # Metric selector
        control_layout.addWidget(QLabel("Metric:"))
        self.metric_selector = QComboBox()
        self.metric_selector.addItems([
            "Correlation", "Curvature", "Velocity", 
            "Classification", "Intensity"
        ])
        self.metric_selector.currentIndexChanged.connect(self.update_display)
        control_layout.addWidget(self.metric_selector)
        
        # Point selection for single-point time series
        control_layout.addWidget(QLabel("Point:"))
        self.point_selector = QSpinBox()
        self.point_selector.setMinimum(0)
        self.point_selector.setMaximum(0)
        self.point_selector.valueChanged.connect(self.update_display)
        control_layout.addWidget(self.point_selector)
        
        # Update button
        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_display)
        control_layout.addWidget(self.update_button)
        
        # Add control layout to main layout
        layout.addLayout(control_layout)
        
    def plot_correlation_time_series(self, correlations: List[float], frames: List[int],
                                   mean_curvatures: Optional[List[float]] = None,
                                   mean_velocities: Optional[List[float]] = None,
                                   title: str = "Correlation vs Time"):
        """Plot correlation coefficient over time.
        
        Args:
            correlations: Correlation values
            frames: Frame indices
            mean_curvatures: Optional mean curvature values
            mean_velocities: Optional mean velocity values
            title: Plot title
        """
        # Clear figure
        self.fig.clear()
        
        # Create subplots
        if mean_curvatures is not None and mean_velocities is not None:
            # Three subplots: correlation, curvature, velocity
            ax1 = self.fig.add_subplot(311)
            ax2 = self.fig.add_subplot(312, sharex=ax1)
            ax3 = self.fig.add_subplot(313, sharex=ax1)
        else:
            # Single plot
            ax1 = self.fig.add_subplot(111)
            ax2 = None
            ax3 = None
            
        # Plot correlation vs time
        ax1.plot(frames, correlations, 'b-o', linewidth=2)
        ax1.set_ylabel('Correlation Coefficient')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line at zero
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot mean curvature and velocity if provided
        if ax2 is not None and ax3 is not None:
            # Curvature plot
            ax2.plot(frames, mean_curvatures, 'r-', linewidth=2)
            ax2.set_ylabel('Mean Curvature (nm⁻¹)')
            ax2.grid(True, alpha=0.3)
            
            # Velocity plot
            ax3.plot(frames, mean_velocities, 'g-', linewidth=2)
            ax3.set_ylabel('Mean Velocity (nm/s)')
            ax3.set_xlabel('Frame Number')
            ax3.grid(True, alpha=0.3)
        else:
            # Add x-axis label to correlation plot
            ax1.set_xlabel('Frame Number')
            
        # Update layout and canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_classification_time_series(self, classifications: List[Dict[str, float]],
                                      frames: List[int],
                                      title: str = "Movement Classification vs Time"):
        """Plot movement classification percentages over time.
        
        Args:
            classifications: List of dictionaries with class percentages
            frames: Frame indices
            title: Plot title
        """
        # Clear figure
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Extract percentages for each classification
        expanding = [c.get('expanding', 0) for c in classifications]
        retracting = [c.get('retracting', 0) for c in classifications]
        flowing = [c.get('flowing', 0) for c in classifications]
        stationary = [c.get('stationary', 0) for c in classifications]
        
        # Plot percentages
        ax.plot(frames, expanding, 'r-', linewidth=2, label='Expanding')
        ax.plot(frames, retracting, 'b-', linewidth=2, label='Retracting')
        ax.plot(frames, flowing, 'g-', linewidth=2, label='Flowing')
        ax.plot(frames, stationary, 'k-', linewidth=2, label='Stationary')
        
        # Set labels and title
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_point_time_series(self, curvatures: List[np.ndarray], 
                             velocities: Optional[List[np.ndarray]] = None,
                             classifications: Optional[List[List[str]]] = None,
                             intensities: Optional[List[np.ndarray]] = None,
                             frames: Optional[List[int]] = None,
                             point_idx: Point index to track
            title: Plot title
        """
        # Clear figure
        self.fig.clear()
        
        # Create subplots based on available data
        num_plots = 1  # Start with curvature
        if velocities is not None:
            num_plots += 1
        if intensities is not None:
            num_plots += 1
            
        # Create axes
        axes = []
        for i in range(num_plots):
            axes.append(self.fig.add_subplot(num_plots, 1, i+1))
            
        # Create x-axis values
        if frames is not None:
            x_values = frames
        else:
            x_values = list(range(len(curvatures)))
            
        # Current axis index
        ax_idx = 0
            
        # Plot curvature over time
        if len(curvatures) > 0 and point_idx < len(curvatures[0]):
            # Extract curvature values for the point
            curvature_values = [c[point_idx] if point_idx < len(c) else np.nan for c in curvatures]
            
            # Plot curvature
            axes[ax_idx].plot(x_values, curvature_values, 'r-', linewidth=2)
            axes[ax_idx].set_ylabel('Curvature (nm⁻¹)')
            axes[ax_idx].set_title(f"{title} - Point {point_idx}")
            axes[ax_idx].grid(True, alpha=0.3)
            
            # Color background by classification if available
            if classifications is not None:
                self._add_classification_background(
                    axes[ax_idx], x_values, classifications, point_idx
                )
                
            ax_idx += 1
            
        # Plot velocity over time
        if velocities is not None and len(velocities) > 0 and point_idx < len(velocities[0]):
            # Extract velocity values for the point
            velocity_values = [v[point_idx] if point_idx < len(v) else np.nan for v in velocities]
            
            # Plot velocity
            axes[ax_idx].plot(x_values, velocity_values, 'b-', linewidth=2)
            axes[ax_idx].set_ylabel('Normal Velocity (nm/s)')
            axes[ax_idx].grid(True, alpha=0.3)
            
            # Color background by classification if available
            if classifications is not None:
                self._add_classification_background(
                    axes[ax_idx], x_values, classifications, point_idx
                )
                
            ax_idx += 1
            
        # Plot intensity over time
        if intensities is not None and len(intensities) > 0 and point_idx < len(intensities[0]):
            # Extract intensity values for the point
            intensity_values = [i[point_idx] if point_idx < len(i) else np.nan for i in intensities]
            
            # Plot intensity
            axes[ax_idx].plot(x_values, intensity_values, 'g-', linewidth=2)
            axes[ax_idx].set_ylabel('Intensity (a.u.)')
            axes[ax_idx].grid(True, alpha=0.3)
            
            # Color background by classification if available
            if classifications is not None:
                self._add_classification_background(
                    axes[ax_idx], x_values, classifications, point_idx
                )
                
        # Set x-axis label on bottom subplot
        axes[-1].set_xlabel('Frame Number')
        
        # Update layout and canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
    def _add_classification_background(self, ax, x_values, classifications, point_idx):
        """Add background coloring based on movement classification.
        
        Args:
            ax: Matplotlib axis
            x_values: X-axis values (frame numbers)
            classifications: List of classification lists (one per frame)
            point_idx: Point index to track
        """
        if len(x_values) != len(classifications):
            return
            
        # Get classification for each frame
        point_classes = []
        for i, cls_list in enumerate(classifications):
            if point_idx < len(cls_list):
                point_classes.append(cls_list[point_idx])
            else:
                point_classes.append(None)
                
        # Add background color spans
        current_class = None
        start_idx = 0
        
        for i, cls in enumerate(point_classes):
            if cls != current_class:
                # End previous span
                if current_class is not None and i > start_idx:
                    # Get color for this classification
                    color = self.movement_colors.get(current_class, 'gray')
                    
                    # Add background span
                    ax.axvspan(
                        x_values[start_idx], x_values[i-1],
                        alpha=0.2, color=color
                    )
                    
                # Start new span
                current_class = cls
                start_idx = i
                
        # Add final span
        if current_class is not None and start_idx < len(x_values):
            color = self.movement_colors.get(current_class, 'gray')
            ax.axvspan(
                x_values[start_idx], x_values[-1],
                alpha=0.2, color=color
            )
    
    def update_display(self):
        """Update display based on current options."""
        # This method should be connected to main window to update visualization
        # when options change
        pass int = 0,
                             title: str = "Point Time Series"):
        """Plot time series for a specific point on the contour.
        
        Args:
            curvatures: List of curvature arrays (one per frame)
            velocities: Optional list of velocity arrays (one per frame)
            classifications: Optional list of classification lists (one per frame)
            intensities: Optional list of intensity arrays (one per frame)
            frames: Optional frame indices
            point_idx: