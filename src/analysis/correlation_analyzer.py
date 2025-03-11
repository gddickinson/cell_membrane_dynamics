#!/usr/bin/env python3
# src/analysis/correlation_analyzer.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
from dataclasses import dataclass

@dataclass
class CorrelationData:
    """Container for curvature-motion correlation data."""
    curvatures: np.ndarray  # Curvature values
    velocities: np.ndarray  # Velocity vectors
    normal_velocities: np.ndarray  # Normal component of velocity
    tangential_velocities: np.ndarray  # Tangential component of velocity
    classifications: List[str]  # Movement classifications
    correlation: float  # Correlation coefficient
    p_value: float  # Statistical significance
    frame_index: int  # Frame index
    points: Optional[np.ndarray] = None  # Points in the contour (optional)


class CorrelationAnalyzer:
    """Class for analyzing correlation between curvature and membrane dynamics."""
    
    def __init__(self, pixel_size: float = 100.0):
        """Initialize correlation analyzer.
        
        Args:
            pixel_size: Size of pixel in nanometers
        """
        self.pixel_size = pixel_size
        self.correlation_history = []
        
    def analyze_correlation(self, curvatures: np.ndarray, dynamics_data: Dict, 
                           frame_index: int) -> CorrelationData:
        """Analyze correlation between curvature and membrane dynamics.
        
        Args:
            curvatures: Curvature values for contour points
            dynamics_data: Membrane dynamics data
            frame_index: Frame index
            
        Returns:
            CorrelationData object
        """
        # Ensure data consistency
        n_curvatures = len(curvatures)
        velocities = dynamics_data['velocities']
        
        if n_curvatures != len(velocities):
            raise ValueError(f"Number of curvature values ({n_curvatures}) does not match " 
                           f"number of velocity vectors ({len(velocities)})")
        
        # Extract normal and tangential components
        normal_components = dynamics_data.get('normal_components', 
                                          np.zeros(n_curvatures))
        tangential_components = dynamics_data.get('tangential_components', 
                                               np.zeros(n_curvatures))
        
        # Movement classifications
        classifications = dynamics_data.get('classifications', 
                                         ['unknown'] * n_curvatures)
        
        # Calculate correlation between curvature and normal velocity
        correlation, p_value = stats.pearsonr(curvatures, normal_components)
        
        # Create correlation data object
        corr_data = CorrelationData(
            curvatures=curvatures,
            velocities=velocities,
            normal_velocities=normal_components,
            tangential_velocities=tangential_components,
            classifications=classifications,
            correlation=correlation,
            p_value=p_value,
            frame_index=frame_index,
            points=dynamics_data.get('points', None)
        )
        
        # Add to history
        self.correlation_history.append(corr_data)
        
        return corr_data
    
    def analyze_temporal_correlation(self) -> Dict:
        """Analyze how correlation evolves over time.
        
        Returns:
            Dictionary with temporal correlation analysis
        """
        if len(self.correlation_history) < 2:
            return {'temporal_correlation': 0}
        
        # Extract correlations over time
        correlations = [data.correlation for data in self.correlation_history]
        frames = [data.frame_index for data in self.correlation_history]
        
        # Calculate temporal stability
        correlation_stability = np.std(correlations)
        
        # Calculate if correlation is strengthening or weakening over time
        if len(correlations) >= 3:
            # Calculate linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                frames, correlations)
            
            trend = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err
            }
        else:
            trend = {
                'slope': 0,
                'r_value': 0,
                'p_value': 1
            }
        
        return {
            'correlations': correlations,
            'frames': frames,
            'mean_correlation': np.mean(correlations),
            'correlation_stability': correlation_stability,
            'trend': trend
        }
    
    def create_correlation_visualization(self, correlation_data: CorrelationData) -> plt.Figure:
        """Create visualization of curvature-motion correlation.
        
        Args:
            correlation_data: CorrelationData object
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 10))
        
        # Create scatter plot of curvature vs. normal velocity
        ax1 = fig.add_subplot(221)
        
        # Color points by classification
        colors = {
            'expanding': 'red',
            'retracting': 'blue',
            'stationary': 'gray',
            'flowing': 'green',
            'unknown': 'black'
        }
        
        # Create scatter plot with colored points
        for cls in colors.keys():
            mask = [c == cls for c in correlation_data.classifications]
            if any(mask):
                ax1.scatter(
                    correlation_data.curvatures[mask], 
                    correlation_data.normal_velocities[mask],
                    color=colors[cls],
                    alpha=0.7,
                    label=cls
                )
        
        # Add trend line
        if len(correlation_data.curvatures) > 1:
            z = np.polyfit(correlation_data.curvatures, 
                         correlation_data.normal_velocities, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(correlation_data.curvatures), 
                               max(correlation_data.curvatures), 100)
            ax1.plot(x_range, p(x_range), 'r--', alpha=0.7)
        
        ax1.set_xlabel('Curvature (nm⁻¹)')
        ax1.set_ylabel('Normal Velocity (nm/s)')
        ax1.set_title(f'Curvature vs. Normal Velocity\nr = {correlation_data.correlation:.3f}, p = {correlation_data.p_value:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Create curvature profile
        ax2 = fig.add_subplot(222)
        x = range(len(correlation_data.curvatures))
        ax2.plot(x, correlation_data.curvatures, 'b-', linewidth=2)
        ax2.set_xlabel('Contour Position')
        ax2.set_ylabel('Curvature (nm⁻¹)')
        ax2.set_title('Curvature Profile')
        ax2.grid(True, alpha=0.3)
        
        # Create velocity profile
        ax3 = fig.add_subplot(223)
        ax3.plot(x, correlation_data.normal_velocities, 'r-', 
               label='Normal', linewidth=2)
        ax3.plot(x, correlation_data.tangential_velocities, 'g-', 
               label='Tangential', linewidth=2)
        ax3.set_xlabel('Contour Position')
        ax3.set_ylabel('Velocity (nm/s)')
        ax3.set_title('Velocity Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Create motion classification pie chart
        ax4 = fig.add_subplot(224)
        class_counts = {}
        for cls in colors.keys():
            count = correlation_data.classifications.count(cls)
            if count > 0:
                class_counts[cls] = count
        
        if class_counts:
            ax4.pie(class_counts.values(), labels=class_counts.keys(), 
                  colors=[colors[cls] for cls in class_counts.keys()],
                  autopct='%1.1f%%')
            ax4.set_title('Motion Classification')
        else:
            ax4.text(0.5, 0.5, 'No classification data', 
                   horizontalalignment='center', verticalalignment='center')
        
        fig.tight_layout()
        return fig
    
    def create_curvature_motion_map(self, correlation_data: CorrelationData, 
                                  membrane_image: np.ndarray) -> np.ndarray:
        """Create visualization map showing curvature and motion on membrane.
        
        Args:
            correlation_data: CorrelationData object
            membrane_image: Image of membrane (grayscale or RGB)
            
        Returns:
            Visualization image
        """
        # Ensure we have contour points
        if correlation_data.points is None or len(correlation_data.points) == 0:
            raise ValueError("No contour points in correlation data")
        
        # Create RGB image if input is grayscale
        if len(membrane_image.shape) == 2:
            vis_image = cv2.cvtColor(membrane_image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = membrane_image.copy()
        
        # Create contour visualization with curvature coloring
        curvatures = correlation_data.curvatures
        points = correlation_data.points.astype(np.int32)
        
        # Normalize curvature for coloring
        if np.max(np.abs(curvatures)) > 0:
            normalized_curvatures = curvatures / np.max(np.abs(curvatures))
        else:
            normalized_curvatures = np.zeros_like(curvatures)
        
        # Draw curvature as contour coloring
        for i in range(len(points) - 1):
            # RGB: positive curvature = red, negative = blue
            if normalized_curvatures[i] > 0:
                color = (0, 0, int(255 * normalized_curvatures[i]))  # Red
            else:
                color = (int(-255 * normalized_curvatures[i]), 0, 0)  # Blue
                
            # Draw line segment with curvature coloring
            cv2.line(vis_image, tuple(points[i]), tuple(points[i+1]), color, 2)
        
        # Connect last and first point
        if len(points) > 1:
            i = len(points) - 1
            if normalized_curvatures[i] > 0:
                color = (0, 0, int(255 * normalized_curvatures[i]))
            else:
                color = (int(-255 * normalized_curvatures[i]), 0, 0)
            cv2.line(vis_image, tuple(points[i]), tuple(points[0]), color, 2)
        
        # Draw motion vectors
        for i, point in enumerate(points):
            x, y = point
            
            # Skip points near the edge
            if x < 10 or y < 10 or x >= vis_image.shape[1]-10 or y >= vis_image.shape[0]-10:
                continue
            
            # Create vector from velocity
            velocity = correlation_data.velocities[i]
            dx, dy = velocity
            
            # Get classification for color
            classification = correlation_data.classifications[i]
            
            # Set color based on classification
            if classification == 'expanding':
                color = (0, 0, 255)  # Red
            elif classification == 'retracting':
                color = (255, 0, 0)  # Blue
            elif classification == 'flowing':
                color = (0, 255, 0)  # Green
            else:  # stationary or unknown
                color = (128, 128, 128)  # Gray
            
            # Scale for visibility (normalize by pixel size)
            scale = 50.0 / self.pixel_size
            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)
            
            # Draw arrow
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:  # Only draw non-zero vectors
                cv2.arrowedLine(vis_image, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
        
        return vis_image
    
    def analyze_curvature_by_classification(self, correlation_data: CorrelationData) -> Dict:
        """Analyze curvature characteristics by movement classification.
        
        Args:
            correlation_data: CorrelationData object
            
        Returns:
            Dictionary with analysis results
        """
        # Group curvatures by classification
        curvature_by_class = {}
        
        for cls in set(correlation_data.classifications):
            # Get indices for this classification
            indices = [i for i, c in enumerate(correlation_data.classifications) if c == cls]
            
            if indices:
                curvatures = correlation_data.curvatures[indices]
                normal_velocities = correlation_data.normal_velocities[indices]
                
                curvature_by_class[cls] = {
                    'count': len(indices),
                    'mean_curvature': np.mean(curvatures),
                    'std_curvature': np.std(curvatures),
                    'mean_normal_velocity': np.mean(normal_velocities),
                    'std_normal_velocity': np.std(normal_velocities)
                }
                
                # Calculate correlation within this class
                if len(indices) > 2:
                    corr, p_val = stats.pearsonr(curvatures, normal_velocities)
                    curvature_by_class[cls]['correlation'] = corr
                    curvature_by_class[cls]['p_value'] = p_val
                else:
                    curvature_by_class[cls]['correlation'] = 0
                    curvature_by_class[cls]['p_value'] = 1
        
        return curvature_by_class
    
    def generate_correlation_report(self, correlation_data: CorrelationData) -> pd.DataFrame:
        """Generate a statistical report of curvature-motion correlation.
        
        Args:
            correlation_data: CorrelationData object
            
        Returns:
            DataFrame with statistical report
        """
        # Overall correlation statistics
        overall_stats = {
            'metric': ['Overall Correlation'],
            'correlation': [correlation_data.correlation],
            'p_value': [correlation_data.p_value],
            'mean_curvature': [np.mean(correlation_data.curvatures)],
            'std_curvature': [np.std(correlation_data.curvatures)],
            'mean_normal_velocity': [np.mean(correlation_data.normal_velocities)],
            'std_normal_velocity': [np.std(correlation_data.normal_velocities)]
        }
        
        # Get curvature statistics by classification
        class_stats = self.analyze_curvature_by_classification(correlation_data)
        
        # Combine into dataframe
        rows = []
        rows.append(overall_stats)
        
        for cls, stats in class_stats.items():
            row = {
                'metric': [f'Class: {cls}'],
                'correlation': [stats.get('correlation', 0)],
                'p_value': [stats.get('p_value', 1)],
                'mean_curvature': [stats['mean_curvature']],
                'std_curvature': [stats['std_curvature']],
                'mean_normal_velocity': [stats['mean_normal_velocity']],
                'std_normal_velocity': [stats['std_normal_velocity']],
                'count': [stats['count']],
                'percent': [100 * stats['count'] / len(correlation_data.curvatures)]
            }
            rows.append(row)
        
        # Create DataFrame from rows
        df = pd.concat([pd.DataFrame(row) for row in rows], ignore_index=True)
        return df
    
    def create_binned_analysis(self, correlation_data: CorrelationData, 
                             n_bins: int = 10) -> Dict:
        """Create binned analysis of curvature-velocity relationship.
        
        Args:
            correlation_data: CorrelationData object
            n_bins: Number of curvature bins
            
        Returns:
            Dictionary with binned analysis
        """
        # Create curvature bins
        curvatures = correlation_data.curvatures
        normal_velocities = correlation_data.normal_velocities
        
        min_curv = np.min(curvatures)
        max_curv = np.max(curvatures)
        
        if min_curv == max_curv:
            # If all curvatures are the same, create a single bin
            bin_edges = np.array([min_curv - 0.001, min_curv + 0.001])
            n_bins = 1
        else:
            # Create evenly spaced bins
            bin_edges = np.linspace(min_curv, max_curv, n_bins + 1)
        
        # Initialize arrays for binned statistics
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = np.zeros(n_bins)
        bin_stds = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=int)
        
        # Assign velocities to bins and calculate statistics
        for i in range(n_bins):
            if i == n_bins - 1:
                # Include upper bound in last bin
                mask = (curvatures >= bin_edges[i]) & (curvatures <= bin_edges[i+1])
            else:
                mask = (curvatures >= bin_edges[i]) & (curvatures < bin_edges[i+1])
            
            if np.any(mask):
                bin_velocities = normal_velocities[mask]
                bin_means[i] = np.mean(bin_velocities)
                bin_stds[i] = np.std(bin_velocities)
                bin_counts[i] = np.sum(mask)
        
        return {
            'bin_centers': bin_centers,
            'bin_means': bin_means,
            'bin_stds': bin_stds,
            'bin_counts': bin_counts,
            'bin_edges': bin_edges
        }