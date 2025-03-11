#!/usr/bin/env python3
# src/analysis/optical_flow.py

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class OpticalFlowAnalyzer:
    """Class for optical flow analysis of membrane movement."""
    
    def __init__(self, window_size: int = 15, max_level: int = 2, 
                 pixel_size: float = 100.0, time_delta: float = 1.0):
        """Initialize optical flow analyzer.
        
        Args:
            window_size: Size of search window for optical flow
            max_level: Number of pyramid levels for optical flow
            pixel_size: Size of pixel in nanometers
            time_delta: Time between frames in seconds
        """
        self.window_size = (window_size, window_size)
        self.max_level = max_level
        self.pixel_size = pixel_size
        self.time_delta = time_delta
        
        # Initialize parameters for Farneback dense flow
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # Create colormap for flow visualization
        self.flow_cmap = self._create_flow_colormap()
        
    def _create_flow_colormap(self) -> LinearSegmentedColormap:
        """Create colormap for flow visualization.
        
        Returns:
            Matplotlib colormap
        """
        # Create a circular color mapping based on flow direction
        # with saturation for magnitude
        nsamp = 8
        theta = np.linspace(0, 2*np.pi, nsamp)
        colors = []
        
        # Use HSV color model: hue for direction, saturation for magnitude
        for angle in theta:
            hue = angle / (2*np.pi)  # Map angle to hue (0-1)
            rgb = plt.cm.hsv(hue)
            colors.append(rgb)
            
        # Close the circle
        colors.append(colors[0])
        return LinearSegmentedColormap.from_list('flow_colormap', colors)
        
    def calculate_sparse_flow(self, prev_frame: np.ndarray, current_frame: np.ndarray, 
                            points: np.ndarray) -> Dict:
        """Calculate optical flow for specific points using Lucas-Kanade method.
        
        Args:
            prev_frame: Previous frame
            current_frame: Current frame
            points: Points to track [N, 2]
            
        Returns:
            Dictionary with tracked points, velocities, and status
        """
        # Convert frames to grayscale if needed
        if len(prev_frame.shape) > 2 and prev_frame.shape[2] > 1:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            curr_gray = current_frame
        
        # Ensure points are in the correct format
        points_to_track = points.astype(np.float32).reshape(-1, 1, 2)
        
        # Calculate Lucas-Kanade optical flow
        new_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, 
            curr_gray, 
            points_to_track, 
            None, 
            winSize=self.window_size,
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Reshape to [N, 2]
        new_points = new_points.reshape(-1, 2)
        status = status.flatten().astype(bool)
        
        # Calculate velocities in nm/s
        velocities = (new_points - points) * self.pixel_size / self.time_delta
        
        return {
            'points': points,
            'new_points': new_points,
            'velocities': velocities,
            'status': status,
            'errors': err.flatten()
        }
    
    def calculate_dense_flow(self, prev_frame: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
        """Calculate dense optical flow using Farneback method.
        
        Args:
            prev_frame: Previous frame
            current_frame: Current frame
            
        Returns:
            Flow field [height, width, 2]
        """
        # Convert frames to grayscale if needed
        if len(prev_frame.shape) > 2 and prev_frame.shape[2] > 1:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            curr_gray = current_frame
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, 
            curr_gray, 
            None, 
            **self.farneback_params
        )
        
        # Convert to nm/s
        flow = flow * self.pixel_size / self.time_delta
        
        return flow
    
    def create_flow_visualization(self, frame: np.ndarray, flow: np.ndarray, 
                                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Create visualization of flow field.
        
        Args:
            frame: Original frame
            flow: Flow field [height, width, 2]
            mask: Optional mask to limit visualization
            
        Returns:
            Visualization image
        """
        # Create a copy of the frame
        vis = frame.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # Calculate flow magnitude and direction
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create HSV image for flow visualization
        hsv = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        
        # Angle to hue, scaled to 0-179 for OpenCV
        hsv[..., 0] = angle * 180 / np.pi / 2
        
        # Magnitude to saturation, normalized
        if np.max(magnitude) > 0:
            hsv[..., 1] = np.minimum(magnitude * 255 / np.max(magnitude), 255)
        
        # Full value
        hsv[..., 2] = 255
        
        # Convert HSV to BGR
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Apply mask if provided
        if mask is not None:
            flow_vis = cv2.bitwise_and(flow_vis, flow_vis, mask=mask)
        
        # Blend with original image
        alpha = 0.5
        vis = cv2.addWeighted(vis, 1-alpha, flow_vis, alpha, 0)
        
        return vis
    
    def extract_membrane_flow(self, flow: np.ndarray, contour: np.ndarray, 
                            width: int = 5) -> Dict:
        """Extract flow vectors along membrane contour.
        
        Args:
            flow: Dense flow field
            contour: Membrane contour points
            width: Width of band around contour to sample
            
        Returns:
            Dictionary with membrane flow data
        """
        # Create mask along contour
        h, w = flow.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour.astype(np.int32)], 0, 255, width)
        
        # Sample flow vectors along contour
        flow_vectors = []
        for point in contour:
            x, y = int(point[0]), int(point[1])
            
            # Check bounds
            if 0 <= x < w and 0 <= y < h:
                flow_vector = flow[y, x]
                flow_vectors.append(flow_vector)
            else:
                flow_vectors.append(np.zeros(2))
        
        # Calculate tangent and normal components
        tangents = []
        normals = []
        
        for i in range(len(contour)):
            prev_idx = (i - 1) % len(contour)
            next_idx = (i + 1) % len(contour)
            
            # Calculate tangent using central difference
            tangent = contour[next_idx] - contour[prev_idx]
            t_norm = np.linalg.norm(tangent)
            
            if t_norm > 0:
                tangent = tangent / t_norm
                normal = np.array([-tangent[1], tangent[0]])  # 90° CCW
                
                # Ensure normal points inward (approximate)
                center = np.mean(contour, axis=0)
                to_center = center - contour[i]
                if np.dot(normal, to_center) < 0:
                    normal = -normal
                    
                # Get flow components
                flow_vec = flow_vectors[i]
                tangential_component = np.dot(flow_vec, tangent)
                normal_component = np.dot(flow_vec, normal)
                
                tangents.append(tangential_component)
                normals.append(normal_component)
            else:
                tangents.append(0)
                normals.append(0)
        
        return {
            'contour': contour,
            'flow_vectors': np.array(flow_vectors),
            'tangential_component': np.array(tangents),
            'normal_component': np.array(normals),
            'magnitudes': np.linalg.norm(flow_vectors, axis=1)
        }
    
    def visualize_membrane_flow(self, frame: np.ndarray, contour: np.ndarray, 
                              flow_data: Dict, scale: float = 5.0) -> np.ndarray:
        """Visualize flow vectors along membrane.
        
        Args:
            frame: Original frame
            contour: Membrane contour
            flow_data: Flow data from extract_membrane_flow
            scale: Scaling factor for vector display
            
        Returns:
            Visualization image
        """
        # Create a copy of the frame
        vis = frame.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # Draw contour
        cv2.drawContours(vis, [contour.astype(np.int32)], 0, (255, 255, 0), 2)
        
        # Draw flow vectors
        flow_vectors = flow_data['flow_vectors']
        
        for i, point in enumerate(contour):
            x, y = int(point[0]), int(point[1])
            
            # Skip points near the edge
            if x < 10 or y < 10 or x >= vis.shape[1]-10 or y >= vis.shape[0]-10:
                continue
                
            # Get flow vector
            fx, fy = flow_vectors[i]
            
            # Set color based on direction (outward/inward)
            normal_component = flow_data['normal_component'][i]
            
            if normal_component > 0:  # Inward
                color = (0, 0, 255)  # Red
            else:  # Outward
                color = (255, 0, 0)  # Blue
                
            # Scale for visibility
            end_x = int(x + fx * scale)
            end_y = int(y + fy * scale)
            
            # Draw vector
            cv2.arrowedLine(vis, (x, y), (end_x, end_y), color, 2, tipLength=0.3)
        
        return vis
        
    def analyze_temporal_coherence(self, flow_history: List[np.ndarray], 
                                  contour_history: List[np.ndarray]) -> Dict:
        """Analyze temporal coherence of membrane flow.
        
        Args:
            flow_history: List of flow fields over time
            contour_history: List of contours over time
            
        Returns:
            Dictionary with temporal coherence metrics
        """
        if len(flow_history) < 2 or len(contour_history) < 2:
            return {'coherence': 0, 'persistence': 0}
        
        # Extract contour flows for each frame
        contour_flows = []
        for i in range(len(flow_history)):
            flow = flow_history[i]
            contour = contour_history[i]
            
            # Extract flow along contour
            flow_data = self.extract_membrane_flow(flow, contour)
            contour_flows.append(flow_data)
        
        # Calculate temporal coherence metrics
        coherence_values = []
        
        for i in range(len(contour_flows) - 1):
            current = contour_flows[i]
            next_frame = contour_flows[i + 1]
            
            # Resample to have same number of points
            n_points = min(len(current['contour']), len(next_frame['contour']))
            
            # Calculate how consistent normal components are between frames
            current_normals = current['normal_component'][:n_points]
            next_normals = next_frame['normal_component'][:n_points]
            
            # Correlation between consecutive frames' normal components
            correlation = np.corrcoef(current_normals, next_normals)[0, 1]
            if np.isnan(correlation):
                correlation = 0
                
            coherence_values.append(correlation)
        
        # Calculate persistence (how many frames show consistent direction)
        binary_directions = []
        for flow_data in contour_flows:
            # Classify points as expanding/retracting based on normal component
            directions = np.sign(flow_data['normal_component'])
            binary_directions.append(directions)
        
        # Count how many frames each point maintains same direction
        persistence = np.zeros(len(binary_directions[0]))
        
        for i in range(len(persistence)):
            current_run = 1
            direction = binary_directions[0][i]
            
            for j in range(1, len(binary_directions)):
                if i < len(binary_directions[j]) and binary_directions[j][i] == direction:
                    current_run += 1
                else:
                    break
                    
            persistence[i] = current_run
            
        return {
            'coherence': np.mean(coherence_values),
            'persistence': np.mean(persistence),
            'max_persistence': np.max(persistence),
            'coherence_values': coherence_values
        }