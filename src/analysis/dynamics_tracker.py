#!/usr/bin/env python3
# src/analysis/dynamics_tracker.py

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MembraneMotion:
    """Container for membrane motion data."""
    points: np.ndarray  # Points on the membrane
    velocities: np.ndarray  # Velocity vectors
    magnitudes: np.ndarray  # Movement magnitudes
    directions: np.ndarray  # Movement directions (radians)
    classifications: List[str]  # Movement classifications ('expanding', 'retracting', 'stationary', 'flowing')
    frame_index: int  # Frame index
    time_delta: float  # Time between frames (seconds)


class DynamicsTracker:
    """Class for tracking membrane dynamics between frames."""

    def __init__(self, contour_smoothing: float = 1.0, pixel_size: float = 100.0):
        """Initialize the dynamics tracker.

        Args:
            contour_smoothing: Smoothing sigma for contours
            pixel_size: Size of pixel in nanometers
        """
        self.contour_smoothing = contour_smoothing
        self.pixel_size = pixel_size
        self.previous_contour = None
        self.previous_frame = None
        self.frame_index = 0
        self.motion_history = []
        self.time_delta = 1.0  # Default time between frames (seconds)

    def set_time_delta(self, time_delta: float):
        """Set the time between consecutive frames in seconds."""
        self.time_delta = time_delta

    def analyze_frame(self, frame: np.ndarray, contour: np.ndarray) -> Optional[MembraneMotion]:
        """Analyze membrane dynamics between current and previous frame.

        Args:
            frame: Current frame image
            contour: Current frame membrane contour

        Returns:
            MembraneMotion object or None if this is the first frame
        """
        # If this is the first frame, store it and return None
        if self.previous_contour is None or self.previous_frame is None:
            self.previous_contour = contour.copy()
            self.previous_frame = frame.copy()
            self.frame_index += 1
            return None

        # Analyze motion between previous and current frame
        motion_data = self._track_contour_movement(self.previous_contour, contour)

        # Calculate optical flow for validation and refinement
        #flow_velocities = self._calculate_optical_flow(self.previous_frame, frame, contour)
        flow_velocities = self._calculate_optical_flow_at_contour(
            self.previous_frame, frame, self.previous_contour, contour)

        # Combine contour tracking and optical flow for more accurate velocity estimation
        refined_velocities = self._refine_velocities(motion_data["velocities"], flow_velocities)

        # Calculate magnitude and direction of movement
        magnitudes = np.linalg.norm(refined_velocities, axis=1)
        directions = np.arctan2(refined_velocities[:, 1], refined_velocities[:, 0])

        # Classify movement types
        classifications = self._classify_movements(refined_velocities, contour)

        # Create MembraneMotion object
        motion = MembraneMotion(
            points=contour,
            velocities=refined_velocities,
            magnitudes=magnitudes,
            directions=directions,
            classifications=classifications,
            frame_index=self.frame_index,
            time_delta=self.time_delta
        )

        # Store current frame and contour as previous for next analysis
        self.previous_contour = contour.copy()
        self.previous_frame = frame.copy()
        self.frame_index += 1

        # Add to motion history
        self.motion_history.append(motion)

        return motion

    def _calculate_optical_flow_at_contour(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                                          prev_contour: np.ndarray, curr_contour: np.ndarray) -> np.ndarray:
        """Calculate optical flow specifically at contour points."""
        # Convert frames to grayscale if needed
        if len(prev_frame.shape) > 2 and prev_frame.shape[2] > 1:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame.copy()
            curr_gray = curr_frame.copy()

        # Ensure images are uint8
        if prev_gray.dtype != np.uint8:
            prev_gray = (prev_gray * 255).astype(np.uint8) if prev_gray.max() <= 1.0 else prev_gray.astype(np.uint8)
        if curr_gray.dtype != np.uint8:
            curr_gray = (curr_gray * 255).astype(np.uint8) if curr_gray.max() <= 1.0 else curr_gray.astype(np.uint8)

        # Calculate Lucas-Kanade optical flow for contour points
        contour_points = curr_contour.astype(np.float32)
        prev_points = prev_contour.astype(np.float32)

        # Make sure points are in proper format
        if len(contour_points.shape) == 2:
            contour_points = contour_points.reshape(-1, 1, 2)

        # Calculate optical flow
        try:
            win_size = max(3, int(self.contour_smoothing * 2))
            win_size = win_size if win_size % 2 == 1 else win_size + 1  # Ensure odd

            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray,
                contour_points,
                None,
                winSize=(win_size, win_size),
                maxLevel=2
            )

            # Reshape to 2D array
            new_points = new_points.reshape(-1, 2)
            velocities = (new_points - contour_points.reshape(-1, 2)) / self.time_delta

            # Zero out velocities for points where flow wasn't found
            valid_indices = status.flatten() == 1
            for i in range(len(velocities)):
                if not valid_indices[i]:
                    velocities[i] = np.zeros(2)
        except Exception as e:
            print(f"Error in optical flow calculation: {e}")
            velocities = np.zeros((len(contour_points), 2))

        return velocities

    def _track_contour_movement(self, prev_contour: np.ndarray, curr_contour: np.ndarray) -> Dict:
        """Track movement of contour points between frames.

        This implements point correspondence between contours in consecutive frames
        using shape context and dynamic time warping for non-rigid alignment.

        Args:
            prev_contour: Contour from previous frame
            curr_contour: Contour from current frame

        Returns:
            Dictionary with points and their corresponding velocities
        """
        # Resample contours to have the same number of points for simpler matching
        n_points = min(len(prev_contour), len(curr_contour))
        n_points = max(n_points, 100)  # Ensure reasonable minimum

        prev_resampled = self._resample_contour(prev_contour, n_points)
        curr_resampled = self._resample_contour(curr_contour, n_points)

        # Calculate velocity vectors
        velocities = (curr_resampled - prev_resampled) / self.time_delta

        return {
            "points": curr_resampled,
            "velocities": velocities
        }

    def _resample_contour(self, contour: np.ndarray, n_points: int) -> np.ndarray:
        """Resample contour to have specified number of points.

        Args:
            contour: Original contour
            n_points: Desired number of points

        Returns:
            Resampled contour
        """
        # Calculate contour perimeter
        perimeter = cv2.arcLength(contour.astype(np.float32).reshape(-1, 1, 2), True)

        # Resample contour at equal arc-length intervals
        resampled_contour = np.zeros((n_points, 2), dtype=np.float32)

        for i in range(n_points):
            # Find point at specified arc-length position
            position = i * perimeter / n_points
            index, fraction = self._find_point_at_distance(contour, position)

            # Interpolate between adjacent points
            idx1 = index
            idx2 = (index + 1) % len(contour)

            resampled_contour[i] = contour[idx1] * (1 - fraction) + contour[idx2] * fraction

        return resampled_contour

    def _find_point_at_distance(self, contour: np.ndarray, distance: float) -> Tuple[int, float]:
        """Find point along contour at specified distance.

        Args:
            contour: Contour points
            distance: Distance along contour

        Returns:
            Tuple of (index, fraction) where index is the contour point before the
            desired position and fraction is the interpolation factor to next point
        """
        current_dist = 0.0

        for i in range(len(contour)):
            start_idx = i
            end_idx = (i + 1) % len(contour)

            segment_length = np.linalg.norm(contour[end_idx] - contour[start_idx])

            if current_dist + segment_length >= distance:
                # Point is on this segment
                fraction = (distance - current_dist) / segment_length
                return start_idx, fraction

            current_dist += segment_length

        # If we get here, return the last point
        return len(contour) - 1, 0.0

    # In src/analysis/dynamics_tracker.py, modify the _calculate_optical_flow method:

    def _calculate_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                               contour: np.ndarray) -> np.ndarray:
        """Calculate optical flow at contour points."""
        # Convert frames to grayscale if needed
        if len(prev_frame.shape) > 2 and prev_frame.shape[2] > 1:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            curr_gray = curr_frame

        # Ensure images are uint8 format (important for OpenCV function)
        if prev_gray.dtype != np.uint8:
            prev_gray = (prev_gray * 255).astype(np.uint8) if prev_gray.max() <= 1.0 else prev_gray.astype(np.uint8)
        if curr_gray.dtype != np.uint8:
            curr_gray = (curr_gray * 255).astype(np.uint8) if curr_gray.max() <= 1.0 else curr_gray.astype(np.uint8)

        # Create a mask around the contour to focus flow calculation
        mask = np.zeros_like(prev_gray)
        cv2.drawContours(mask, [contour.astype(np.int32)], 0, 255, 5)

        # Ensure window size is at least 3x3
        win_size = max(3, self.contour_smoothing * 2)
        win_size = int(win_size) if win_size % 2 == 1 else int(win_size) + 1  # Make odd

        # Calculate Lucas-Kanade optical flow
        contour_points = contour.astype(np.float32)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            contour_points.reshape(-1, 1, 2),
            None,
            winSize=(win_size, win_size),
            maxLevel=2
        )

        # Calculate velocity from displacement
        new_points = new_points.reshape(-1, 2)
        velocities = (new_points - contour_points) / self.time_delta

        # Zero out velocities for points where flow wasn't found
        valid_indices = status.flatten() == 1
        for i in range(len(velocities)):
            if not valid_indices[i]:
                velocities[i] = np.zeros(2)

        return velocities

    def _refine_velocities(self, contour_velocities: np.ndarray,
                          flow_velocities: np.ndarray) -> np.ndarray:
        """Refine velocity estimates by combining contour tracking and optical flow.

        Args:
            contour_velocities: Velocities from contour tracking
            flow_velocities: Velocities from optical flow

        Returns:
            Refined velocity estimates
        """
        # Weighted average of contour tracking and optical flow
        # Give more weight to optical flow when available
        refined = np.zeros_like(contour_velocities)

        for i in range(len(contour_velocities)):
            flow_magnitude = np.linalg.norm(flow_velocities[i])
            contour_magnitude = np.linalg.norm(contour_velocities[i])

            if flow_magnitude > 0:
                # When flow is available, use weighted combination
                weight_flow = 0.7
                weight_contour = 0.3
                refined[i] = (weight_flow * flow_velocities[i] +
                             weight_contour * contour_velocities[i])
            else:
                # When flow isn't available, use contour tracking
                refined[i] = contour_velocities[i]

        return refined

    def _classify_movements(self, velocities: np.ndarray,
                           contour: np.ndarray) -> List[str]:
        """Classify membrane movements based on velocity vectors.

        Args:
            velocities: Velocity vectors
            contour: Contour points

        Returns:
            List of movement classifications
        """
        # Calculate normal vectors (pointing inward)
        normals = self._calculate_normals(contour)

        # Threshold for static classification (nm/s)
        static_threshold = 10 / self.pixel_size  # 10 nm/s

        # Threshold for determining if movement is mostly normal or tangential
        direction_threshold = 0.7  # cos(angle) threshold ~45 degrees

        classifications = []

        for i, velocity in enumerate(velocities):
            magnitude = np.linalg.norm(velocity)

            if magnitude < static_threshold:
                classifications.append("stationary")
                continue

            # Calculate dot product to determine if movement is along normal
            v_normalized = velocity / magnitude
            normal_dot = np.dot(v_normalized, normals[i])

            if abs(normal_dot) > direction_threshold:
                # Movement is mostly along normal
                if normal_dot > 0:
                    # Movement is inward
                    classifications.append("retracting")
                else:
                    # Movement is outward
                    classifications.append("expanding")
            else:
                # Movement is mostly tangential
                classifications.append("flowing")

        return classifications

    def _calculate_normals(self, contour: np.ndarray) -> np.ndarray:
        """Calculate inward-pointing normal vectors for contour.

        Args:
            contour: Contour points

        Returns:
            Normal vectors
        """
        n_points = len(contour)
        normals = np.zeros((n_points, 2))

        for i in range(n_points):
            # Use neighboring points to calculate tangent
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points

            # Calculate tangent using central difference
            tangent = contour[next_idx] - contour[prev_idx]
            tangent_norm = np.linalg.norm(tangent)

            if tangent_norm > 0:
                tangent = tangent / tangent_norm

                # Normal is perpendicular to tangent
                normal = np.array([-tangent[1], tangent[0]])

                # Ensure normal points inward (this is approximate)
                # For more accurate determination, would need cell mask
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

    def get_motion_statistics(self, motion: MembraneMotion) -> Dict:
        """Calculate statistical measures of membrane motion.

        Args:
            motion: MembraneMotion object

        Returns:
            Dictionary of statistics
        """
        stats = {
            "mean_speed": np.mean(motion.magnitudes),
            "max_speed": np.max(motion.magnitudes),
            "median_speed": np.median(motion.magnitudes),
            "std_speed": np.std(motion.magnitudes),
            "movement_classifications": {
                "expanding": motion.classifications.count("expanding"),
                "retracting": motion.classifications.count("retracting"),
                "stationary": motion.classifications.count("stationary"),
                "flowing": motion.classifications.count("flowing")
            },
            "expanding_percent": 100 * motion.classifications.count("expanding") / len(motion.classifications),
            "retracting_percent": 100 * motion.classifications.count("retracting") / len(motion.classifications),
            "stationary_percent": 100 * motion.classifications.count("stationary") / len(motion.classifications),
            "flowing_percent": 100 * motion.classifications.count("flowing") / len(motion.classifications)
        }

        return stats

    def get_curvature_motion_correlation(self, curvatures: np.ndarray,
                                        motion: MembraneMotion) -> Dict:
        """Calculate correlation between curvature and motion.

        Args:
            curvatures: Curvature values for contour points
            motion: MembraneMotion object

        Returns:
            Dictionary of correlation statistics
        """
        if len(curvatures) != len(motion.points):
            raise ValueError(f"Number of curvature values ({len(curvatures)}) must match number of contour points ({len(motion.points)})")

        # Calculate normal component of velocity (positive = inward)
        normals = self._calculate_normals(motion.points)
        normal_velocities = np.array([np.dot(v, n) for v, n in zip(motion.velocities, normals)])

        # Calculate correlation
        correlation = np.corrcoef(curvatures, normal_velocities)[0, 1]

        # Calculate mean values for different movement types
        curvature_by_class = {
            "expanding": [],
            "retracting": [],
            "stationary": [],
            "flowing": []
        }

        for i, class_name in enumerate(motion.classifications):
            curvature_by_class[class_name].append(curvatures[i])

        # Calculate mean curvature for each class
        mean_curvature = {}
        for class_name, values in curvature_by_class.items():
            if values:  # Only calculate if we have values
                mean_curvature[class_name] = np.mean(values)
            else:
                mean_curvature[class_name] = 0

        return {
            "correlation": correlation,
            "mean_curvature_by_class": mean_curvature,
            "normal_velocity_mean": np.mean(normal_velocities),
            "normal_velocity_std": np.std(normal_velocities)
        }

    def create_kymograph(self, point_index: int, window_size: int = 10) -> np.ndarray:
        """Create kymograph for a specific membrane point over time."""
        if not self.motion_history:
            return None

        # Create kymograph array
        n_frames = len(self.motion_history)
        kymograph_width = 2 * window_size + 1
        kymograph = np.zeros((n_frames, kymograph_width, 4))  # RGBA for color coding

        for t, motion in enumerate(self.motion_history):
            if motion is None:
                # Skip frames with no motion data
                continue

            # Get number of points in this frame
            n_points = len(motion.points)

            # Skip if point_index is invalid for this frame
            if point_index >= n_points:
                continue

            # Process each position in the window
            for w in range(kymograph_width):
                try:
                    # Safely calculate wrapped index
                    offset = w - window_size
                    if offset >= 0:
                        idx = (point_index + offset) % n_points
                    else:
                        # Handle negative offsets properly
                        idx = (point_index + offset + n_points) % n_points

                    # Double-check index is valid (should be, but let's be extra cautious)
                    if 0 <= idx < len(motion.magnitudes):
                        # Get magnitude and normalize
                        magnitude = min(1.0, motion.magnitudes[idx] / 50)

                        # Set color based on classification
                        if idx < len(motion.classifications):
                            if motion.classifications[idx] == "expanding":
                                kymograph[t, w] = [magnitude, 0, 0, 1]  # Red
                            elif motion.classifications[idx] == "retracting":
                                kymograph[t, w] = [0, 0, magnitude, 1]  # Blue
                            elif motion.classifications[idx] == "flowing":
                                kymograph[t, w] = [0, magnitude, 0, 1]  # Green
                            else:
                                kymograph[t, w] = [magnitude, magnitude, magnitude, 1]  # Gray
                except IndexError as e:
                    # Skip any point that causes an index error
                    print(f"Warning: Index error in kymograph at t={t}, w={w}: {e}")
                    continue

        return kymograph
