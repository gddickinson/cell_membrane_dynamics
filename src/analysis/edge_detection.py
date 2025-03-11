#!/usr/bin/env python3
# src/analysis/edge_detection.py

import numpy as np
import cv2
from skimage import morphology, filters, measure
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional, List, Dict

class EdgeDetector:
    """Class for detecting and processing cell edges from binary masks."""

    def __init__(self, smoothing_sigma: float = 1.0, min_size: int = 100):
        """Initialize edge detector.

        Args:
            smoothing_sigma: Gaussian smoothing sigma for contours
            min_size: Minimum object size to consider
        """
        self.smoothing_sigma = smoothing_sigma
        self.min_size = min_size

    def detect_edge(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Detect cell edge from binary segmentation, excluding border regions.

        Args:
            mask: Binary mask image

        Returns:
            Contour as Nx2 array of (x,y) coordinates or None if detection fails
        """
        try:
            # Ensure mask is binary
            binary_mask = mask > 0

            # Get dimensions
            height, width = binary_mask.shape[:2]

            # Create a new mask with a black border to explicitly remove edge regions
            border_margin = 30  # Increased border margin
            processed_mask = np.zeros_like(binary_mask)

            # Copy only the central portion of the mask (ignore border regions)
            processed_mask[border_margin:height-border_margin, border_margin:width-border_margin] = \
                binary_mask[border_margin:height-border_margin, border_margin:width-border_margin]

            # Clean up small objects in the remaining mask
            processed_mask = morphology.remove_small_objects(
                processed_mask, min_size=self.min_size
            )

            # Convert to uint8 for OpenCV
            mask_uint8 = processed_mask.astype(np.uint8) * 255

            # Find contours using OpenCV
            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )

            if not contours or len(contours) == 0:
                return None

            # Get largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Convert to Nx2 array and remove redundant dimension
            if largest_contour.size == 0:
                return None

            contour_points = largest_contour.squeeze()

            # Handle case where only one point is returned (rare but can happen)
            if len(contour_points.shape) < 2:
                return None

            # Apply optional smoothing
            if self.smoothing_sigma > 0:
                smoothed_contour = self.smooth_contour(contour_points)
                return smoothed_contour
            else:
                return contour_points

        except Exception as e:
            print(f"Error in edge detection: {e}")
            return None

    def smooth_contour(self, contour: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to contour.

        Args:
            contour: Original contour points

        Returns:
            Smoothed contour
        """
        # Ensure contour is at least 3 points
        if len(contour) < 3:
            return contour

        # Apply Gaussian smoothing to coordinates
        smoothed = gaussian_filter1d(
            contour.astype(float),
            self.smoothing_sigma,
            axis=0,
            mode='wrap'  # Wrap around for closed contours
        )

        return smoothed

    def calculate_normals(self, contour: np.ndarray) -> np.ndarray:
        """Calculate normal vectors along contour.

        Args:
            contour: Contour points

        Returns:
            Normal vectors for each point
        """
        n_points = len(contour)
        normals = np.zeros((n_points, 2))

        for i in range(n_points):
            # Use neighboring points
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points

            # Calculate tangent using central difference
            tangent = contour[next_idx] - contour[prev_idx]
            tangent_norm = np.linalg.norm(tangent)

            if tangent_norm > 0:
                tangent = tangent / tangent_norm

                # Normal is perpendicular to tangent (90° counterclockwise)
                normal = np.array([-tangent[1], tangent[0]])

                # Ensure normal points inward (approximate)
                # This assumes a roughly convex cell shape
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

    def get_sampling_points(self, contour: np.ndarray, n_samples: int) -> np.ndarray:
        """Get evenly spaced sampling points along the contour.

        Args:
            contour: Contour points
            n_samples: Number of samples to generate

        Returns:
            Indices of sampling points
        """
        n_points = len(contour)
        if n_samples >= n_points:
            # Return all points if we want more samples than available points
            return np.arange(n_points)
        else:
            # Generate evenly spaced indices
            return np.linspace(0, n_points-1, n_samples, dtype=int)

    def create_edge_image(self, mask_shape: Tuple[int, int], contour: np.ndarray) -> np.ndarray:
        """Create binary edge image from contour.

        Args:
            mask_shape: Shape of the original mask (height, width)
            contour: Contour points

        Returns:
            Binary edge image
        """
        edge_image = np.zeros(mask_shape, dtype=np.uint8)

        # Convert contour to int32 for OpenCV
        contour_int = contour.astype(np.int32)

        # Draw contour on empty image
        cv2.drawContours(edge_image, [contour_int], 0, 255, 1)

        return edge_image

    def analyze_contour(self, contour: np.ndarray) -> Dict:
        """Calculate basic contour metrics.

        Args:
            contour: Contour points

        Returns:
            Dictionary of contour metrics
        """
        # Convert to int32 for OpenCV
        contour_int = contour.astype(np.int32)

        # Calculate area
        area = cv2.contourArea(contour_int)

        # Calculate perimeter
        perimeter = cv2.arcLength(contour_int, True)

        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour_int)

        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 0

        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'bounding_box': (x, y, w, h),
            'aspect_ratio': aspect_ratio,
            'n_points': len(contour)
        }

    def segment_cell(self, image: np.ndarray) -> np.ndarray:
        """Segment cell from grayscale image.

        Args:
            image: Grayscale image

        Returns:
            Binary mask
        """
        # Normalize image
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Apply Gaussian filter to reduce noise
        smoothed = filters.gaussian(normalized, sigma=1.0)

        # Calculate threshold using Otsu's method
        threshold = filters.threshold_otsu(smoothed)
        binary = smoothed > threshold

        # Clean up small objects and holes
        clean_mask = morphology.remove_small_objects(binary, min_size=self.min_size)
        clean_mask = morphology.remove_small_holes(clean_mask, area_threshold=self.min_size)

        # Keep largest object only
        labels = measure.label(clean_mask)
        if labels.max() > 0:
            largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
            clean_mask = labels == largest_label

        return clean_mask
