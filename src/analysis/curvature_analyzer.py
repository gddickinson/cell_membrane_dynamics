#!/usr/bin/env python3
# src/analysis/curvature_analyzer.py

import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.interpolate import splprep, splev

class CurvatureAnalyzer:
    """Class for analyzing membrane curvature."""

    def __init__(self, pixel_size: float = 100.0, segment_length: int = 9):
        """Initialize the curvature analyzer.
        
        Args:
            pixel_size: Size of pixel in nanometers
            segment_length: Length of segment to use for curvature calculation
        """
        self.pixel_size = pixel_size
        self.segment_length = segment_length
        self.ref_curvatures = {
            'plasma_membrane': 1/(10000),  # nm^-1
            'transport_vesicle': 1/(40),    # nm^-1
            'endocytic_vesicle': 1/(100)    # nm^-1
        }

    def calculate_curvature(self, contour: np.ndarray, method: str = 'finite_difference') -> np.ndarray:
        """Calculate curvature along the contour.
        
        Args:
            contour: Contour points as Nx2 array
            method: Method to use ('finite_difference', 'circle_fit', or 'spline')
            
        Returns:
            Array of curvature values for each point
        """
        if method == 'finite_difference':
            return self._finite_difference_curvature(contour)
        elif method == 'circle_fit':
            return self._circle_fit_curvature(contour)
        elif method == 'spline':
            return self._spline_curvature(contour)
        else:
            raise ValueError(f"Unknown curvature method: {method}")

    def _finite_difference_curvature(self, contour: np.ndarray) -> np.ndarray:
        """Calculate curvature using finite differences.
        
        Args:
            contour: Contour points
            
        Returns:
            Curvature at each point
        """
        n_points = len(contour)
        curvatures = np.zeros(n_points)
        
        for i in range(n_points):
            # Use points before and after for curvature calculation
            # with circular indexing for closed contours
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            
            # Get points
            prev = contour[prev_idx]
            curr = contour[i]
            next_pt = contour[next_idx]
            
            # Calculate curvature using finite differences
            try:
                dx1 = curr[0] - prev[0]
                dy1 = curr[1] - prev[1]
                dx2 = next_pt[0] - curr[0]
                dy2 = next_pt[1] - curr[1]
                
                # First derivatives
                dx = (dx1 + dx2) / 2
                dy = (dy1 + dy2) / 2
                
                # Second derivatives
                d2x = dx2 - dx1
                d2y = dy2 - dy1
                
                # Curvature formula: k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
                num = dx * d2y - dy * d2x
                denom = (dx**2 + dy**2)**(1.5)
                
                if abs(denom) > 1e-10:
                    # Curvature with sign (negative = convex, positive = concave)
                    # Convert to physical units (nm^-1)
                    curvatures[i] = -num / denom / self.pixel_size
                else:
                    curvatures[i] = 0
            except:
                curvatures[i] = 0
        
        return curvatures

    def _circle_fit_curvature(self, contour: np.ndarray) -> np.ndarray:
        """Calculate curvature by fitting circles to segments.
        
        Args:
            contour: Contour points
            
        Returns:
            Curvature at each point
        """
        n_points = len(contour)
        curvatures = np.zeros(n_points)
        half_segment = self.segment_length // 2
        
        for i in range(n_points):
            # Get segment centered at current point
            indices = np.arange(i - half_segment, i + half_segment + 1) % n_points
            segment = contour[indices]
            
            if len(segment) < 3:
                continue
                
            # Fit circle to segment
            try:
                # Translate segment to origin
                center = segment.mean(axis=0)
                centered = segment - center
                
                # Scale to nm
                centered_nm = centered * self.pixel_size
                
                # Apply algebraic circle fitting (Taubin method)
                x = centered_nm[:, 0]
                y = centered_nm[:, 1]
                z = x*x + y*y
                
                # Check for valid values
                if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                    continue
                
                # Create design matrix
                ZXY = np.column_stack((z, x, y))
                A = np.dot(ZXY.T, ZXY)
                
                # Constraint matrix
                B = np.array([[4, 0, 0], [0, 1, 0], [0, 0, 1]])
                
                # Solve generalized eigenvalue problem
                eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(B), A))
                
                # Get eigenvector corresponding to smallest eigenvalue
                v = eigenvectors[:, np.argmin(np.abs(eigenvalues))]
                
                # Check for valid eigenvector
                if np.any(np.isnan(v)) or np.any(np.isinf(v)) or np.abs(v[0]) < 1e-10:
                    continue
                
                # Calculate center (a, b) and radius r of fitted circle
                a = -v[1]/(2*v[0])
                b = -v[2]/(2*v[0])
                
                # Calculate radius (now in nm)
                r = np.sqrt(a*a + b*b - v[0]/v[2])
                
                # Verify the fit is reasonable
                min_radius = (self.segment_length * self.pixel_size) / 4
                if (not np.isfinite(r) or r < min_radius):
                    continue
                
                # Determine sign of curvature (whether the curve is concave or convex)
                # This requires determining if center is "inside" or "outside" the curve
                
                # Calculate normal vector at center point
                segment_dir = segment[-1] - segment[0]
                normal = np.array([-segment_dir[1], segment_dir[0]])
                normal_norm = np.linalg.norm(normal)
                
                if normal_norm < 1e-10:
                    continue
                    
                normal = normal / normal_norm
                
                # Convert center back to pixel space for comparison
                center_point = np.array([a, b]) / self.pixel_size + center
                
                # Vector from mean point to center of the circle
                to_center = center_point - segment.mean(axis=0)
                to_center_norm = np.linalg.norm(to_center)
                
                if to_center_norm > 0:
                    # Curvature is positive when curve bulges in the direction of normal
                    sign = np.sign(np.dot(to_center/to_center_norm, normal))
                    
                    # Curvature = 1/radius (with sign)
                    curvatures[i] = sign * (1/r)
                else:
                    curvatures[i] = 0
                    
            except (np.linalg.LinAlgError, ZeroDivisionError, ValueError):
                curvatures[i] = 0
                
        return curvatures

    def _spline_curvature(self, contour: np.ndarray) -> np.ndarray:
        """Calculate curvature using spline fitting.
        
        Args:
            contour: Contour points
            
        Returns:
            Curvature at each point
        """
        n_points = len(contour)
        
        # Extract x and y coordinates
        x = contour[:, 0]
        y = contour[:, 1]
        
        # Create periodic parameter for the points
        t = np.linspace(0, 1, n_points)
        
        try:
            # Fit a periodic spline to the contour (s=smoothing)
            tck, u = splprep([x, y], u=t, s=n_points/50, per=1)
            
            # Evaluate the spline at the original points
            spline_points = np.array(splev(t, tck, der=0)).T
            
            # Evaluate first and second derivatives
            spline_1st_deriv = np.array(splev(t, tck, der=1)).T  # [dx/dt, dy/dt]
            spline_2nd_deriv = np.array(splev(t, tck, der=2)).T  # [d²x/dt², d²y/dt²]
            
            # Calculate curvature using the formula:
            # κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
            curvatures = np.zeros(n_points)
            
            for i in range(n_points):
                # First derivatives
                dx_dt = spline_1st_deriv[i, 0]
                dy_dt = spline_1st_deriv[i, 1]
                
                # Second derivatives
                d2x_dt2 = spline_2nd_deriv[i, 0]
                d2y_dt2 = spline_2nd_deriv[i, 1]
                
                # Curvature formula
                numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
                denominator = (dx_dt**2 + dy_dt**2)**(1.5)
                
                if abs(denominator) > 1e-10:
                    # Convert to physical units (nm^-1) and apply sign convention
                    # (positive = concave, negative = convex)
                    curvatures[i] = -numerator / denominator / self.pixel_size
                else:
                    curvatures[i] = 0
            
            return curvatures
            
        except Exception as e:
            print(f"Spline fitting failed: {e}")
            # Fall back to finite difference method
            return self._finite_difference_curvature(contour)

    def get_curvature_statistics(self, curvatures: np.ndarray) -> Dict:
        """Calculate statistical measures of curvature.
        
        Args:
            curvatures: Curvature values
            
        Returns:
            Dictionary of statistics
        """
        # Filter out zero values (failed calculations)
        valid_curvatures = curvatures[curvatures != 0]
        
        if len(valid_curvatures) == 0:
            return {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'median': 0,
                'positive_percent': 0,
                'negative_percent': 0,
                'n_valid': 0,
                'n_total': len(curvatures)
            }
        
        # Calculate basic statistics
        stats = {
            'mean': np.mean(valid_curvatures),
            'std': np.std(valid_curvatures),
            'min': np.min(valid_curvatures),
            'max': np.max(valid_curvatures),
            'median': np.median(valid_curvatures),
            'positive_percent': 100 * np.sum(valid_curvatures > 0) / len(valid_curvatures),
            'negative_percent': 100 * np.sum(valid_curvatures < 0) / len(valid_curvatures),
            'n_valid': len(valid_curvatures),
            'n_total': len(curvatures)
        }
        
        return stats

    def identify_high_curvature_regions(self, 
                                      contour: np.ndarray, 
                                      curvatures: np.ndarray, 
                                      threshold: float = None) -> List[Dict]:
        """Identify regions of high curvature.
        
        Args:
            contour: Contour points
            curvatures: Curvature values
            threshold: Curvature threshold (if None, use mean + 1.5*std)
            
        Returns:
            List of dictionaries with region information
        """
        # Set threshold if not provided
        if threshold is None:
            valid_curvatures = curvatures[curvatures != 0]
            if len(valid_curvatures) < 3:
                return []
            threshold = np.mean(valid_curvatures) + 1.5 * np.std(valid_curvatures)
        
        # Find regions above threshold
        high_regions = []
        region_start = None
        
        for i in range(len(curvatures)):
            if abs(curvatures[i]) > threshold:
                if region_start is None:
                    region_start = i
            elif region_start is not None:
                # End of region
                region_end = i - 1
                
                # Only count regions with at least 3 points
                if region_end - region_start + 1 >= 3:
                    # Calculate region properties
                    region_indices = np.arange(region_start, region_end + 1)
                    region_curvatures = curvatures[region_indices]
                    region_points = contour[region_indices]
                    
                    region = {
                        'start_idx': region_start,
                        'end_idx': region_end,
                        'length': region_end - region_start + 1,
                        'mean_curvature': np.mean(region_curvatures),
                        'max_curvature': np.max(np.abs(region_curvatures)),
                        'points': region_points
                    }
                    
                    high_regions.append(region)
                
                region_start = None
        
        # Check if last region wraps around
        if region_start is not None:
            region_end = len(curvatures) - 1
            
            if region_end - region_start + 1 >= 3:
                region_indices = np.arange(region_start, region_end + 1)
                region_curvatures = curvatures[region_indices]
                region_points = contour[region_indices]
                
                region = {
                    'start_idx': region_start,
                    'end_idx': region_end,
                    'length': region_end - region_start + 1,
                    'mean_curvature': np.mean(region_curvatures),
                    'max_curvature': np.max(np.abs(region_curvatures)),
                    'points': region_points
                }
                
                high_regions.append(region)
        
        return high_regions