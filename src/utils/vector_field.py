#!/usr/bin/env python3
# src/utils/vector_field.py

import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt

class VectorField:
    """Class for handling 2D vector fields."""
    
    def __init__(self, vectors: np.ndarray, origin_points: Optional[np.ndarray] = None):
        """Initialize the vector field.
        
        Args:
            vectors: Vector field as 2D or 3D array
                    - If 2D: Shape [N, 2] for N vectors (dx, dy)
                    - If 3D: Shape [height, width, 2] for dense vector field
            origin_points: Origin points for vectors (if vectors is 2D)
                          - Shape [N, 2] for N points (x, y)
        """
        self.is_dense = len(vectors.shape) == 3
        self.vectors = vectors
        
        if self.is_dense:
            self.height, self.width = vectors.shape[:2]
            self.origin_points = None
        else:
            if origin_points is None:
                raise ValueError("Origin points must be provided for sparse vector field")
                
            if len(vectors) != len(origin_points):
                raise ValueError("Number of vectors must match number of origin points")
                
            self.vectors = vectors
            self.origin_points = origin_points
            self.height = None
            self.width = None
            
    def get_magnitudes(self) -> np.ndarray:
        """Calculate vector magnitudes.
        
        Returns:
            Array of magnitudes
        """
        if self.is_dense:
            return np.sqrt(self.vectors[:, :, 0]**2 + self.vectors[:, :, 1]**2)
        else:
            return np.sqrt(self.vectors[:, 0]**2 + self.vectors[:, 1]**2)
            
    def get_directions(self) -> np.ndarray:
        """Calculate vector directions in radians.
        
        Returns:
            Array of directions
        """
        if self.is_dense:
            return np.arctan2(self.vectors[:, :, 1], self.vectors[:, :, 0])
        else:
            return np.arctan2(self.vectors[:, 1], self.vectors[:, 0])
            
    def get_component(self, direction: np.ndarray) -> np.ndarray:
        """Calculate vector components along specified direction.
        
        Args:
            direction: Direction vector (x, y)
            
        Returns:
            Array of vector components
        """
        # Normalize direction
        norm = np.sqrt(direction[0]**2 + direction[1]**2)
        if norm < 1e-10:
            raise ValueError("Direction vector cannot be zero")
            
        direction = direction / norm
        
        if self.is_dense:
            return (self.vectors[:, :, 0] * direction[0] + 
                   self.vectors[:, :, 1] * direction[1])
        else:
            return (self.vectors[:, 0] * direction[0] + 
                   self.vectors[:, 1] * direction[1])
                   
    def get_normal_component(self, direction: np.ndarray) -> np.ndarray:
        """Calculate vector components normal to specified direction.
        
        Args:
            direction: Direction vector (x, y)
            
        Returns:
            Array of vector components
        """
        # Normalize direction
        norm = np.sqrt(direction[0]**2 + direction[1]**2)
        if norm < 1e-10:
            raise ValueError("Direction vector cannot be zero")
            
        direction = direction / norm
        
        # Calculate normal direction (90° counterclockwise)
        normal = np.array([-direction[1], direction[0]])
        
        if self.is_dense:
            return (self.vectors[:, :, 0] * normal[0] + 
                   self.vectors[:, :, 1] * normal[1])
        else:
            return (self.vectors[:, 0] * normal[0] + 
                   self.vectors[:, 1] * normal[1])
                   
    def resample(self, points: np.ndarray) -> np.ndarray:
        """Resample dense vector field at specified points.
        
        Args:
            points: Sample points as Nx2 array (x, y)
            
        Returns:
            Vectors at sample points
        """
        if not self.is_dense:
            raise ValueError("Resampling requires dense vector field")
            
        # Check if points are within bounds
        x_coords = np.clip(points[:, 0], 0, self.width - 1)
        y_coords = np.clip(points[:, 1], 0, self.height - 1)
        
        # Simple nearest neighbor interpolation
        x_indices = np.round(x_coords).astype(int)
        y_indices = np.round(y_coords).astype(int)
        
        resampled = self.vectors[y_indices, x_indices]
        return resampled
        
    def visualize(self, background: Optional[np.ndarray] = None, 
                 scale: float = 1.0, skip: int = 1) -> plt.Figure:
        """Create visualization of vector field.
        
        Args:
            background: Optional background image
            scale: Scaling factor for vectors
            skip: Skip factor for sparser visualization
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show background if provided
        if background is not None:
            ax.imshow(background, cmap='gray')
        
        # Plot vectors
        if self.is_dense:
            # Create grid of sampling points
            y_indices, x_indices = np.indices((self.height, self.width))
            
            # Skip points for clearer visualization
            x_coords = x_indices[::skip, ::skip].flatten()
            y_coords = y_indices[::skip, ::skip].flatten()
            
            # Get vectors at sampled points
            u = self.vectors[::skip, ::skip, 0].flatten()
            v = self.vectors[::skip, ::skip, 1].flatten()
            
            # Plot vector field
            ax.quiver(x_coords, y_coords, u, v, scale=scale, 
                     color='y', width=0.002, headwidth=5)
        else:
            # Skip points for clearer visualization
            x = self.origin_points[::skip, 0]
            y = self.origin_points[::skip, 1]
            u = self.vectors[::skip, 0]
            v = self.vectors[::skip, 1]
            
            # Plot vector field
            ax.quiver(x, y, u, v, scale=scale, 
                     color='y', width=0.002, headwidth=5)
        
        # Set aspect ratio and labels
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Vector Field Visualization')
        
        return fig
        
    def to_dense(self, shape: Tuple[int, int], 
                interpolation: str = 'linear') -> 'VectorField':
        """Convert sparse vector field to dense.
        
        Args:
            shape: Output shape (height, width)
            interpolation: Interpolation method ('nearest' or 'linear')
            
        Returns:
            Dense vector field
        """
        if self.is_dense:
            return self
            
        from scipy.interpolate import griddata
        
        # Create dense grid
        height, width = shape
        y_indices, x_indices = np.mgrid[0:height, 0:width]
        
        # Interpolate x and y components separately
        vx = griddata(self.origin_points, self.vectors[:, 0], 
                     (x_indices, y_indices), method=interpolation,
                     fill_value=0)
        vy = griddata(self.origin_points, self.vectors[:, 1], 
                     (x_indices, y_indices), method=interpolation,
                     fill_value=0)
        
        # Combine components
        dense_vectors = np.stack([vx, vy], axis=-1)
        
        return VectorField(dense_vectors)
        
    def to_sparse(self, n_points: int = 1000, 
                 method: str = 'random') -> 'VectorField':
        """Convert dense vector field to sparse.
        
        Args:
            n_points: Number of sample points
            method: Sampling method ('random', 'grid', or 'uniform')
            
        Returns:
            Sparse vector field
        """
        if not self.is_dense:
            return self
            
        if method == 'random':
            # Random sampling
            y_indices = np.random.randint(0, self.height, n_points)
            x_indices = np.random.randint(0, self.width, n_points)
            
        elif method == 'grid':
            # Grid sampling
            grid_size = int(np.sqrt(n_points))
            y_step = max(1, self.height // grid_size)
            x_step = max(1, self.width // grid_size)
            
            y_indices, x_indices = np.mgrid[0:self.height:y_step, 0:self.width:x_step]
            y_indices = y_indices.flatten()[:n_points]
            x_indices = x_indices.flatten()[:n_points]
            
        elif method == 'uniform':
            # Uniform sampling
            n_points_sqrt = int(np.sqrt(n_points))
            y_indices = np.linspace(0, self.height-1, n_points_sqrt).astype(int)
            x_indices = np.linspace(0, self.width-1, n_points_sqrt).astype(int)
            
            # Create mesh grid
            y_indices, x_indices = np.meshgrid(y_indices, x_indices)
            y_indices = y_indices.flatten()
            x_indices = x_indices.flatten()
            
        else:
            raise ValueError(f"Unknown sampling method: {method}")
            
        # Get vectors at sample points
        vectors = self.vectors[y_indices, x_indices]
        
        # Create origin points
        origin_points = np.column_stack([x_indices, y_indices])
        
        return VectorField(vectors, origin_points)
        
    def calculate_divergence(self) -> np.ndarray:
        """Calculate divergence of vector field.
        
        Returns:
            Divergence field
        """
        if not self.is_dense:
            raise ValueError("Divergence calculation requires dense vector field")
            
        # Calculate gradients
        vx = self.vectors[:, :, 0]
        vy = self.vectors[:, :, 1]
        
        # Use central differences
        dvx_dx = np.zeros_like(vx)
        dvy_dy = np.zeros_like(vy)
        
        # X gradient
        dvx_dx[:, 1:-1] = (vx[:, 2:] - vx[:, :-2]) / 2
        
        # Y gradient
        dvy_dy[1:-1, :] = (vy[2:, :] - vy[:-2, :]) / 2
        
        # Handle boundaries
        dvx_dx[:, 0] = vx[:, 1] - vx[:, 0]
        dvx_dx[:, -1] = vx[:, -1] - vx[:, -2]
        
        dvy_dy[0, :] = vy[1, :] - vy[0, :]
        dvy_dy[-1, :] = vy[-1, :] - vy[-2, :]
        
        return dvx_dx + dvy_dy
        
    def calculate_curl(self) -> np.ndarray:
        """Calculate curl (vorticity) of vector field.
        
        Returns:
            Curl field
        """
        if not self.is_dense:
            raise ValueError("Curl calculation requires dense vector field")
            
        # Calculate gradients
        vx = self.vectors[:, :, 0]
        vy = self.vectors[:, :, 1]
        
        # Use central differences
        dvx_dy = np.zeros_like(vx)
        dvy_dx = np.zeros_like(vy)
        
        # X gradient of vy
        dvy_dx[:, 1:-1] = (vy[:, 2:] - vy[:, :-2]) / 2
        
        # Y gradient of vx
        dvx_dy[1:-1, :] = (vx[2:, :] - vx[:-2, :]) / 2
        
        # Handle boundaries
        dvy_dx[:, 0] = vy[:, 1] - vy[:, 0]
        dvy_dx[:, -1] = vy[:, -1] - vy[:, -2]
        
        dvx_dy[0, :] = vx[1, :] - vx[0, :]
        dvx_dy[-1, :] = vx[-1, :] - vx[-2, :]
        
        # Curl = dvy/dx - dvx/dy
        return dvy_dx - dvx_dy
        
    def smooth(self, sigma: float = 1.0) -> 'VectorField':
        """Apply Gaussian smoothing to vector field.
        
        Args:
            sigma: Gaussian sigma parameter
            
        Returns:
            Smoothed vector field
        """
        if not self.is_dense:
            raise ValueError("Smoothing requires dense vector field")
            
        from scipy.ndimage import gaussian_filter
        
        # Smooth x and y components separately
        vx_smooth = gaussian_filter(self.vectors[:, :, 0], sigma)
        vy_smooth = gaussian_filter(self.vectors[:, :, 1], sigma)
        
        # Combine smoothed components
        smoothed_vectors = np.stack([vx_smooth, vy_smooth], axis=-1)
        
        return VectorField(smoothed_vectors)