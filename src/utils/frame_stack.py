#!/usr/bin/env python3
# src/utils/frame_stack.py

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
from .image_utils import load_tiff_sequence

class FrameStack:
    """Class for handling time-series image stacks."""
    
    def __init__(self, frames: Optional[List[np.ndarray]] = None, 
                filename: Optional[str] = None,
                time_delta: float = 1.0):
        """Initialize the frame stack.
        
        Args:
            frames: List of image frames (optional)
            filename: Source filename (optional)
            time_delta: Time between frames in seconds
        """
        self.frames = frames if frames is not None else []
        self.filename = filename
        self.time_delta = time_delta
        self.current_index = 0
        self.metadata = {}
        
    def load_from_file(self, file_path: str) -> bool:
        """Load frames from file.
        
        Args:
            file_path: Path to first file in sequence or multi-page TIFF
            
        Returns:
            Success flag
        """
        try:
            # Load sequence
            frames = load_tiff_sequence(file_path)
            
            if not frames:
                return False
                
            self.frames = frames
            self.filename = os.path.basename(file_path)
            self.current_index = 0
            
            return True
            
        except Exception as e:
            print(f"Error loading frame stack: {e}")
            return False
            
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame.
        
        Returns:
            Current frame or None if stack is empty
        """
        if not self.frames:
            return None
            
        return self.frames[self.current_index]
        
    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Get frame at specified index.
        
        Args:
            index: Frame index
            
        Returns:
            Frame at index or None if index is out of range
        """
        if not self.frames or index < 0 or index >= len(self.frames):
            return None
            
        return self.frames[index]
        
    def next_frame(self) -> Optional[np.ndarray]:
        """Get next frame and advance current index.
        
        Returns:
            Next frame or None if at end of stack
        """
        if not self.frames or self.current_index >= len(self.frames) - 1:
            return None
            
        self.current_index += 1
        return self.frames[self.current_index]
        
    def previous_frame(self) -> Optional[np.ndarray]:
        """Get previous frame and decrement current index.
        
        Returns:
            Previous frame or None if at beginning of stack
        """
        if not self.frames or self.current_index <= 0:
            return None
            
        self.current_index -= 1
        return self.frames[self.current_index]
        
    def set_current_index(self, index: int) -> bool:
        """Set current frame index.
        
        Args:
            index: New frame index
            
        Returns:
            Success flag
        """
        if not self.frames or index < 0 or index >= len(self.frames):
            return False
            
        self.current_index = index
        return True
        
    def get_num_frames(self) -> int:
        """Get number of frames in the stack.
        
        Returns:
            Number of frames
        """
        return len(self.frames)
        
    def is_empty(self) -> bool:
        """Check if the stack is empty.
        
        Returns:
            True if stack is empty, False otherwise
        """
        return len(self.frames) == 0
        
    def calculate_frame_difference(self, index1: int, index2: int) -> Optional[np.ndarray]:
        """Calculate difference between two frames.
        
        Args:
            index1: First frame index
            index2: Second frame index
            
        Returns:
            Difference image or None if indices are out of range
        """
        if (not self.frames or index1 < 0 or index1 >= len(self.frames) or
            index2 < 0 or index2 >= len(self.frames)):
            return None
            
        frame1 = self.frames[index1]
        frame2 = self.frames[index2]
        
        # Handle different dtypes
        if frame1.dtype != frame2.dtype:
            # Convert to float for consistent subtraction
            f1 = frame1.astype(np.float32)
            f2 = frame2.astype(np.float32)
            
            # Normalize if needed
            if f1.max() > 1.0:
                f1 = f1 / 255.0
            if f2.max() > 1.0:
                f2 = f2 / 255.0
                
            diff = f2 - f1
        else:
            diff = frame2.astype(np.float32) - frame1.astype(np.float32)
            
        return diff
        
    def get_time_series(self, x: int, y: int, window_size: int = 1) -> np.ndarray:
        """Extract time series for a pixel or small region.
        
        Args:
            x: X coordinate
            y: Y coordinate
            window_size: Size of averaging window
            
        Returns:
            Time series array
        """
        if not self.frames:
            return np.array([])
            
        # Check if coordinates are valid
        if (x < 0 or x >= self.frames[0].shape[1] or
            y < 0 or y >= self.frames[0].shape[0]):
            return np.array([])
            
        # Extract time series
        half_window = window_size // 2
        time_series = []
        
        for frame in self.frames:
            # Define region bounds with border checking
            x_min = max(0, x - half_window)
            x_max = min(frame.shape[1] - 1, x + half_window)
            y_min = max(0, y - half_window)
            y_max = min(frame.shape[0] - 1, y + half_window)
            
            # Calculate mean value in window
            region = frame[y_min:y_max+1, x_min:x_max+1]
            time_series.append(np.mean(region))
            
        return np.array(time_series)
        
    def get_max_projection(self) -> Optional[np.ndarray]:
        """Create maximum intensity projection of the stack.
        
        Returns:
            Maximum projection or None if stack is empty
        """
        if not self.frames:
            return None
            
        return np.max(self.frames, axis=0)
        
    def get_mean_projection(self) -> Optional[np.ndarray]:
        """Create mean intensity projection of the stack.
        
        Returns:
            Mean projection or None if stack is empty
        """
        if not self.frames:
            return None
            
        return np.mean(self.frames, axis=0)
        
    def get_std_projection(self) -> Optional[np.ndarray]:
        """Create standard deviation projection of the stack.
        
        Returns:
            Standard deviation projection or None if stack is empty
        """
        if not self.frames:
            return None
            
        return np.std(self.frames, axis=0)
        
    def save_metadata(self, key: str, value: Any):
        """Save metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata.
        
        Args:
            key: Metadata key
            default: Default value if key doesn't exist
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)