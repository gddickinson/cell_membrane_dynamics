#!/usr/bin/env python3
# src/utils/image_utils.py

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from skimage import filters, morphology, measure

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    # Handle binary images
    if image.dtype == bool:
        return image.astype(np.float32)
        
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.float32)
        
    return ((image - img_min) / (img_max - img_min)).astype(np.float32)

def convert_to_8bit(image: np.ndarray) -> np.ndarray:
    """Convert image to 8-bit for display.
    
    Args:
        image: Input image
        
    Returns:
        8-bit image
    """
    if image.dtype == np.uint8:
        return image
        
    # Normalize and convert to 8-bit
    normalized = normalize_image(image)
    return (normalized * 255).astype(np.uint8)

def segment_cell(image: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Segment cell from grayscale image.
    
    Args:
        image: Grayscale image
        min_size: Minimum object size
        
    Returns:
        Binary mask
    """
    # Normalize image
    normalized = normalize_image(image)
    
    # Apply Gaussian filter to reduce noise
    smoothed = filters.gaussian(normalized, sigma=1.0)
    
    # Calculate threshold using Otsu's method
    threshold = filters.threshold_otsu(smoothed)
    binary = smoothed > threshold
    
    # Clean up small objects and holes
    clean_mask = morphology.remove_small_objects(binary, min_size=min_size)
    clean_mask = morphology.remove_small_holes(clean_mask, area_threshold=min_size)
    
    # Keep largest object only
    labels = measure.label(clean_mask)
    if labels.max() > 0:
        largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
        clean_mask = labels == largest_label
    
    return clean_mask.astype(np.uint8)

def enhance_contrast(image: np.ndarray, percentile_low: float = 1, percentile_high: float = 99) -> np.ndarray:
    """Enhance image contrast using percentile-based normalization.
    
    Args:
        image: Input image
        percentile_low: Lower percentile for contrast stretching
        percentile_high: Upper percentile for contrast stretching
        
    Returns:
        Contrast-enhanced image
    """
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)
    
    if p_high == p_low:
        return normalize_image(image)
        
    enhanced = np.clip(image, p_low, p_high)
    return normalize_image(enhanced)

def create_composite_image(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create a composite image by blending two images.
    
    Args:
        image1: First image
        image2: Second image
        alpha: Blending factor (0-1)
        
    Returns:
        Blended image
    """
    # Convert to 8-bit if needed
    img1_8bit = convert_to_8bit(image1)
    img2_8bit = convert_to_8bit(image2)
    
    # Convert to color if grayscale
    if len(img1_8bit.shape) == 2:
        img1_8bit = cv2.cvtColor(img1_8bit, cv2.COLOR_GRAY2BGR)
    if len(img2_8bit.shape) == 2:
        img2_8bit = cv2.cvtColor(img2_8bit, cv2.COLOR_GRAY2BGR)
    
    # Make sure images have same dimensions
    if img1_8bit.shape != img2_8bit.shape:
        # Resize second image to match first
        img2_8bit = cv2.resize(img2_8bit, (img1_8bit.shape[1], img1_8bit.shape[0]))
    
    # Blend images
    blended = cv2.addWeighted(img1_8bit, alpha, img2_8bit, 1-alpha, 0)
    return blended

def load_tiff_sequence(first_file_path: str) -> List[np.ndarray]:
    """Load a sequence of TIFF images starting with the given file.
    
    Args:
        first_file_path: Path to first file in sequence
        
    Returns:
        List of images
    """
    import os
    import re
    
    directory = os.path.dirname(first_file_path)
    filename = os.path.basename(first_file_path)
    
    # Try to load as multi-page TIFF first
    try:
        import tifffile
        with tifffile.TiffFile(first_file_path) as tif:
            num_pages = len(tif.pages)
            if num_pages > 1:
                # Load multi-page TIFF
                return [page.asarray() for page in tif.pages]
    except (ImportError, Exception):
        # Fall back to searching for sequence
        pass
    
    # Extract the numbering pattern from the filename
    match = re.search(r'(\d+)', filename)
    if not match:
        # No number in filename, just load the single file
        return [cv2.imread(first_file_path, cv2.IMREAD_ANYDEPTH)]
    
    # Get the numerical part and its position
    num_str = match.group(1)
    num_pos = match.start(1)
    prefix = filename[:num_pos]
    suffix = filename[num_pos + len(num_str):]
    num_digits = len(num_str)
    num_value = int(num_str)
    
    # Search for files with the same pattern
    sequence = []
    i = 0
    
    while True:
        curr_num = num_value + i
        curr_filename = f"{prefix}{curr_num:0{num_digits}d}{suffix}"
        curr_path = os.path.join(directory, curr_filename)
        
        if os.path.exists(curr_path):
            img = cv2.imread(curr_path, cv2.IMREAD_ANYDEPTH)
            sequence.append(img)
            i += 1
        else:
            break
    
    return sequence

def overlay_contour(image: np.ndarray, contour: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), 
                  thickness: int = 2) -> np.ndarray:
    """Overlay contour on image.
    
    Args:
        image: Input image
        contour: Contour points as Nx2 array
        color: RGB color tuple
        thickness: Line thickness
        
    Returns:
        Image with overlaid contour
    """
    # Convert to 8-bit if needed
    img_8bit = convert_to_8bit(image)
    
    # Convert to color if grayscale
    if len(img_8bit.shape) == 2:
        img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    
    # Draw contour
    contour_img = img_8bit.copy()
    cv2.drawContours(contour_img, [contour.astype(np.int32)], 0, color, thickness)
    
    return contour_img

def overlay_vectors(image: np.ndarray, points: np.ndarray, vectors: np.ndarray, 
                   color: Tuple[int, int, int] = (0, 0, 255), scale: float = 1.0, 
                   thickness: int = 1, skip: int = 1) -> np.ndarray:
    """Overlay vectors on image.
    
    Args:
        image: Input image
        points: Start points as Nx2 array
        vectors: Vectors as Nx2 array
        color: RGB color tuple
        scale: Scaling factor for vectors
        thickness: Line thickness
        skip: Skip factor for sparser visualization
        
    Returns:
        Image with overlaid vectors
    """
    # Convert to 8-bit if needed
    img_8bit = convert_to_8bit(image)
    
    # Convert to color if grayscale
    if len(img_8bit.shape) == 2:
        img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    
    # Draw vectors
    vector_img = img_8bit.copy()
    
    for i in range(0, len(points), skip):
        start_point = tuple(points[i].astype(int))
        
        # Calculate end point
        end_x = int(start_point[0] + vectors[i, 0] * scale)
        end_y = int(start_point[1] + vectors[i, 1] * scale)
        end_point = (end_x, end_y)
        
        # Check if vector is non-zero and within image bounds
        if (vectors[i, 0]**2 + vectors[i, 1]**2 > 1e-6 and
            0 <= start_point[0] < img_8bit.shape[1] and
            0 <= start_point[1] < img_8bit.shape[0] and
            0 <= end_point[0] < img_8bit.shape[1] and
            0 <= end_point[1] < img_8bit.shape[0]):
            
            # Draw arrow
            cv2.arrowedLine(vector_img, start_point, end_point, color, thickness, tipLength=0.3)
    
    return vector_img

def create_colored_mask(mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), 
                       alpha: float = 0.5) -> np.ndarray:
    """Create a colored mask for overlay.
    
    Args:
        mask: Binary mask
        color: RGB color tuple
        alpha: Transparency (0-1)
        
    Returns:
        Colored mask with alpha channel
    """
    # Create colored mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    
    # Set RGB channels
    colored_mask[mask > 0, 0] = color[0]
    colored_mask[mask > 0, 1] = color[1]
    colored_mask[mask > 0, 2] = color[2]
    
    # Set alpha channel
    colored_mask[mask > 0, 3] = int(255 * alpha)
    
    return colored_mask