"""Input validation utilities."""

from typing import Tuple, Optional, List
import numpy as np


def validate_volume_shape(
    arr: np.ndarray,
    expected_ndim: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    name: str = "array",
) -> None:
    """Validate array shape.
    
    Args:
        arr: Array to validate
        expected_ndim: Expected number of dimensions
        expected_shape: Expected shape (use -1 for any size in a dimension)
        name: Name for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(
            f"{name} has {arr.ndim} dimensions, expected {expected_ndim}. "
            f"Shape: {arr.shape}"
        )
    
    if expected_shape is not None:
        for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValueError(
                    f"{name} dimension {i} has size {actual}, expected {expected}. "
                    f"Shape: {arr.shape}, expected: {expected_shape}"
                )


def validate_frame_range(
    frames: List[int],
    total_frames: int,
    name: str = "frames",
) -> None:
    """Validate that frame indices are within bounds.
    
    Args:
        frames: List of frame indices
        total_frames: Total number of frames in dataset
        name: Name for error messages
        
    Raises:
        ValueError: If any frame is out of bounds
    """
    if not frames:
        raise ValueError(f"{name} list is empty")
    
    min_frame = min(frames)
    max_frame = max(frames)
    
    if min_frame < 0:
        raise ValueError(f"{name} contains negative index: {min_frame}")
    
    if max_frame >= total_frames:
        raise ValueError(
            f"{name} contains index {max_frame} >= total_frames ({total_frames})"
        )


def validate_dtype(
    arr: np.ndarray,
    expected_dtypes: Tuple[np.dtype, ...],
    name: str = "array",
) -> None:
    """Validate array dtype.
    
    Args:
        arr: Array to validate
        expected_dtypes: Tuple of acceptable dtypes
        name: Name for error messages
        
    Raises:
        ValueError: If dtype is not in expected list
    """
    if arr.dtype not in expected_dtypes:
        raise ValueError(
            f"{name} has dtype {arr.dtype}, expected one of {expected_dtypes}"
        )

