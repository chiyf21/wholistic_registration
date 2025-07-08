"""
Common utilities for wbi module with graceful CuPy/NumPy fallback.
"""

import numpy as np

# Try to import CuPy, fallback to NumPy if not available
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy is available - using GPU acceleration")
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
    print("CuPy not available - falling back to NumPy (CPU only)")

# Try to import CuPy SciPy modules, fallback to regular SciPy
try:
    import cupyx.scipy.ndimage as cupy_ndimage
    CUPYX_NDIMAGE_AVAILABLE = True
except ImportError:
    import scipy.ndimage as cupy_ndimage
    CUPYX_NDIMAGE_AVAILABLE = False

try:
    from cupyx.scipy.interpolate import RegularGridInterpolator as CupyRegularGridInterpolator
    CUPYX_INTERPOLATE_AVAILABLE = True
except ImportError:
    from scipy.interpolate import RegularGridInterpolator as CupyRegularGridInterpolator
    CUPYX_INTERPOLATE_AVAILABLE = False

# Create aliases for easier imports
Gimage = cupy_ndimage
RegularGridInterpolator = CupyRegularGridInterpolator

# Export the main variables for import
__all__ = ['cp', 'Gimage', 'RegularGridInterpolator', 'CUPY_AVAILABLE', 'CUPYX_NDIMAGE_AVAILABLE', 'CUPYX_INTERPOLATE_AVAILABLE', 'option']

option={
    'layer':3,
    'iter':10,
    'r':5,
    'zRatio':27.693,
    'motion':0,
    'mask_ref':0,
    'mask_mov':0
}