"""
Common utilities for wbi module with graceful CuPy/NumPy fallback.
"""

import numpy as np

# Try to import CuPy, fallback to NumPy if not available
CUPY_AVAILABLE = False
cp_original = None

try:
    import cupy as cp_original
    # Test if CUDA is actually available
    try:
        cp_original.cuda.Device(0).compute_capability
        cp = cp_original
        CUPY_AVAILABLE = True
        print("CuPy is available with CUDA - using GPU acceleration")
    except cp_original.cuda.runtime.CUDARuntimeError:
        # CuPy is installed but no CUDA device available
        cp = np
        CUPY_AVAILABLE = False
        print("CuPy installed but no CUDA device available - falling back to NumPy (CPU only)")
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
    print("CuPy not available - falling back to NumPy (CPU only)")

# Try to import CuPy SciPy modules, fallback to regular SciPy
if CUPY_AVAILABLE and cp_original is not None:
    try:
        import cupyx.scipy.ndimage as cupy_ndimage
        CUPYX_NDIMAGE_AVAILABLE = True
    except (ImportError, cp_original.cuda.runtime.CUDARuntimeError):
        import scipy.ndimage as cupy_ndimage
        CUPYX_NDIMAGE_AVAILABLE = False

    try:
        from cupyx.scipy.interpolate import RegularGridInterpolator as CupyRegularGridInterpolator
        CUPYX_INTERPOLATE_AVAILABLE = True
    except (ImportError, cp_original.cuda.runtime.CUDARuntimeError):
        from scipy.interpolate import RegularGridInterpolator as CupyRegularGridInterpolator
        CUPYX_INTERPOLATE_AVAILABLE = False
else:
    print("CuPy not available - import standardscipy")
    import scipy.ndimage as cupy_ndimage
    CUPYX_NDIMAGE_AVAILABLE = False
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
    'mask_mov':0,
    'save_ite':100,
    'thresFactor':5,
    'mask_size_range':[5,500],
    'smoothPenalty_raw':0.05,
    'tol':1e-4
}