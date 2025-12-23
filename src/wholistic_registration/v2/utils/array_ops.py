"""
Array operations with GPU/CPU backend abstraction.

This module provides a unified interface for array operations that works
with both NumPy (CPU) and CuPy (GPU) backends.
"""

from typing import Union, Literal, Optional
from dataclasses import dataclass
import numpy as np

# Type alias for array types
ArrayLike = Union[np.ndarray, "cp.ndarray"]


@dataclass
class ArrayModule:
    """Container for array module and device info.
    
    Attributes:
        xp: The array module (numpy or cupy)
        device: Device type ('cpu' or 'cuda')
        gpu_id: GPU device ID if using CUDA
    """
    xp: any
    device: Literal['cpu', 'cuda']
    gpu_id: Optional[int] = None
    
    def __repr__(self):
        if self.device == 'cuda':
            return f"ArrayModule(device='cuda:{self.gpu_id}')"
        return "ArrayModule(device='cpu')"


_current_backend: Optional[ArrayModule] = None


def get_array_module(
    device: Literal['cuda', 'cpu'] = 'cuda',
    gpu_id: int = 0
) -> ArrayModule:
    """Get the array module for the specified device.
    
    Args:
        device: 'cuda' for GPU or 'cpu' for CPU
        gpu_id: GPU device ID when using CUDA
        
    Returns:
        ArrayModule with the appropriate array library (numpy or cupy)
        
    Example:
        >>> am = get_array_module('cuda', gpu_id=0)
        >>> x = am.xp.zeros((100, 100))  # Creates CuPy array on GPU 0
    """
    global _current_backend
    
    if device == 'cuda':
        try:
            import cupy as cp
            cp.cuda.Device(gpu_id).use()
            _current_backend = ArrayModule(xp=cp, device='cuda', gpu_id=gpu_id)
            return _current_backend
        except ImportError:
            raise ImportError(
                "CuPy is required for CUDA backend. "
                "Install with: pip install cupy-cuda11x (adjust for your CUDA version)"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CUDA device {gpu_id}: {e}")
    else:
        _current_backend = ArrayModule(xp=np, device='cpu', gpu_id=None)
        return _current_backend


def to_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert array to NumPy, handling both CPU and GPU arrays.
    
    Args:
        arr: Input array (NumPy or CuPy)
        
    Returns:
        NumPy array (copied from GPU if necessary)
    """
    if isinstance(arr, np.ndarray):
        return arr
    
    # Check if it's a CuPy array
    if hasattr(arr, 'get'):
        return arr.get()
    
    # Fallback: try to convert via numpy
    return np.asarray(arr)


def to_gpu(arr: np.ndarray, xp=None) -> ArrayLike:
    """Convert NumPy array to GPU array.
    
    Args:
        arr: Input NumPy array
        xp: Array module (cupy). If None, uses current backend.
        
    Returns:
        GPU array (CuPy) or original array if CPU backend
    """
    if xp is None:
        if _current_backend is None:
            raise RuntimeError("No backend initialized. Call get_array_module() first.")
        xp = _current_backend.xp
    
    if xp.__name__ == 'numpy':
        return arr
    
    return xp.asarray(arr)


def ensure_contiguous(arr: ArrayLike) -> ArrayLike:
    """Ensure array is contiguous in memory.
    
    This is important for GPU operations and some numerical algorithms.
    """
    if hasattr(arr, 'flags'):
        if not arr.flags['C_CONTIGUOUS']:
            if hasattr(arr, 'get'):  # CuPy
                import cupy as cp
                return cp.ascontiguousarray(arr)
            else:
                return np.ascontiguousarray(arr)
    return arr


def free_gpu_memory():
    """Free unused GPU memory.
    
    Call this periodically when processing large datasets to prevent
    GPU memory exhaustion.
    """
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    except ImportError:
        pass  # No CuPy, nothing to free

