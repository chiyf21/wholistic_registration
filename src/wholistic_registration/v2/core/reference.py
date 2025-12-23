"""
Reference image computation for registration.

The reference image is computed from a window of frames by:
1. Computing pairwise correlations between all frames
2. Finding the frame most similar to all others (most stable)
3. Averaging the top-N most correlated frames
4. Optionally combining membrane and calcium channels

This creates a reference that represents the "typical" appearance
of the tissue, robust to transient artifacts.
"""

from typing import Tuple, Optional, List
import numpy as np

from ..utils.array_ops import get_array_module, to_numpy, to_gpu, ArrayLike, ensure_contiguous
from ..utils.logging import get_logger
from ..config import ChannelConfig
from .transforms import combine_channels


class ReferenceComputer:
    """Computes reference images from frame windows.
    
    The reference computer finds the most temporally-stable frame
    in a window and averages it with similar frames to create
    a robust reference for registration.
    
    Attributes:
        max_correlation_frames: Maximum frames to use for correlation
        device: Compute device ('cuda' or 'cpu')
        gpu_id: GPU device ID
        
    Example:
        >>> computer = ReferenceComputer(max_correlation_frames=50)
        >>> membrane_window = np.random.rand(20, 10, 100, 100)  # (T, Z, Y, X)
        >>> calcium_window = np.random.rand(20, 10, 100, 100)
        >>> reference = computer.compute(
        ...     membrane_window, calcium_window,
        ...     channel_config=ChannelConfig(dual_channel=True, k=50)
        ... )
    """
    
    def __init__(
        self,
        max_correlation_frames: int = 50,
        spatial_downsample: int = 4,
        device: str = 'cuda',
        gpu_id: int = 0,
    ):
        """Initialize reference computer.
        
        Args:
            max_correlation_frames: Maximum frames for correlation computation
            spatial_downsample: Downsample factor for correlation (speed optimization)
            device: 'cuda' or 'cpu'
            gpu_id: GPU device ID
        """
        self.max_correlation_frames = max_correlation_frames
        self.spatial_downsample = spatial_downsample
        self.device = device
        self.gpu_id = gpu_id
        self.logger = get_logger()
        
        # Initialize array module
        self._am = get_array_module(device, gpu_id)
        self._xp = self._am.xp
    
    def compute(
        self,
        membrane_frames: np.ndarray,
        calcium_frames: Optional[np.ndarray],
        channel_config: ChannelConfig,
    ) -> np.ndarray:
        """Compute reference image from frame window.
        
        Args:
            membrane_frames: (T, Z, Y, X) or (T, Y, X) membrane channel
            calcium_frames: (T, Z, Y, X) or (T, Y, X) calcium channel, or None
            channel_config: Channel configuration (dual_channel, k, transform)
            
        Returns:
            Reference image: (Z, Y, X) or (Y, X)
        """
        xp = self._xp
        
        # Transfer to GPU if needed
        mem_gpu = to_gpu(membrane_frames, xp)
        
        # Find most stable frame and top correlated frames
        mem_ref, top_indices = self._pick_initial_reference(mem_gpu)
        
        # If dual channel, combine with calcium
        if channel_config.dual_channel and calcium_frames is not None:
            ca_gpu = to_gpu(calcium_frames, xp)
            
            # Average calcium from top correlated frames
            ca_ref = xp.mean(ca_gpu[top_indices], axis=0)
            
            # Combine channels
            reference = combine_channels(
                mem_ref,
                ca_ref,
                channel_config.transform,
                channel_config.k,
            )
        else:
            reference = mem_ref
        
        # Return as numpy
        return to_numpy(reference)
    
    def _pick_initial_reference(
        self,
        frames: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Find the most stable frame and top correlated indices.
        
        Algorithm:
        1. Select middle Z slices for efficiency
        2. Spatially downsample for correlation computation
        3. Compute T×T correlation matrix
        4. Find frame with highest mean correlation to others
        5. Get indices of top-N most correlated frames
        6. Average these frames at full resolution
        
        Args:
            frames: (T, Z, Y, X) or (T, Y, X) array on GPU
            
        Returns:
            Tuple of:
                - Reference image at full resolution
                - Indices of top correlated frames
        """
        xp = self._xp
        ndim = frames.ndim
        
        # Step 1: Select middle Z slices for correlation
        if ndim == 4:
            T, Z, Y, X = frames.shape
            if Z >= 3:
                z_mid = Z // 2
                frames_used = frames[:, z_mid-1:z_mid+2, :, :]
            else:
                frames_used = frames
        elif ndim == 3:
            T, Y, X = frames.shape
            frames_used = frames[:, None, :, :]  # Add fake Z dim
        else:
            raise ValueError(f"frames must be 3D or 4D, got {ndim}D")
        
        # Step 2: Spatial downsample for correlation
        ds = self.spatial_downsample
        frames_small = frames_used[:, :, ::ds, ::ds]
        frames_small = ensure_contiguous(frames_small).astype(xp.float32)
        
        # Step 3: Flatten and compute correlation matrix
        T = frames_small.shape[0]
        frames_flat = frames_small.reshape(T, -1)
        
        # Demean each frame
        frames_mean = frames_flat.mean(axis=1, keepdims=True)
        frames_demeaned = frames_flat - frames_mean
        
        # Correlation matrix: (T, T)
        cc = frames_demeaned @ frames_demeaned.T
        
        # Normalize to [-1, 1]
        diag = xp.sqrt(xp.diag(cc)) + 1e-12
        cc = cc / (diag[:, None] * diag[None, :])
        
        # Step 4: Find best frame (highest mean correlation)
        ncorr = min(self.max_correlation_frames, T - 1)
        
        # Sort correlations descending for each row
        cc_sorted = -xp.sort(-cc, axis=1)
        
        # Mean of top correlations (excluding self which is 1.0)
        best_cc = cc_sorted[:, 1:ncorr+1].mean(axis=1)
        
        # Frame with highest mean correlation
        imax = int(xp.argmax(best_cc).item())
        
        # Step 5: Get top correlated frame indices
        indsort = xp.argsort(-cc[imax])
        top_indices = indsort[:ncorr]
        
        # Step 6: Average top frames at full resolution
        ref_img = frames[top_indices].mean(axis=0)
        
        self.logger.debug(
            f"Reference: best frame {imax}, using {ncorr} frames for averaging"
        )
        
        return ref_img, top_indices


def compute_reference_simple(
    membrane_frames: np.ndarray,
    calcium_frames: Optional[np.ndarray] = None,
    dual_channel: bool = True,
    transform: str = 'log10',
    k: float = 50.0,
    device: str = 'cuda',
) -> np.ndarray:
    """Simple function interface for reference computation.
    
    For quick use without creating a ReferenceComputer instance.
    
    Args:
        membrane_frames: (T, Z, Y, X) membrane channel
        calcium_frames: (T, Z, Y, X) calcium channel, or None
        dual_channel: Whether to combine channels
        transform: Calcium transform type
        k: Calcium weight
        device: 'cuda' or 'cpu'
        
    Returns:
        Reference image: (Z, Y, X)
    """
    from ..config import ChannelConfig
    
    computer = ReferenceComputer(device=device)
    config = ChannelConfig(dual_channel=dual_channel, transform=transform, k=k)
    
    return computer.compute(membrane_frames, calcium_frames, config)

