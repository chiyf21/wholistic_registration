"""
Frame registration using pyramid-based motion estimation.

This module wraps the existing motion estimation algorithm with a clean interface.
The algorithm uses:
- Multi-scale pyramid processing (coarse-to-fine)
- Patch-based motion computation
- Smoothness regularization

**NOTE ON ALGORITHM PRESERVATION**:
This module wraps the existing calFlow3d_Wei_v1 algorithm without modification
to the core computation. Only the interface is cleaned up.
"""

from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from ..utils.array_ops import get_array_module, to_numpy, to_gpu, free_gpu_memory
from ..utils.logging import get_logger
from ..config import PyramidConfig, MaskConfig, ChannelConfig
from .transforms import combine_channels


@dataclass
class RegistrationResult:
    """Result of registering a batch of frames.
    
    Attributes:
        membrane_registered: Registered membrane channel (T, Z, Y, X)
        calcium_registered: Registered calcium channel (T, Z, Y, X)
        motion_fields: Motion vectors (T, Z, Y, X, 3) or None
        errors: Per-frame registration errors
        reference: Reference image used
    """
    membrane_registered: np.ndarray
    calcium_registered: np.ndarray
    motion_fields: Optional[np.ndarray]
    errors: list
    reference: np.ndarray


class FrameRegistrar:
    """Registers frames to a reference image.
    
    Uses pyramid-based optical flow estimation to compute motion fields
    and warp images to align with the reference.
    
    Example:
        >>> registrar = FrameRegistrar(pyramid_config, mask_config, channel_config)
        >>> result = registrar.register_batch(
        ...     membrane_frames, calcium_frames, reference
        ... )
    """
    
    def __init__(
        self,
        pyramid: PyramidConfig,
        mask: MaskConfig,
        channels: ChannelConfig,
        device: str = 'cuda',
        gpu_id: int = 0,
    ):
        """Initialize registrar.
        
        Args:
            pyramid: Pyramid registration settings
            mask: Outlier masking settings
            channels: Channel combination settings
            device: 'cuda' or 'cpu'
            gpu_id: GPU device ID
        """
        self.pyramid = pyramid
        self.mask = mask
        self.channels = channels
        self.device = device
        self.gpu_id = gpu_id
        self.logger = get_logger()
        
        # Initialize array module
        self._am = get_array_module(device, gpu_id)
        self._xp = self._am.xp
        
        # Import the existing algorithm
        # This preserves the original implementation
        self._import_legacy_modules()
    
    def _import_legacy_modules(self):
        """Import legacy motion estimation modules.
        
        NOTE: These are the original algorithm implementations.
        We wrap them without modification.
        """
        try:
            from wholistic_registration.utils import calFlow3d_Wei_v1
            from wholistic_registration.utils import mask as mask_module
            from wholistic_registration.utils import prep
            self._calflow = calFlow3d_Wei_v1
            self._mask_module = mask_module
            self._prep = prep
            self.logger.debug("Loaded legacy motion estimation modules")
        except ImportError as e:
            self.logger.warning(f"Could not import legacy modules: {e}")
            self.logger.warning("Registration will not work without legacy modules")
            self._calflow = None
            self._mask_module = None
            self._prep = None
    
    def register_batch(
        self,
        membrane_frames: np.ndarray,
        calcium_frames: np.ndarray,
        reference: np.ndarray,
        initial_motion: Optional[np.ndarray] = None,
        return_motion: bool = False,
        verbose: bool = False,
        start_frame_idx: int = 0,
    ) -> RegistrationResult:
        """Register a batch of frames to a reference.
        
        Args:
            membrane_frames: (T, Z, Y, X) membrane channel
            calcium_frames: (T, Z, Y, X) calcium channel
            reference: (Z, Y, X) reference image
            initial_motion: Optional initial motion estimate
            return_motion: Whether to return motion fields
            verbose: Print per-frame progress
            start_frame_idx: Starting frame index (for logging)
            
        Returns:
            RegistrationResult with registered data
            
        **ALGORITHM NOTES**:
        - Uses the original calFlow3d_Wei_v1.getMotion() for motion estimation
        - Motion is computed on combined channel if dual_channel=True
        - Both membrane and calcium are warped using the same motion field
        """
        if self._calflow is None:
            raise RuntimeError(
                "Legacy motion estimation modules not available. "
                "Make sure wholistic_registration.utils is importable."
            )
        
        xp = self._xp
        T = membrane_frames.shape[0]
        is_3d = membrane_frames.ndim == 4
        
        # Prepare reference
        if is_3d:
            # Original code expected (X, Y, Z) order for 3D
            ref_transposed = reference.transpose(2, 1, 0)
        else:
            ref_transposed = reference.T
        
        # Get masks for reference
        ref_mask = self._mask_module.getMask(ref_transposed, self.mask.threshold_factor)
        ref_mask = self._mask_module.bwareafilt3_wei(ref_mask, list(self.mask.intensity_range))
        
        # Build options dict for legacy algorithm
        option = {
            'r': self.pyramid.patch_radius,
            'layer': self.pyramid.layers,
            'iter': self.pyramid.iterations,
            'movRange': self.pyramid.movement_range,
            'mask_ref': ref_mask,
        }
        
        # Compute smoothness penalty normalization
        pnlt_factor = self._prep.getSmPnltNormFctr(ref_transposed, option)
        smooth_penalty = pnlt_factor * self.pyramid.smooth_penalty
        
        # Initialize outputs
        mem_registered = []
        ca_registered = []
        motion_fields = [] if return_motion else None
        errors = []
        
        # Initialize motion
        if initial_motion is None:
            if is_3d:
                current_motion = np.zeros([*ref_transposed.shape, 2, 3])
            else:
                current_motion = np.zeros([*ref_transposed.shape, 2, 3])
        else:
            current_motion = initial_motion
        
        option['motion'] = current_motion
        
        # Process each frame
        for i in range(T):
            frame_idx = start_frame_idx + i
            
            # Get current frame
            mem_frame = membrane_frames[i]
            ca_frame = calcium_frames[i]
            
            # Transpose to legacy format
            if is_3d:
                mem_t = mem_frame.transpose(2, 1, 0)
                ca_t = ca_frame.transpose(2, 1, 0)
            else:
                mem_t = mem_frame.T
                ca_t = ca_frame.T
            
            # Combine channels for registration
            if self.channels.dual_channel:
                ca_transformed = combine_channels(
                    np.zeros_like(ca_t), ca_t,
                    self.channels.transform, self.channels.k
                )
                moving = mem_t + ca_transformed
            else:
                moving = mem_t
            
            # Get mask for moving image
            option['mask_mov'] = self._mask_module.getMask(moving, self.mask.threshold_factor)
            option['mask_mov'] = self._mask_module.bwareafilt3_wei(
                option['mask_mov'], list(self.mask.intensity_range)
            )
            
            # Compute motion
            # NOTE: This is the original algorithm, unchanged
            motion, _, new_coords, error_logs = self._calflow.getMotion(
                moving, ref_transposed, smooth_penalty, option
            )
            
            # Apply motion correction
            mem_corrected = self._calflow.correctMotion(mem_t, motion)
            ca_corrected = self._calflow.correctMotion(ca_t, motion)
            
            # Compute error
            moving_corrected = self._calflow.correctMotion(moving, motion)
            initial_error = np.mean((moving - ref_transposed) ** 2)
            final_error = np.mean((moving_corrected - ref_transposed) ** 2)
            
            if verbose:
                self.logger.info(
                    f"Frame {frame_idx}: error {initial_error:.4f} → {final_error:.4f} "
                    f"({100*(1-final_error/initial_error):.1f}% reduction)"
                )
            
            # Transpose back
            if is_3d:
                mem_registered.append(mem_corrected.transpose(2, 1, 0))
                ca_registered.append(ca_corrected.transpose(2, 1, 0))
                if return_motion:
                    motion_fields.append(motion.transpose(2, 1, 0, 3, 4))
            else:
                mem_registered.append(mem_corrected.T)
                ca_registered.append(ca_corrected.T)
                if return_motion:
                    motion_fields.append(motion)
            
            errors.append({'initial': initial_error, 'final': final_error})
            
            # Update motion for next frame
            option['motion'] = motion
        
        # Free GPU memory periodically
        if self.device == 'cuda':
            free_gpu_memory()
        
        return RegistrationResult(
            membrane_registered=np.stack(mem_registered, axis=0),
            calcium_registered=np.stack(ca_registered, axis=0),
            motion_fields=np.stack(motion_fields, axis=0) if motion_fields else None,
            errors=errors,
            reference=reference,
        )
    
    def register_single(
        self,
        membrane_frame: np.ndarray,
        calcium_frame: np.ndarray,
        reference: np.ndarray,
        initial_motion: Optional[np.ndarray] = None,
        return_motion: bool = False,
    ) -> RegistrationResult:
        """Register a single frame.
        
        Convenience wrapper around register_batch for single frames.
        """
        mem_batch = membrane_frame[np.newaxis, ...]
        ca_batch = calcium_frame[np.newaxis, ...]
        
        result = self.register_batch(
            mem_batch, ca_batch, reference,
            initial_motion=initial_motion,
            return_motion=return_motion,
        )
        
        return RegistrationResult(
            membrane_registered=result.membrane_registered[0],
            calcium_registered=result.calcium_registered[0],
            motion_fields=result.motion_fields[0] if result.motion_fields is not None else None,
            errors=result.errors,
            reference=reference,
        )

