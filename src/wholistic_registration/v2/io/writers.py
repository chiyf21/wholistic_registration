"""
OME-TIFF writer for registration output.

Writes registered volumes as OME-TIFF files with proper metadata.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
from dataclasses import dataclass

from .metadata import Metadata
from ..utils.logging import get_logger


@dataclass
class OMETiffWriter:
    """Writer for OME-TIFF output files.
    
    Creates a directory structure with one TIFF per frame:
        output_dir/
            membrane/
                frame_000000.ome.tif
                frame_000001.ome.tif
                ...
            calcium/
                frame_000000.ome.tif
                ...
            reference/  (optional)
                ref_000000-000019.ome.tif
                ...
            motion/  (optional)
                motion_000000.h5
                ...
    
    Example:
        >>> writer = OMETiffWriter(output_dir="/results/", metadata=metadata)
        >>> writer.write_frame(membrane_vol, "membrane", frame_idx=0)
        >>> writer.write_frame(calcium_vol, "calcium", frame_idx=0)
    """
    
    output_dir: Union[str, Path]
    metadata: Metadata
    compression: str = 'zlib'
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.logger = get_logger()
        
        # Create subdirectories
        self._dirs = {
            'membrane': self.output_dir / 'membrane',
            'calcium': self.output_dir / 'calcium',
            'reference': self.output_dir / 'reference',
            'motion': self.output_dir / 'motion',
        }
        
        for dir_path in self._dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            f.write(self.metadata.to_json())
        
        self.logger.debug(f"Initialized OME-TIFF writer at {self.output_dir}")
    
    def _get_ome_metadata(self) -> dict:
        """Build OME-TIFF metadata dict with physical sizes and frame rate."""
        return {
            'axes': 'TZCYX',
            'PhysicalSizeX': self.metadata.spacing_x,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': self.metadata.spacing_y,
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeZ': self.metadata.spacing_z,
            'PhysicalSizeZUnit': 'µm',
            'TimeIncrement': 1.0 / self.metadata.frame_rate_hz if self.metadata.frame_rate_hz > 0 else 1.0,
            'TimeIncrementUnit': 's',
        }
    
    def _get_compression(self) -> Optional[str]:
        """Get compression setting for tifffile."""
        compression_map = {
            'none': None,
            'zlib': 'zlib',
            'lzw': 'lzw',
        }
        return compression_map.get(self.compression, 'zlib')
    
    def write_frame(
        self,
        volume: np.ndarray,
        channel_name: str,
        frame_idx: int,
    ) -> Path:
        """Write a single frame volume.
        
        Args:
            volume: 3D array (Z, Y, X) or 2D array (Y, X)
            channel_name: 'membrane' or 'calcium'
            frame_idx: Frame index for filename
            
        Returns:
            Path to written file
        """
        import tifffile
        
        if channel_name not in self._dirs:
            raise ValueError(f"Unknown channel: {channel_name}")
        
        # Ensure 5D shape for OME-TIFF: (T, Z, C, Y, X)
        if volume.ndim == 2:
            # (Y, X) -> (1, 1, 1, Y, X)
            vol_5d = volume[np.newaxis, np.newaxis, np.newaxis, :, :]
        elif volume.ndim == 3:
            # (Z, Y, X) -> (1, Z, 1, Y, X)
            vol_5d = volume[np.newaxis, :, np.newaxis, :, :]
        else:
            raise ValueError(f"Volume must be 2D or 3D, got shape {volume.shape}")
        
        # Build filename
        filename = self._dirs[channel_name] / f"frame_{frame_idx:06d}.ome.tif"
        
        tifffile.imwrite(
            filename,
            vol_5d.astype(np.float32),
            compression=self._get_compression(),
            metadata=self._get_ome_metadata(),
        )
        
        return filename
    
    def write_reference(
        self,
        volume: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> Path:
        """Write a reference volume used for a range of frames.
        
        Args:
            volume: Reference volume (Z, Y, X) or (Y, X)
            start_frame: First frame this reference was used for
            end_frame: Last frame (exclusive) this reference was used for
            
        Returns:
            Path to written file
        """
        import tifffile
        
        # Ensure 5D shape
        if volume.ndim == 2:
            vol_5d = volume[np.newaxis, np.newaxis, np.newaxis, :, :]
        elif volume.ndim == 3:
            vol_5d = volume[np.newaxis, :, np.newaxis, :, :]
        else:
            raise ValueError(f"Volume must be 2D or 3D, got shape {volume.shape}")
        
        filename = self._dirs['reference'] / f"ref_{start_frame:06d}_{end_frame:06d}.ome.tif"
        
        # Reference files also get full OME metadata
        tifffile.imwrite(
            filename,
            vol_5d.astype(np.float32),
            compression=self._get_compression(),
            metadata=self._get_ome_metadata(),
        )
        
        self.logger.debug(f"Wrote reference for frames {start_frame}-{end_frame}")
        return filename
    
    def write_motion(
        self,
        motion_field: np.ndarray,
        frame_idx: int,
    ) -> Path:
        """Write motion field for a frame.
        
        Args:
            motion_field: Motion vectors, shape (Z, Y, X, 3) or (Y, X, 2)
            frame_idx: Frame index
            
        Returns:
            Path to written file
        """
        import h5py
        
        filename = self._dirs['motion'] / f"motion_{frame_idx:06d}.h5"
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset(
                'motion',
                data=motion_field.astype(np.float32),
                compression='gzip',
            )
            f.attrs['frame_idx'] = frame_idx
        
        return filename
    
    def write_batch(
        self,
        membrane_batch: np.ndarray,
        calcium_batch: np.ndarray,
        frame_indices: list,
        reference: Optional[np.ndarray] = None,
        motion_batch: Optional[np.ndarray] = None,
    ) -> None:
        """Write a batch of frames efficiently.
        
        Args:
            membrane_batch: (T, Z, Y, X) membrane volumes
            calcium_batch: (T, Z, Y, X) calcium volumes
            frame_indices: List of frame indices
            reference: Optional reference volume
            motion_batch: Optional (T, Z, Y, X, 3) motion fields
        """
        for i, frame_idx in enumerate(frame_indices):
            self.write_frame(membrane_batch[i], 'membrane', frame_idx)
            self.write_frame(calcium_batch[i], 'calcium', frame_idx)
            
            if motion_batch is not None:
                self.write_motion(motion_batch[i], frame_idx)
        
        if reference is not None:
            self.write_reference(
                reference,
                frame_indices[0],
                frame_indices[-1] + 1,
            )

