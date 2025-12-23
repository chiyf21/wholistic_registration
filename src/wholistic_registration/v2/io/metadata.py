"""
Unified metadata handling for microscopy data.

Provides a consistent interface for metadata regardless of input format.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Dict, Any
import json


@dataclass
class Metadata:
    """Unified metadata container for microscopy volumes.
    
    Attributes:
        n_frames: Total number of time points
        n_channels: Number of channels
        shape_zyx: Spatial dimensions as (Z, Y, X)
        voxel_size_um: Voxel size in micrometers as (z, y, x)
        frame_rate_hz: Acquisition frame rate in Hz
        channel_names: Optional list of channel names
        source_format: Original file format ('nd2', 'zarr', 'tiff', etc.)
        source_path: Path to original file
        extra: Additional format-specific metadata
    """
    n_frames: int
    n_channels: int
    shape_zyx: Tuple[int, int, int]
    voxel_size_um: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    frame_rate_hz: float = 1.0
    channel_names: Optional[List[str]] = None
    source_format: str = "unknown"
    source_path: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_z(self) -> int:
        """Number of Z slices."""
        return self.shape_zyx[0]
    
    @property
    def n_y(self) -> int:
        """Height in pixels."""
        return self.shape_zyx[1]
    
    @property
    def n_x(self) -> int:
        """Width in pixels."""
        return self.shape_zyx[2]
    
    @property
    def is_3d(self) -> bool:
        """Whether data is 3D (Z > 1)."""
        return self.n_z > 1
    
    @property
    def spacing_x(self) -> float:
        """X pixel size in micrometers."""
        return self.voxel_size_um[2]
    
    @property
    def spacing_y(self) -> float:
        """Y pixel size in micrometers."""
        return self.voxel_size_um[1]
    
    @property
    def spacing_z(self) -> float:
        """Z spacing in micrometers."""
        return self.voxel_size_um[0]
    
    @property
    def z_ratio(self) -> float:
        """Ratio of Z spacing to XY spacing."""
        xy_avg = (self.spacing_x + self.spacing_y) / 2
        if xy_avg == 0:
            return 1.0
        return self.spacing_z / xy_avg
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Metadata':
        """Create from dictionary."""
        # Convert lists back to tuples
        if 'shape_zyx' in data:
            data['shape_zyx'] = tuple(data['shape_zyx'])
        if 'voxel_size_um' in data:
            data['voxel_size_um'] = tuple(data['voxel_size_um'])
        return cls(**data)
    
    def __repr__(self) -> str:
        return (
            f"Metadata(\n"
            f"  frames={self.n_frames}, channels={self.n_channels}\n"
            f"  shape_zyx={self.shape_zyx}\n"
            f"  voxel_size_um={self.voxel_size_um}\n"
            f"  frame_rate={self.frame_rate_hz:.2f} Hz\n"
            f"  format={self.source_format}\n"
            f")"
        )

