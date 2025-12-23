"""
Configuration dataclasses for the registration pipeline.

All configuration is defined using dataclasses with validation.
Supports loading from YAML files and saving to YAML.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Union, Callable, Any
from pathlib import Path
import yaml


@dataclass
class DownsampleConfig:
    """Spatial and temporal downsampling settings.
    
    Attributes:
        xy: Spatial downsampling factor for X/Y dimensions. 
            Factor of 4 reduces (X,Y) → (X/4, Y/4).
        z_slices: List of Z-slice indices to use (0-indexed), or None for all slices.
        t_chunk: Number of frames to process per chunk. Larger = more memory, 
                 but potentially faster.
    """
    xy: int = 1
    z_slices: Optional[List[int]] = None
    t_chunk: int = 20
    
    def __post_init__(self):
        if self.xy < 1:
            raise ValueError(f"xy must be >= 1, got {self.xy}")
        if self.t_chunk < 1:
            raise ValueError(f"t_chunk must be >= 1, got {self.t_chunk}")
        if self.z_slices is not None and len(self.z_slices) == 0:
            raise ValueError("z_slices cannot be empty list, use None for all slices")


@dataclass
class ChannelConfig:
    """Dual-channel registration settings.
    
    When dual_channel is True, registration uses combined signal:
        combined = membrane + k * transform(calcium)
    
    Attributes:
        dual_channel: Whether to use both membrane and calcium channels.
        transform: Transformation applied to calcium channel before combining.
            - 'log10': log₁₀(1 + x) - compresses dynamic range
            - 'sqrt': √x - mild compression
            - 'log2': log₂(1 + x)
            - 'raw': x (no transformation)
        k: Weight coefficient for transformed calcium channel.
           Higher k = more calcium influence on registration.
           Typical values: k=0.3 for 'raw', k=50-300 for 'log10'.
        membrane_channel: Channel index for membrane (default 1).
        calcium_channel: Channel index for calcium (default 0).
    """
    dual_channel: bool = True
    transform: Literal['log10', 'sqrt', 'log2', 'raw'] = 'log10'
    k: float = 50.0
    membrane_channel: int = 1
    calcium_channel: int = 0
    
    def __post_init__(self):
        if self.k < 0:
            raise ValueError(f"k must be >= 0, got {self.k}")
        valid_transforms = {'log10', 'sqrt', 'log2', 'raw'}
        if self.transform not in valid_transforms:
            raise ValueError(f"transform must be one of {valid_transforms}, got {self.transform}")


@dataclass
class ReferenceConfig:
    """Reference image computation settings.
    
    The reference image is computed from a rolling window of recently-registered
    frames using correlation-based selection.
    
    Attributes:
        window_size: Number of frames in the rolling window for reference computation.
                     These frames are used to pick the most stable reference.
        initial_frames: Number of frames from video center used for initial reference.
        max_correlation_frames: Maximum frames to consider when computing correlations.
    """
    window_size: int = 20
    initial_frames: int = 40
    max_correlation_frames: int = 50
    
    def __post_init__(self):
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.initial_frames < self.window_size:
            raise ValueError(
                f"initial_frames ({self.initial_frames}) must be >= window_size ({self.window_size})"
            )


@dataclass 
class PyramidConfig:
    """Multi-scale pyramid registration settings.
    
    The algorithm uses a coarse-to-fine pyramid approach where motion is first
    estimated at low resolution, then refined at higher resolutions.
    
    Attributes:
        layers: Number of pyramid levels. More layers = larger displacement capture,
                but may miss fine details. Typical values: 1-3.
        patch_radius: Radius r of motion estimation patches. Patch size = (2r+1)².
                      Smaller = more control points but noisier.
        iterations: Maximum iterations per pyramid level.
        smooth_penalty: Regularization strength. Higher = smoother motion fields
                        but potentially higher intensity error.
        movement_range: Maximum expected displacement per iteration (pixels).
    """
    layers: int = 1
    patch_radius: int = 5
    iterations: int = 10
    smooth_penalty: float = 0.08
    movement_range: float = 5.0
    
    def __post_init__(self):
        if self.layers < 0:
            raise ValueError(f"layers must be >= 0, got {self.layers}")
        if self.patch_radius < 1:
            raise ValueError(f"patch_radius must be >= 1, got {self.patch_radius}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.smooth_penalty < 0:
            raise ValueError(f"smooth_penalty must be >= 0, got {self.smooth_penalty}")


@dataclass
class MaskConfig:
    """Outlier masking settings.
    
    Pixels are masked based on intensity statistics to exclude outliers
    from motion estimation.
    
    Attributes:
        threshold_factor: Pixels > threshold_factor * std are masked as outliers.
        intensity_range: [min, max] intensity range for valid pixels.
    """
    threshold_factor: float = 5.0
    intensity_range: tuple = (5, 4000)
    
    def __post_init__(self):
        if self.threshold_factor <= 0:
            raise ValueError(f"threshold_factor must be > 0, got {self.threshold_factor}")
        if len(self.intensity_range) != 2:
            raise ValueError(f"intensity_range must have 2 elements, got {len(self.intensity_range)}")
        if self.intensity_range[0] >= self.intensity_range[1]:
            raise ValueError(f"intensity_range[0] must be < intensity_range[1]")


@dataclass
class OutputConfig:
    """Output settings.
    
    Attributes:
        save_reference: Whether to save reference images used for each chunk.
        save_motion: Whether to save motion field vectors.
        compression: OME-TIFF compression method.
    """
    save_reference: bool = True
    save_motion: bool = False
    compression: Literal['none', 'zlib', 'lzw'] = 'zlib'


@dataclass
class BackendConfig:
    """Compute backend settings.
    
    Attributes:
        device: Compute device - 'cuda' for GPU, 'cpu' for CPU-only.
        gpu_id: GPU device ID when using CUDA.
        use_dask: Whether to use Dask for lazy loading and chunked processing.
        dask_chunk_size: Chunk size for Dask arrays (frames per chunk).
    """
    device: Literal['cuda', 'cpu'] = 'cuda'
    gpu_id: int = 0
    use_dask: bool = True
    dask_chunk_size: int = 10
    
    def __post_init__(self):
        if self.device == 'cuda':
            try:
                import cupy as cp
                n_gpus = cp.cuda.runtime.getDeviceCount()
                if self.gpu_id >= n_gpus:
                    raise ValueError(f"gpu_id {self.gpu_id} invalid, only {n_gpus} GPUs available")
            except ImportError:
                raise ImportError("CuPy required for CUDA backend. Install with: pip install cupy-cuda11x")


@dataclass
class RegistrationConfig:
    """Main configuration for the registration pipeline.
    
    Combines all sub-configurations and file paths.
    
    Example:
        >>> config = RegistrationConfig(
        ...     input_path="/data/experiment.nd2",
        ...     output_dir="/results/registered/",
        ...     downsample=DownsampleConfig(xy=4, t_chunk=20),
        ...     channels=ChannelConfig(dual_channel=True, k=50),
        ... )
        >>> config.save_yaml("config.yaml")
        
        >>> # Or load from file:
        >>> config = RegistrationConfig.from_yaml("config.yaml")
    """
    # Required paths
    input_path: str
    output_dir: str
    
    # Sub-configurations (with defaults)
    downsample: DownsampleConfig = field(default_factory=DownsampleConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    backend: BackendConfig = field(default_factory=lambda: BackendConfig.__new__(BackendConfig))
    
    # Metadata (populated from input file)
    _metadata: dict = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        # Convert paths to Path objects for validation
        input_p = Path(self.input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        
        # Create output directory if needed
        output_p = Path(self.output_dir)
        output_p.mkdir(parents=True, exist_ok=True)
        
        # Initialize backend properly if it was created with __new__
        if not hasattr(self.backend, 'device') or self.backend.device is None:
            object.__setattr__(self, 'backend', BackendConfig())
    
    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = self._to_serializable_dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def _to_serializable_dict(self) -> dict:
        """Convert to dictionary, excluding private fields."""
        result = {}
        for key, value in asdict(self).items():
            if key.startswith('_'):
                continue
            if isinstance(value, dict):
                # Convert tuple to list for YAML serialization
                value = {k: list(v) if isinstance(v, tuple) else v for k, v in value.items()}
            result[key] = value
        return result
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'RegistrationConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses
        if 'downsample' in data and isinstance(data['downsample'], dict):
            data['downsample'] = DownsampleConfig(**data['downsample'])
        if 'channels' in data and isinstance(data['channels'], dict):
            data['channels'] = ChannelConfig(**data['channels'])
        if 'reference' in data and isinstance(data['reference'], dict):
            data['reference'] = ReferenceConfig(**data['reference'])
        if 'pyramid' in data and isinstance(data['pyramid'], dict):
            data['pyramid'] = PyramidConfig(**data['pyramid'])
        if 'mask' in data and isinstance(data['mask'], dict):
            # Convert list back to tuple for intensity_range
            if 'intensity_range' in data['mask']:
                data['mask']['intensity_range'] = tuple(data['mask']['intensity_range'])
            data['mask'] = MaskConfig(**data['mask'])
        if 'output' in data and isinstance(data['output'], dict):
            data['output'] = OutputConfig(**data['output'])
        if 'backend' in data and isinstance(data['backend'], dict):
            data['backend'] = BackendConfig(**data['backend'])
        
        return cls(**data)
    
    @property
    def input_format(self) -> str:
        """Detect input format from file extension."""
        p = Path(self.input_path)
        if p.suffix.lower() == '.nd2':
            return 'nd2'
        elif p.suffix.lower() == '.zarr' or p.is_dir() and (p / '.zarray').exists():
            return 'zarr'
        elif p.suffix.lower() in ('.tif', '.tiff'):
            return 'tiff'
        elif p.is_dir():
            # Check if directory contains TIFF files
            tiffs = list(p.glob('*.tif')) + list(p.glob('*.tiff'))
            if tiffs:
                return 'tiff_series'
            # Check for zarr
            if (p / '.zarray').exists() or (p / '.zgroup').exists():
                return 'zarr'
        raise ValueError(f"Cannot determine input format for: {self.input_path}")

