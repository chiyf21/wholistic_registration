"""
Data readers for various microscopy formats.

Supports:
- ND2 (Nikon)
- Zarr (chunked arrays)
- TIFF (single multi-page or series)

All readers provide a consistent interface via BaseReader.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Union, Tuple
import numpy as np

from .metadata import Metadata
from ..utils.logging import get_logger

# Lazy imports for optional dependencies
_nd2 = None
_zarr = None
_tifffile = None
_dask = None
_dask_array = None


def _import_nd2():
    global _nd2
    if _nd2 is None:
        import nd2
        _nd2 = nd2
    return _nd2


def _import_zarr():
    global _zarr
    if _zarr is None:
        import zarr
        _zarr = zarr
    return _zarr


def _import_tifffile():
    global _tifffile
    if _tifffile is None:
        import tifffile
        _tifffile = tifffile
    return _tifffile


def _import_dask():
    global _dask, _dask_array
    if _dask is None:
        import dask
        import dask.array as da
        _dask = dask
        _dask_array = da
    return _dask, _dask_array


class BaseReader(ABC):
    """Abstract base class for data readers.
    
    All readers must implement:
        - metadata property
        - read_frames method
        - read_frame method
    """
    
    def __init__(self, path: Union[str, Path]):
        """Initialize reader.
        
        Args:
            path: Path to data file or directory
        """
        self.path = Path(path)
        self._metadata: Optional[Metadata] = None
        self.logger = get_logger()
    
    @property
    @abstractmethod
    def metadata(self) -> Metadata:
        """Get dataset metadata."""
        pass
    
    @abstractmethod
    def read_frames(
        self,
        frames: List[int],
        channel: int,
        z_slices: Optional[List[int]] = None,
        xy_downsample: int = 1,
    ) -> np.ndarray:
        """Read multiple frames.
        
        Args:
            frames: List of frame indices to read
            channel: Channel index
            z_slices: List of Z-slice indices (None = all)
            xy_downsample: Spatial downsampling factor
            
        Returns:
            Array of shape (T, Z, Y, X) for 3D or (T, Y, X) for 2D
        """
        pass
    
    def read_frame(
        self,
        frame: int,
        channel: int,
        z_slices: Optional[List[int]] = None,
        xy_downsample: int = 1,
    ) -> np.ndarray:
        """Read a single frame.
        
        Args:
            frame: Frame index
            channel: Channel index
            z_slices: List of Z-slice indices (None = all)
            xy_downsample: Spatial downsampling factor
            
        Returns:
            Array of shape (Z, Y, X) for 3D or (Y, X) for 2D
        """
        result = self.read_frames([frame], channel, z_slices, xy_downsample)
        return result[0]
    
    def close(self):
        """Close any open file handles."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ND2Reader(BaseReader):
    """Reader for Nikon ND2 files."""
    
    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        nd2 = _import_nd2()
        self._file = nd2.ND2File(str(self.path))
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Extract metadata from ND2 file."""
        f = self._file
        sizes = f.sizes
        
        # Get dimensions
        n_frames = sizes.get('T', 1)
        n_channels = sizes.get('C', 1)
        n_z = sizes.get('Z', 1)
        
        # Get first frame to determine XY size
        shape = f.shape
        # ND2 shape order is typically (T, C, Z, Y, X) but can vary
        # Use sizes dict which is more reliable
        n_y = sizes.get('Y', shape[-2] if len(shape) >= 2 else 1)
        n_x = sizes.get('X', shape[-1] if len(shape) >= 1 else 1)
        
        # Get voxel sizes
        try:
            voxel_size = f.voxel_size()
            voxel_um = (
                voxel_size.z if hasattr(voxel_size, 'z') else 1.0,
                voxel_size.y if hasattr(voxel_size, 'y') else 1.0,
                voxel_size.x if hasattr(voxel_size, 'x') else 1.0,
            )
        except:
            voxel_um = (1.0, 1.0, 1.0)
        
        # Get frame rate
        try:
            # Try to get from experiment metadata
            frame_rate = 1.0  # Default
            if hasattr(f, 'experiment') and f.experiment:
                for loop in f.experiment:
                    if hasattr(loop, 'parameters') and loop.parameters:
                        if hasattr(loop.parameters, 'periodMs'):
                            frame_rate = 1000.0 / loop.parameters.periodMs
                            break
        except:
            frame_rate = 1.0
        
        # Channel names
        try:
            channel_names = [ch.channel.name for ch in f.metadata.channels] if f.metadata.channels else None
        except:
            channel_names = None
        
        self._metadata = Metadata(
            n_frames=n_frames,
            n_channels=n_channels,
            shape_zyx=(n_z, n_y, n_x),
            voxel_size_um=voxel_um,
            frame_rate_hz=frame_rate,
            channel_names=channel_names,
            source_format='nd2',
            source_path=str(self.path),
        )
        
        self.logger.debug(f"Loaded ND2 metadata: {self._metadata}")
    
    @property
    def metadata(self) -> Metadata:
        return self._metadata
    
    def read_frames(
        self,
        frames: List[int],
        channel: int,
        z_slices: Optional[List[int]] = None,
        xy_downsample: int = 1,
    ) -> np.ndarray:
        """Read frames from ND2 file."""
        from skimage.transform import resize
        
        f = self._file
        result = []
        
        for frame_idx in frames:
            # Read the frame
            # ND2 indexing depends on the file structure
            try:
                if self.metadata.n_channels > 1:
                    if self.metadata.is_3d:
                        # Shape: (Z, C, Y, X) or (C, Z, Y, X) - varies
                        data = np.asarray(f.read_frame(frame_idx))
                        # Handle different axis orders
                        if data.ndim == 4:
                            # Find channel axis
                            if data.shape[0] == self.metadata.n_channels:
                                # (C, Z, Y, X)
                                data = data[channel]  # Now (Z, Y, X)
                            elif data.shape[1] == self.metadata.n_channels:
                                # (Z, C, Y, X)
                                data = data[:, channel]  # Now (Z, Y, X)
                            else:
                                # Assume first axis is channel if small
                                data = data[channel]
                        elif data.ndim == 3:
                            # (Z, Y, X) - channel already selected or single channel
                            pass
                    else:
                        data = np.asarray(f.read_frame(frame_idx))
                        if data.ndim == 3:
                            data = data[channel]
                else:
                    data = np.asarray(f.read_frame(frame_idx))
            except:
                # Fallback: try direct array indexing
                arr = f.asarray()
                if arr.ndim == 5:  # (T, C, Z, Y, X)
                    data = arr[frame_idx, channel]
                elif arr.ndim == 4:  # (T, Z, Y, X) or (T, C, Y, X)
                    if arr.shape[1] == self.metadata.n_z:
                        data = arr[frame_idx]
                    else:
                        data = arr[frame_idx, channel]
                elif arr.ndim == 3:  # (T, Y, X)
                    data = arr[frame_idx]
                else:
                    raise ValueError(f"Unexpected array shape: {arr.shape}")
            
            # Select Z slices
            if z_slices is not None and data.ndim == 3:
                data = data[z_slices]
            
            # Downsample XY
            if xy_downsample > 1:
                if data.ndim == 3:
                    # 3D: (Z, Y, X)
                    new_shape = (
                        data.shape[0],
                        data.shape[1] // xy_downsample,
                        data.shape[2] // xy_downsample,
                    )
                else:
                    # 2D: (Y, X)
                    new_shape = (
                        data.shape[0] // xy_downsample,
                        data.shape[1] // xy_downsample,
                    )
                data = resize(
                    data, new_shape, 
                    order=1, 
                    anti_aliasing=True, 
                    preserve_range=True
                ).astype(data.dtype)
            
            result.append(data)
        
        return np.stack(result, axis=0)
    
    def close(self):
        if hasattr(self, '_file') and self._file:
            self._file.close()


class ZarrReader(BaseReader):
    """Reader for Zarr arrays."""
    
    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        zarr = _import_zarr()
        self._root = zarr.open(str(self.path), mode='r')
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Extract metadata from Zarr."""
        root = self._root
        
        # Try to find the main data array
        if isinstance(root, _zarr.hierarchy.Group):
            # Look for common array names
            for name in ['data', 'membrane', 'calcium', '0']:
                if name in root:
                    data_arr = root[name]
                    break
            else:
                # Use first array found
                arrays = list(root.arrays())
                if arrays:
                    data_arr = arrays[0][1]
                else:
                    raise ValueError("No arrays found in Zarr group")
        else:
            data_arr = root
        
        shape = data_arr.shape
        
        # Infer dimensions from shape
        # Common orders: (T, Z, Y, X), (T, C, Z, Y, X), (T, Z, C, Y, X)
        if len(shape) == 5:
            n_frames, n_channels, n_z, n_y, n_x = shape
        elif len(shape) == 4:
            n_frames, n_z, n_y, n_x = shape
            n_channels = 1
        elif len(shape) == 3:
            n_frames, n_y, n_x = shape
            n_z = 1
            n_channels = 1
        else:
            raise ValueError(f"Unexpected Zarr shape: {shape}")
        
        # Try to get metadata from attrs
        attrs = dict(root.attrs) if hasattr(root, 'attrs') else {}
        meta = attrs.get('Metadata', attrs.get('metadata', {}))
        
        voxel_size = meta.get('voxelsize', [1.0, 1.0, 1.0])
        if len(voxel_size) == 3:
            voxel_um = tuple(voxel_size)
        else:
            voxel_um = (1.0, 1.0, 1.0)
        
        frame_rate = meta.get('fps', meta.get('frame_rate', 1.0))
        
        self._metadata = Metadata(
            n_frames=n_frames,
            n_channels=n_channels,
            shape_zyx=(n_z, n_y, n_x),
            voxel_size_um=voxel_um,
            frame_rate_hz=frame_rate,
            source_format='zarr',
            source_path=str(self.path),
            extra=meta,
        )
        
        self._data_arr = data_arr
        self.logger.debug(f"Loaded Zarr metadata: {self._metadata}")
    
    @property
    def metadata(self) -> Metadata:
        return self._metadata
    
    def read_frames(
        self,
        frames: List[int],
        channel: int,
        z_slices: Optional[List[int]] = None,
        xy_downsample: int = 1,
    ) -> np.ndarray:
        """Read frames from Zarr array."""
        from skimage.transform import resize
        
        result = []
        for frame_idx in frames:
            # Read frame
            data = np.asarray(self._data_arr[frame_idx])
            
            # Handle multi-channel
            if data.ndim == 4 and data.shape[0] == self.metadata.n_channels:
                data = data[channel]
            
            # Select Z slices
            if z_slices is not None and data.ndim == 3:
                data = data[z_slices]
            
            # Downsample
            if xy_downsample > 1:
                if data.ndim == 3:
                    new_shape = (
                        data.shape[0],
                        data.shape[1] // xy_downsample,
                        data.shape[2] // xy_downsample,
                    )
                else:
                    new_shape = (
                        data.shape[0] // xy_downsample,
                        data.shape[1] // xy_downsample,
                    )
                data = resize(
                    data, new_shape, 
                    order=1, 
                    anti_aliasing=True, 
                    preserve_range=True
                ).astype(np.float32)
            
            result.append(data)
        
        return np.stack(result, axis=0)


class TiffReader(BaseReader):
    """Reader for single TIFF files containing multiple volumes."""
    
    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        tifffile = _import_tifffile()
        self._tiff = tifffile.TiffFile(str(self.path))
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Extract metadata from TIFF."""
        tiff = self._tiff
        
        # Get shape from series
        if tiff.series:
            series = tiff.series[0]
            shape = series.shape
            axes = series.axes if hasattr(series, 'axes') else ''
        else:
            shape = tiff.pages[0].shape
            axes = ''
        
        # Parse axes or infer from shape
        if 'T' in axes:
            t_idx = axes.index('T')
            n_frames = shape[t_idx]
        else:
            # Assume first dimension is T if shape is 4D or 5D
            n_frames = shape[0] if len(shape) >= 4 else len(tiff.pages)
        
        if 'C' in axes:
            c_idx = axes.index('C')
            n_channels = shape[c_idx]
        else:
            n_channels = 1
        
        if 'Z' in axes:
            z_idx = axes.index('Z')
            n_z = shape[z_idx]
        else:
            # Infer Z from shape
            if len(shape) >= 4:
                n_z = shape[1] if n_channels == 1 else shape[2]
            else:
                n_z = 1
        
        # Y, X are always last two
        n_y, n_x = shape[-2], shape[-1]
        
        # Get voxel size from metadata
        voxel_um = (1.0, 1.0, 1.0)
        if tiff.pages[0].tags:
            try:
                # Try OME-TIFF metadata
                if tiff.ome_metadata:
                    import xml.etree.ElementTree as ET
                    ome = ET.fromstring(tiff.ome_metadata)
                    # Parse physical sizes from OME-XML
                    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    pixels = ome.find('.//ome:Pixels', ns)
                    if pixels is not None:
                        size_x = float(pixels.get('PhysicalSizeX', 1.0))
                        size_y = float(pixels.get('PhysicalSizeY', 1.0))
                        size_z = float(pixels.get('PhysicalSizeZ', 1.0))
                        voxel_um = (size_z, size_y, size_x)
            except:
                pass
        
        self._metadata = Metadata(
            n_frames=n_frames,
            n_channels=n_channels,
            shape_zyx=(n_z, n_y, n_x),
            voxel_size_um=voxel_um,
            source_format='tiff',
            source_path=str(self.path),
        )
        
        self._shape = shape
        self._axes = axes
        self.logger.debug(f"Loaded TIFF metadata: {self._metadata}")
    
    @property
    def metadata(self) -> Metadata:
        return self._metadata
    
    def read_frames(
        self,
        frames: List[int],
        channel: int,
        z_slices: Optional[List[int]] = None,
        xy_downsample: int = 1,
    ) -> np.ndarray:
        """Read frames from TIFF file."""
        from skimage.transform import resize
        
        # Read all data (may need optimization for very large files)
        data = self._tiff.asarray()
        
        result = []
        for frame_idx in frames:
            # Extract frame
            if data.ndim == 5:  # (T, C, Z, Y, X)
                frame = data[frame_idx, channel]
            elif data.ndim == 4:
                if self.metadata.n_channels > 1:
                    frame = data[frame_idx, channel]
                else:
                    frame = data[frame_idx]
            elif data.ndim == 3:
                frame = data[frame_idx]
            else:
                frame = data
            
            # Select Z slices
            if z_slices is not None and frame.ndim == 3:
                frame = frame[z_slices]
            
            # Downsample
            if xy_downsample > 1:
                if frame.ndim == 3:
                    new_shape = (
                        frame.shape[0],
                        frame.shape[1] // xy_downsample,
                        frame.shape[2] // xy_downsample,
                    )
                else:
                    new_shape = (
                        frame.shape[0] // xy_downsample,
                        frame.shape[1] // xy_downsample,
                    )
                frame = resize(
                    frame, new_shape,
                    order=1,
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(np.float32)
            
            result.append(frame)
        
        return np.stack(result, axis=0)
    
    def close(self):
        if hasattr(self, '_tiff') and self._tiff:
            self._tiff.close()


class TiffSeriesReader(BaseReader):
    """Reader for a series of TIFF files (one per timepoint)."""
    
    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        self._tiff_files: List[Path] = []
        self._find_files()
        self._load_metadata()
    
    def _find_files(self) -> None:
        """Find all TIFF files in directory."""
        p = self.path
        if p.is_file():
            # Single file - treat as directory containing it
            p = p.parent
        
        # Find all TIFF files
        patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        files = []
        for pattern in patterns:
            files.extend(p.glob(pattern))
        
        # Sort by name (assumes numeric ordering)
        self._tiff_files = sorted(set(files))
        
        if not self._tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {p}")
        
        self.logger.info(f"Found {len(self._tiff_files)} TIFF files")
    
    def _load_metadata(self) -> None:
        """Extract metadata from first TIFF file."""
        tifffile = _import_tifffile()
        
        # Read first file to get dimensions
        with tifffile.TiffFile(str(self._tiff_files[0])) as tiff:
            page = tiff.pages[0]
            shape = page.shape
            
            # Each file is one timepoint
            # Shape is typically (Z, Y, X) or (Z, C, Y, X) or (Y, X)
            if len(shape) == 4:
                n_z, n_channels, n_y, n_x = shape
            elif len(shape) == 3:
                n_z, n_y, n_x = shape
                n_channels = 1
            else:
                n_y, n_x = shape
                n_z = 1
                n_channels = 1
        
        self._metadata = Metadata(
            n_frames=len(self._tiff_files),
            n_channels=n_channels,
            shape_zyx=(n_z, n_y, n_x),
            source_format='tiff_series',
            source_path=str(self.path),
        )
        
        self.logger.debug(f"Loaded TIFF series metadata: {self._metadata}")
    
    @property
    def metadata(self) -> Metadata:
        return self._metadata
    
    def read_frames(
        self,
        frames: List[int],
        channel: int,
        z_slices: Optional[List[int]] = None,
        xy_downsample: int = 1,
    ) -> np.ndarray:
        """Read frames from TIFF series."""
        tifffile = _import_tifffile()
        from skimage.transform import resize
        
        result = []
        for frame_idx in frames:
            # Read file
            data = tifffile.imread(str(self._tiff_files[frame_idx]))
            
            # Handle channels
            if data.ndim == 4:  # (Z, C, Y, X)
                data = data[:, channel]
            
            # Select Z slices
            if z_slices is not None and data.ndim == 3:
                data = data[z_slices]
            
            # Downsample
            if xy_downsample > 1:
                if data.ndim == 3:
                    new_shape = (
                        data.shape[0],
                        data.shape[1] // xy_downsample,
                        data.shape[2] // xy_downsample,
                    )
                else:
                    new_shape = (
                        data.shape[0] // xy_downsample,
                        data.shape[1] // xy_downsample,
                    )
                data = resize(
                    data, new_shape,
                    order=1,
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(np.float32)
            
            result.append(data)
        
        return np.stack(result, axis=0)


def create_reader(path: Union[str, Path]) -> BaseReader:
    """Factory function to create appropriate reader based on file type.
    
    Args:
        path: Path to data file or directory
        
    Returns:
        Appropriate reader instance
        
    Example:
        >>> reader = create_reader("/data/experiment.nd2")
        >>> print(reader.metadata)
        >>> frames = reader.read_frames([0, 1, 2], channel=0)
    """
    p = Path(path)
    logger = get_logger()
    
    if p.suffix.lower() == '.nd2':
        logger.info(f"Creating ND2 reader for {p}")
        return ND2Reader(p)
    
    elif p.suffix.lower() == '.zarr' or (p.is_dir() and (p / '.zarray').exists()):
        logger.info(f"Creating Zarr reader for {p}")
        return ZarrReader(p)
    
    elif p.suffix.lower() in ('.tif', '.tiff'):
        logger.info(f"Creating TIFF reader for {p}")
        return TiffReader(p)
    
    elif p.is_dir():
        # Check for zarr first
        if (p / '.zgroup').exists() or (p / '.zarray').exists():
            logger.info(f"Creating Zarr reader for directory {p}")
            return ZarrReader(p)
        
        # Check for TIFF series
        tiffs = list(p.glob('*.tif')) + list(p.glob('*.tiff'))
        if tiffs:
            logger.info(f"Creating TIFF series reader for {p}")
            return TiffSeriesReader(p)
        
        raise ValueError(f"Cannot determine reader type for directory: {p}")
    
    else:
        raise ValueError(f"Unsupported file format: {p}")

