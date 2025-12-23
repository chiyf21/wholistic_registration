"""Tests for I/O module."""

import pytest
import numpy as np
from pathlib import Path

from ..io import create_reader, OMETiffWriter, Metadata
from ..io.readers import TiffSeriesReader, ZarrReader


class TestMetadata:
    """Tests for Metadata class."""
    
    def test_basic_properties(self):
        """Basic properties should work."""
        meta = Metadata(
            n_frames=100,
            n_channels=2,
            shape_zyx=(10, 512, 512),
            voxel_size_um=(2.0, 0.5, 0.5),
            frame_rate_hz=10.0,
        )
        
        assert meta.n_z == 10
        assert meta.n_y == 512
        assert meta.n_x == 512
        assert meta.is_3d == True
        assert meta.spacing_z == 2.0
        assert meta.spacing_x == 0.5
        assert meta.z_ratio == 4.0  # 2.0 / 0.5
    
    def test_2d_detection(self):
        """Should detect 2D data (Z=1)."""
        meta = Metadata(
            n_frames=100,
            n_channels=1,
            shape_zyx=(1, 512, 512),
        )
        assert meta.is_3d == False
    
    def test_serialization(self):
        """Should serialize to dict and JSON."""
        meta = Metadata(
            n_frames=50,
            n_channels=2,
            shape_zyx=(5, 128, 128),
        )
        
        d = meta.to_dict()
        assert d['n_frames'] == 50
        assert d['shape_zyx'] == (5, 128, 128)
        
        json_str = meta.to_json()
        assert '"n_frames": 50' in json_str
    
    def test_from_dict(self):
        """Should deserialize from dict."""
        d = {
            'n_frames': 50,
            'n_channels': 2,
            'shape_zyx': [5, 128, 128],  # List, not tuple
            'voxel_size_um': [1.0, 0.5, 0.5],
        }
        meta = Metadata.from_dict(d)
        
        assert meta.n_frames == 50
        assert meta.shape_zyx == (5, 128, 128)  # Should be tuple


class TestTiffSeriesReader:
    """Tests for TIFF series reader."""
    
    def test_reads_synthetic_data(self, synthetic_tiff_data):
        """Should read synthetic TIFF data."""
        reader = TiffSeriesReader(synthetic_tiff_data / "membrane")
        
        assert reader.metadata.n_frames == 10
        assert reader.metadata.source_format == 'tiff_series'
    
    def test_read_frames(self, synthetic_tiff_data):
        """Should read specific frames."""
        reader = TiffSeriesReader(synthetic_tiff_data / "membrane")
        
        frames = reader.read_frames([0, 1, 2], channel=0)
        
        assert frames.shape[0] == 3
        assert frames.ndim == 4  # (T, Z, Y, X)
    
    def test_xy_downsample(self, synthetic_tiff_data):
        """Should downsample XY."""
        reader = TiffSeriesReader(synthetic_tiff_data / "membrane")
        
        full_res = reader.read_frames([0], channel=0)
        downsampled = reader.read_frames([0], channel=0, xy_downsample=2)
        
        assert downsampled.shape[-1] == full_res.shape[-1] // 2
        assert downsampled.shape[-2] == full_res.shape[-2] // 2


class TestZarrReader:
    """Tests for Zarr reader."""
    
    def test_reads_synthetic_zarr(self, synthetic_zarr_data):
        """Should read synthetic Zarr data."""
        reader = ZarrReader(synthetic_zarr_data)
        
        assert reader.metadata.n_frames == 10
        assert reader.metadata.source_format == 'zarr'
    
    def test_read_frames(self, synthetic_zarr_data):
        """Should read frames from Zarr."""
        reader = ZarrReader(synthetic_zarr_data)
        
        frames = reader.read_frames([0, 5, 9], channel=0)
        
        assert frames.shape[0] == 3


class TestCreateReader:
    """Tests for factory function."""
    
    def test_creates_zarr_reader(self, synthetic_zarr_data):
        """Should create ZarrReader for .zarr path."""
        reader = create_reader(synthetic_zarr_data)
        
        assert isinstance(reader, ZarrReader)
    
    def test_creates_tiff_series_reader(self, synthetic_tiff_data):
        """Should create TiffSeriesReader for TIFF directory."""
        reader = create_reader(synthetic_tiff_data / "membrane")
        
        assert isinstance(reader, TiffSeriesReader)


class TestOMETiffWriter:
    """Tests for OME-TIFF writer."""
    
    @pytest.fixture
    def writer(self, temp_dir):
        """Create a writer for testing."""
        meta = Metadata(
            n_frames=10,
            n_channels=2,
            shape_zyx=(5, 32, 32),
            voxel_size_um=(2.0, 0.5, 0.5),
            frame_rate_hz=10.0,
        )
        return OMETiffWriter(
            output_dir=temp_dir / "output",
            metadata=meta,
        )
    
    def test_creates_directories(self, writer, temp_dir):
        """Should create output subdirectories."""
        assert (temp_dir / "output" / "membrane").exists()
        assert (temp_dir / "output" / "calcium").exists()
    
    def test_writes_frame(self, writer, temp_dir):
        """Should write a frame as OME-TIFF."""
        volume = np.random.rand(5, 32, 32).astype(np.float32)
        
        path = writer.write_frame(volume, 'membrane', frame_idx=0)
        
        assert path.exists()
        assert path.name == "frame_000000.ome.tif"
    
    def test_writes_reference(self, writer, temp_dir):
        """Should write reference volume."""
        volume = np.random.rand(5, 32, 32).astype(np.float32)
        
        path = writer.write_reference(volume, start_frame=0, end_frame=10)
        
        assert path.exists()
        assert "ref_000000_000010" in path.name
    
    def test_writes_batch(self, writer, temp_dir):
        """Should write batch of frames."""
        membrane = np.random.rand(3, 5, 32, 32).astype(np.float32)
        calcium = np.random.rand(3, 5, 32, 32).astype(np.float32)
        
        writer.write_batch(
            membrane, calcium,
            frame_indices=[10, 11, 12],
        )
        
        # Check files exist
        for i in [10, 11, 12]:
            assert (temp_dir / "output" / "membrane" / f"frame_{i:06d}.ome.tif").exists()
            assert (temp_dir / "output" / "calcium" / f"frame_{i:06d}.ome.tif").exists()
    
    def test_ome_metadata_contains_physical_sizes(self, writer, temp_dir):
        """OME-TIFF should contain correct physical size metadata."""
        import tifffile
        import xml.etree.ElementTree as ET
        
        volume = np.random.rand(5, 32, 32).astype(np.float32)
        path = writer.write_frame(volume, 'membrane', frame_idx=0)
        
        with tifffile.TiffFile(str(path)) as tif:
            assert tif.ome_metadata is not None, "Missing OME metadata"
            
            ome = ET.fromstring(tif.ome_metadata)
            pixels = None
            for elem in ome.iter():
                if 'Pixels' in elem.tag:
                    pixels = elem
                    break
            
            assert pixels is not None, "Missing Pixels element in OME-XML"
            
            # Check physical sizes match our metadata
            assert float(pixels.get('PhysicalSizeX')) == 0.5
            assert float(pixels.get('PhysicalSizeY')) == 0.5
            assert float(pixels.get('PhysicalSizeZ')) == 2.0
            assert pixels.get('PhysicalSizeXUnit') == 'µm'
    
    def test_ome_metadata_contains_time_increment(self, writer, temp_dir):
        """OME-TIFF should contain time increment from frame rate."""
        import tifffile
        import xml.etree.ElementTree as ET
        
        volume = np.random.rand(5, 32, 32).astype(np.float32)
        path = writer.write_frame(volume, 'membrane', frame_idx=0)
        
        with tifffile.TiffFile(str(path)) as tif:
            ome = ET.fromstring(tif.ome_metadata)
            for elem in ome.iter():
                if 'Pixels' in elem.tag:
                    # frame_rate=10 Hz -> TimeIncrement=0.1s
                    assert float(elem.get('TimeIncrement')) == pytest.approx(0.1)
                    break
    
    def test_reference_has_ome_metadata(self, writer, temp_dir):
        """Reference files should also have OME metadata."""
        import tifffile
        
        volume = np.random.rand(5, 32, 32).astype(np.float32)
        path = writer.write_reference(volume, start_frame=0, end_frame=10)
        
        with tifffile.TiffFile(str(path)) as tif:
            assert tif.ome_metadata is not None, "Reference file missing OME metadata"
    
    def test_saves_metadata_json(self, writer, temp_dir):
        """Should save metadata.json with full dataset info."""
        import json
        
        metadata_path = temp_dir / "output" / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            data = json.load(f)
        
        assert data['n_frames'] == 10
        assert data['voxel_size_um'] == [2.0, 0.5, 0.5]
        assert data['frame_rate_hz'] == 10.0

