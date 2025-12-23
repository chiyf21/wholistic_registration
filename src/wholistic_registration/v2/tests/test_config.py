"""Tests for configuration module."""

import pytest
import numpy as np
from pathlib import Path

from ..config import (
    RegistrationConfig,
    DownsampleConfig,
    ChannelConfig,
    PyramidConfig,
    ReferenceConfig,
    MaskConfig,
    OutputConfig,
)


class TestDownsampleConfig:
    """Tests for DownsampleConfig."""
    
    def test_default_values(self):
        """Default values should be reasonable."""
        cfg = DownsampleConfig()
        assert cfg.xy == 1
        assert cfg.z_slices is None
        assert cfg.t_chunk == 20
    
    def test_validation_xy(self):
        """Should validate xy >= 1."""
        with pytest.raises(ValueError, match="xy must be >= 1"):
            DownsampleConfig(xy=0)
    
    def test_validation_t_chunk(self):
        """Should validate t_chunk >= 1."""
        with pytest.raises(ValueError, match="t_chunk must be >= 1"):
            DownsampleConfig(t_chunk=0)
    
    def test_validation_z_slices(self):
        """Should validate z_slices not empty."""
        with pytest.raises(ValueError, match="z_slices cannot be empty"):
            DownsampleConfig(z_slices=[])


class TestChannelConfig:
    """Tests for ChannelConfig."""
    
    def test_default_values(self):
        """Default values should be reasonable."""
        cfg = ChannelConfig()
        assert cfg.dual_channel == True
        assert cfg.transform == 'log10'
        assert cfg.k == 50.0
    
    def test_validation_k(self):
        """Should validate k >= 0."""
        with pytest.raises(ValueError, match="k must be >= 0"):
            ChannelConfig(k=-1)
    
    def test_validation_transform(self):
        """Should validate transform is valid."""
        with pytest.raises(ValueError, match="transform must be one of"):
            ChannelConfig(transform='invalid')
    
    def test_valid_transforms(self):
        """All valid transforms should work."""
        for t in ['log10', 'sqrt', 'log2', 'raw']:
            cfg = ChannelConfig(transform=t)
            assert cfg.transform == t


class TestPyramidConfig:
    """Tests for PyramidConfig."""
    
    def test_default_values(self):
        """Default values should be reasonable."""
        cfg = PyramidConfig()
        assert cfg.layers == 1
        assert cfg.patch_radius == 5
        assert cfg.iterations == 10
    
    def test_validation_layers(self):
        """Should validate layers >= 0."""
        with pytest.raises(ValueError, match="layers must be >= 0"):
            PyramidConfig(layers=-1)
    
    def test_validation_patch_radius(self):
        """Should validate patch_radius >= 1."""
        with pytest.raises(ValueError, match="patch_radius must be >= 1"):
            PyramidConfig(patch_radius=0)


class TestReferenceConfig:
    """Tests for ReferenceConfig."""
    
    def test_default_values(self):
        """Default values should be reasonable."""
        cfg = ReferenceConfig()
        assert cfg.window_size == 20
        assert cfg.initial_frames == 40
    
    def test_validation_initial_vs_window(self):
        """Should validate initial_frames >= window_size."""
        with pytest.raises(ValueError, match="initial_frames.*must be >= window_size"):
            ReferenceConfig(window_size=50, initial_frames=30)


class TestMaskConfig:
    """Tests for MaskConfig."""
    
    def test_default_values(self):
        """Default values should be reasonable."""
        cfg = MaskConfig()
        assert cfg.threshold_factor == 5.0
        assert cfg.intensity_range == (5, 4000)
    
    def test_validation_threshold(self):
        """Should validate threshold > 0."""
        with pytest.raises(ValueError, match="threshold_factor must be > 0"):
            MaskConfig(threshold_factor=0)
    
    def test_validation_range(self):
        """Should validate intensity range."""
        with pytest.raises(ValueError, match="intensity_range"):
            MaskConfig(intensity_range=(100, 50))  # min > max


class TestRegistrationConfig:
    """Tests for main RegistrationConfig."""
    
    def test_from_yaml_and_save(self, temp_dir, synthetic_zarr_data):
        """Should save and load from YAML."""
        # Create config
        config = RegistrationConfig(
            input_path=str(synthetic_zarr_data),
            output_dir=str(temp_dir / "output"),
            downsample=DownsampleConfig(xy=2, t_chunk=10),
            channels=ChannelConfig(k=100.0),
        )
        
        # Save to YAML
        yaml_path = temp_dir / "test_config.yaml"
        config.save_yaml(yaml_path)
        
        assert yaml_path.exists()
        
        # Load back
        loaded = RegistrationConfig.from_yaml(yaml_path)
        
        assert loaded.input_path == str(synthetic_zarr_data)
        assert loaded.downsample.xy == 2
        assert loaded.channels.k == 100.0
    
    def test_input_format_detection(self, synthetic_zarr_data, temp_dir):
        """Should detect input format from path."""
        config = RegistrationConfig(
            input_path=str(synthetic_zarr_data),
            output_dir=str(temp_dir / "output"),
        )
        
        assert config.input_format == 'zarr'
    
    def test_creates_output_dir(self, synthetic_zarr_data, temp_dir):
        """Should create output directory if it doesn't exist."""
        output = temp_dir / "new_output_dir"
        assert not output.exists()
        
        config = RegistrationConfig(
            input_path=str(synthetic_zarr_data),
            output_dir=str(output),
        )
        
        assert output.exists()

