"""
Pytest fixtures for testing.

Provides reusable test data and configuration fixtures.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from .synthetic_data import SyntheticDataGenerator, SyntheticDataConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory that's cleaned up after tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def small_synthetic_data():
    """Generate small synthetic dataset for quick tests."""
    config = SyntheticDataConfig(
        n_frames=10,
        shape_zyx=(5, 32, 32),
        n_spheres=5,
        motion_amplitude=2.0,
        motion_type='translation',
    )
    generator = SyntheticDataGenerator(config)
    membrane, calcium, motion_gt = generator.generate()
    return {
        'membrane': membrane,
        'calcium': calcium,
        'motion_gt': motion_gt,
        'config': config,
    }


@pytest.fixture
def synthetic_tiff_data(temp_dir, small_synthetic_data):
    """Generate synthetic data and save as TIFF series."""
    config = SyntheticDataConfig(
        n_frames=10,
        shape_zyx=(5, 32, 32),
        n_spheres=5,
        motion_amplitude=2.0,
    )
    generator = SyntheticDataGenerator(config)
    generator.generate()
    path = generator.save_as_tiff(str(temp_dir / "synthetic_tiff"))
    return path


@pytest.fixture
def synthetic_zarr_data(temp_dir):
    """Generate synthetic data and save as Zarr."""
    config = SyntheticDataConfig(
        n_frames=10,
        shape_zyx=(5, 32, 32),
        n_spheres=5,
        motion_amplitude=2.0,
    )
    generator = SyntheticDataGenerator(config)
    generator.generate()
    path = generator.save_as_zarr(str(temp_dir / "synthetic.zarr"))
    return path


@pytest.fixture
def sample_config(temp_dir, synthetic_zarr_data):
    """Create a sample registration configuration."""
    from ..config import RegistrationConfig, DownsampleConfig, ChannelConfig
    
    config = RegistrationConfig(
        input_path=str(synthetic_zarr_data),
        output_dir=str(temp_dir / "output"),
        downsample=DownsampleConfig(xy=1, t_chunk=5),
        channels=ChannelConfig(dual_channel=True, k=10, transform='log10'),
    )
    return config

