"""Tests for synthetic data generation."""

import pytest
import numpy as np

from .synthetic_data import (
    SyntheticDataGenerator, 
    SyntheticDataConfig,
    generate_simple_test_data,
)


class TestSyntheticDataConfig:
    """Tests for SyntheticDataConfig."""
    
    def test_default_config(self):
        """Default config should be valid."""
        config = SyntheticDataConfig()
        assert config.n_frames == 50
        assert config.shape_zyx == (10, 128, 128)
    
    def test_custom_config(self):
        """Custom config should work."""
        config = SyntheticDataConfig(
            n_frames=20,
            shape_zyx=(5, 64, 64),
            motion_type='drift',
        )
        assert config.n_frames == 20
        assert config.motion_type == 'drift'


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator with small config for fast tests."""
        config = SyntheticDataConfig(
            n_frames=5,
            shape_zyx=(3, 16, 16),
            n_spheres=3,
            random_seed=42,
        )
        return SyntheticDataGenerator(config)
    
    def test_generate_returns_correct_shapes(self, generator):
        """Generated data should have correct shapes."""
        membrane, calcium, motion = generator.generate()
        
        cfg = generator.config
        expected_shape = (cfg.n_frames, *cfg.shape_zyx)
        motion_shape = (*expected_shape, 3)
        
        assert membrane.shape == expected_shape
        assert calcium.shape == expected_shape
        assert motion.shape == motion_shape
    
    def test_membrane_positive(self, generator):
        """Membrane values should be positive."""
        membrane, _, _ = generator.generate()
        assert np.all(membrane >= 0)
    
    def test_calcium_positive(self, generator):
        """Calcium values should be positive."""
        _, calcium, _ = generator.generate()
        assert np.all(calcium >= 0)
    
    def test_motion_types(self):
        """All motion types should work."""
        for motion_type in ['none', 'translation', 'sine', 'drift']:
            config = SyntheticDataConfig(
                n_frames=3,
                shape_zyx=(2, 8, 8),
                motion_type=motion_type,
            )
            generator = SyntheticDataGenerator(config)
            membrane, calcium, motion = generator.generate()
            
            assert membrane.shape[0] == 3
            
            if motion_type == 'none':
                assert np.allclose(motion, 0)
    
    def test_motion_amplitude(self):
        """Motion should respect amplitude setting."""
        config = SyntheticDataConfig(
            n_frames=10,
            shape_zyx=(3, 16, 16),
            motion_type='translation',
            motion_amplitude=5.0,
        )
        generator = SyntheticDataGenerator(config)
        _, _, motion = generator.generate()
        
        # Max motion should be close to amplitude
        max_motion = np.abs(motion).max()
        assert max_motion <= config.motion_amplitude * 1.5
    
    def test_reproducibility(self):
        """Same seed should produce same data."""
        config = SyntheticDataConfig(
            n_frames=3,
            shape_zyx=(2, 8, 8),
            random_seed=42,
        )
        
        gen1 = SyntheticDataGenerator(config)
        mem1, _, _ = gen1.generate()
        
        gen2 = SyntheticDataGenerator(config)
        mem2, _, _ = gen2.generate()
        
        np.testing.assert_array_equal(mem1, mem2)
    
    def test_save_as_tiff(self, generator, tmp_path):
        """Should save as TIFF series."""
        generator.generate()
        output_path = generator.save_as_tiff(str(tmp_path / "tiff_output"))
        
        # Check files exist
        assert (output_path / "membrane").exists()
        assert (output_path / "calcium").exists()
        assert (output_path / "motion_ground_truth.npy").exists()
        assert (output_path / "config.json").exists()
        
        # Check frame count
        mem_files = list((output_path / "membrane").glob("*.tif"))
        assert len(mem_files) == generator.config.n_frames
    
    def test_save_as_zarr(self, generator, tmp_path):
        """Should save as Zarr."""
        generator.generate()
        output_path = generator.save_as_zarr(str(tmp_path / "data.zarr"))
        
        import zarr
        root = zarr.open(str(output_path), mode='r')
        
        assert 'membrane' in root
        assert 'calcium' in root
        assert 'motion_ground_truth' in root
        
        assert root['membrane'].shape == generator.membrane.shape


class TestGenerateSimpleTestData:
    """Tests for helper function."""
    
    def test_basic_call(self):
        """Should work with defaults."""
        membrane, calcium, motion = generate_simple_test_data()
        
        assert membrane.shape == (20, 5, 64, 64)
        assert calcium.shape == (20, 5, 64, 64)
        assert motion.shape == (20, 5, 64, 64, 3)
    
    def test_custom_parameters(self):
        """Should respect custom parameters."""
        membrane, calcium, motion = generate_simple_test_data(
            n_frames=10,
            shape=(3, 32, 32),
            motion_amplitude=2.0,
        )
        
        assert membrane.shape == (10, 3, 32, 32)

