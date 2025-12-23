"""Tests for reference computation."""

import pytest
import numpy as np

from ..core.reference import ReferenceComputer, compute_reference_simple
from ..config import ChannelConfig


class TestReferenceComputer:
    """Tests for ReferenceComputer class."""
    
    @pytest.fixture
    def reference_computer(self):
        """Create a CPU-based reference computer for testing."""
        return ReferenceComputer(
            max_correlation_frames=10,
            spatial_downsample=2,
            device='cpu',
        )
    
    @pytest.fixture
    def sample_frames(self):
        """Generate sample frame data."""
        np.random.seed(42)
        T, Z, Y, X = 20, 5, 32, 32
        
        # Create base pattern
        base = np.random.rand(Z, Y, X).astype(np.float32) * 100
        
        # Add slight variations for each frame
        membrane = np.stack([
            base + np.random.rand(Z, Y, X).astype(np.float32) * 10
            for _ in range(T)
        ])
        calcium = membrane * 0.8 + np.random.rand(T, Z, Y, X).astype(np.float32) * 20
        
        return membrane, calcium
    
    def test_compute_returns_correct_shape(self, reference_computer, sample_frames):
        """Reference should have spatial dimensions of input."""
        membrane, calcium = sample_frames
        config = ChannelConfig(dual_channel=True, k=10.0, transform='log10')
        
        ref = reference_computer.compute(membrane, calcium, config)
        
        # Should be (Z, Y, X)
        assert ref.shape == membrane.shape[1:]
    
    def test_compute_single_channel(self, reference_computer, sample_frames):
        """Should work with single channel (dual_channel=False)."""
        membrane, calcium = sample_frames
        config = ChannelConfig(dual_channel=False)
        
        ref = reference_computer.compute(membrane, None, config)
        
        assert ref.shape == membrane.shape[1:]
    
    def test_reference_is_smooth(self, reference_computer, sample_frames):
        """Reference should be smoother than individual frames (averaged)."""
        membrane, calcium = sample_frames
        config = ChannelConfig(dual_channel=False)
        
        ref = reference_computer.compute(membrane, None, config)
        
        # Reference should have lower variance than individual frames
        # (due to averaging)
        ref_var = np.var(ref)
        frame_vars = [np.var(membrane[i]) for i in range(len(membrane))]
        mean_frame_var = np.mean(frame_vars)
        
        # Reference variance should be similar or lower
        # (not a strict test, but reference shouldn't be more noisy)
        assert ref_var <= mean_frame_var * 1.5
    
    def test_dual_channel_increases_intensity(self, reference_computer, sample_frames):
        """Dual channel reference should have higher mean than membrane only."""
        membrane, calcium = sample_frames
        
        config_single = ChannelConfig(dual_channel=False)
        config_dual = ChannelConfig(dual_channel=True, k=50.0, transform='log10')
        
        ref_single = reference_computer.compute(membrane, None, config_single)
        ref_dual = reference_computer.compute(membrane, calcium, config_dual)
        
        # Dual channel adds transformed calcium, should increase mean
        assert ref_dual.mean() > ref_single.mean()
    
    def test_handles_2d_input(self, reference_computer):
        """Should handle 2D (T, Y, X) input."""
        np.random.seed(42)
        T, Y, X = 10, 32, 32
        membrane = np.random.rand(T, Y, X).astype(np.float32) * 100
        calcium = membrane * 0.8
        
        config = ChannelConfig(dual_channel=True, k=10.0)
        ref = reference_computer.compute(membrane, calcium, config)
        
        # Should be (Y, X)
        assert ref.shape == (Y, X)


class TestComputeReferenceSimple:
    """Tests for simple function interface."""
    
    def test_basic_usage(self):
        """Simple interface should work."""
        np.random.seed(42)
        membrane = np.random.rand(10, 5, 32, 32).astype(np.float32) * 100
        calcium = membrane * 0.8
        
        ref = compute_reference_simple(
            membrane, calcium,
            dual_channel=True,
            transform='log10',
            k=50.0,
            device='cpu',
        )
        
        assert ref.shape == (5, 32, 32)
    
    def test_single_channel(self):
        """Should work with single channel."""
        np.random.seed(42)
        membrane = np.random.rand(10, 5, 32, 32).astype(np.float32) * 100
        
        ref = compute_reference_simple(
            membrane, None,
            dual_channel=False,
            device='cpu',
        )
        
        assert ref.shape == (5, 32, 32)

