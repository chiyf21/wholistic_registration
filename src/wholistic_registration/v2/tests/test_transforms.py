"""Tests for channel transforms."""

import pytest
import numpy as np

from ..core.transforms import apply_channel_transform, combine_channels


class TestApplyChannelTransform:
    """Tests for apply_channel_transform function."""
    
    def test_raw_transform(self):
        """Raw transform should multiply by k."""
        data = np.array([1.0, 10.0, 100.0])
        result = apply_channel_transform(data, 'raw', k=2.0)
        np.testing.assert_array_almost_equal(result, data * 2.0)
    
    def test_log10_transform(self):
        """Log10 transform should apply log10(1+x)."""
        data = np.array([0.0, 9.0, 99.0])  # log10(1+0)=0, log10(10)=1, log10(100)=2
        result = apply_channel_transform(data, 'log10', k=1.0)
        expected = np.log10(1 + data)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sqrt_transform(self):
        """Sqrt transform should apply square root."""
        data = np.array([0.0, 4.0, 9.0, 16.0])
        result = apply_channel_transform(data, 'sqrt', k=1.0)
        expected = np.sqrt(data)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_log2_transform(self):
        """Log2 transform should apply log2(1+x)."""
        data = np.array([0.0, 1.0, 3.0, 7.0])  # log2(1)=0, log2(2)=1, log2(4)=2, log2(8)=3
        result = apply_channel_transform(data, 'log2', k=1.0)
        expected = np.log2(1 + data)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_k_scaling(self):
        """K parameter should scale the result."""
        data = np.array([9.0])  # log10(10) = 1
        result = apply_channel_transform(data, 'log10', k=50.0)
        np.testing.assert_almost_equal(result[0], 50.0)
    
    def test_invalid_transform(self):
        """Invalid transform should raise ValueError."""
        data = np.array([1.0])
        with pytest.raises(ValueError, match="Unknown transform"):
            apply_channel_transform(data, 'invalid', k=1.0)
    
    def test_preserves_shape(self):
        """Transform should preserve array shape."""
        data = np.random.rand(5, 10, 10)
        result = apply_channel_transform(data, 'log10', k=50.0)
        assert result.shape == data.shape
    
    def test_handles_zeros(self):
        """Transform should handle zeros correctly."""
        data = np.array([0.0, 0.0, 0.0])
        
        # log10(1+0) = 0
        result_log = apply_channel_transform(data, 'log10', k=1.0)
        np.testing.assert_array_almost_equal(result_log, [0, 0, 0])
        
        # sqrt(0) = 0
        result_sqrt = apply_channel_transform(data, 'sqrt', k=1.0)
        np.testing.assert_array_almost_equal(result_sqrt, [0, 0, 0])


class TestCombineChannels:
    """Tests for combine_channels function."""
    
    def test_basic_combination(self):
        """Combined = membrane + k * transform(calcium)."""
        membrane = np.array([100.0, 200.0, 300.0])
        calcium = np.array([9.0, 9.0, 9.0])  # log10(10) = 1
        
        result = combine_channels(membrane, calcium, 'log10', k=50.0)
        expected = membrane + 50.0  # membrane + 50 * 1
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_raw_combination(self):
        """With raw transform, combined = membrane + k * calcium."""
        membrane = np.array([100.0])
        calcium = np.array([10.0])
        
        result = combine_channels(membrane, calcium, 'raw', k=0.5)
        expected = np.array([105.0])  # 100 + 0.5 * 10
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_3d_combination(self):
        """Should work with 3D arrays."""
        membrane = np.random.rand(5, 10, 10) * 100
        calcium = np.random.rand(5, 10, 10) * 100
        
        result = combine_channels(membrane, calcium, 'log10', k=50.0)
        
        assert result.shape == membrane.shape
        # Result should be larger than membrane (calcium adds positive values)
        assert result.sum() > membrane.sum()

