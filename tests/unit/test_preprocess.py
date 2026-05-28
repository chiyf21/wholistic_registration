"""Unit tests for pure-NumPy preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from wholistic_registration.utils import preprocess


class TestAutoContrast:
    def test_output_range_is_zero_to_one(self, synthetic_image_2d: np.ndarray) -> None:
        out = preprocess.auto_contrast(synthetic_image_2d)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_dtype_is_floating(self, synthetic_image_2d: np.ndarray) -> None:
        out = preprocess.auto_contrast(synthetic_image_2d)
        assert np.issubdtype(out.dtype, np.floating)

    def test_shape_preserved(self, synthetic_image_3d: np.ndarray) -> None:
        out = preprocess.auto_contrast(synthetic_image_3d)
        assert out.shape == synthetic_image_3d.shape

    def test_clipping_at_percentiles(self) -> None:
        img = np.arange(100, dtype=np.float32)
        out = preprocess.auto_contrast(img, low_percentile=10, high_percentile=90)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)
        # Values below the 10th percentile clip to 0; above the 90th clip to 1.
        assert out[0] == pytest.approx(0.0, abs=1e-6)
        assert out[-1] == pytest.approx(1.0, abs=1e-6)


class TestNormalizeTo255:
    def test_output_range(self, synthetic_image_2d: np.ndarray) -> None:
        out = preprocess.normalize_to_255(synthetic_image_2d)
        assert out.min() >= 0.0
        assert out.max() <= 255.0

    def test_shape_preserved(self, synthetic_image_2d: np.ndarray) -> None:
        out = preprocess.normalize_to_255(synthetic_image_2d)
        assert out.shape == synthetic_image_2d.shape


class TestRobustMeanStd:
    def test_excludes_high_outliers(self) -> None:
        data = np.concatenate([np.ones(99, dtype=np.float32), np.array([1e6])])
        mean, std = preprocess.robust_mean_std(data, percentile=95)
        # The outlier sits above the 95th percentile and must be excluded.
        assert mean == pytest.approx(1.0, abs=1e-6)
        assert std == pytest.approx(0.0, abs=1e-6)

    def test_returns_two_floats(self, synthetic_image_2d: np.ndarray) -> None:
        mean, std = preprocess.robust_mean_std(synthetic_image_2d)
        assert isinstance(float(mean), float)
        assert isinstance(float(std), float)
        assert std >= 0.0
