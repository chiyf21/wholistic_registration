"""Unit tests for the synthetic-data generators in utils/generate_demo_data."""

from __future__ import annotations

import numpy as np

from wholistic_registration.utils import generate_demo_data as gdd


class TestGenerateCell:
    def test_shape_matches_image_size(self) -> None:
        img = gdd.generate_cell(center=(8.0, 8.0), radius=3.0, intensity=1.0, image_size=(16, 16))
        assert img.shape == (16, 16)

    def test_peak_is_near_center(self) -> None:
        img = gdd.generate_cell(center=(7.5, 7.5), radius=2.0, intensity=1.0, image_size=(16, 16))
        peak = np.unravel_index(np.argmax(img), img.shape)
        # The Gaussian peak should land on the pixel closest to the requested center.
        assert peak in {(7, 7), (7, 8), (8, 7), (8, 8)}


class TestGenerateMotionField:
    def test_shape_is_height_width_2(self) -> None:
        field = gdd.generate_motion_field(image_size=(16, 16), max_displacement=5.0, seed=0)
        assert field.shape == (16, 16, 2)
        assert field.dtype == np.float32

    def test_max_magnitude_bounded_by_displacement(self) -> None:
        field = gdd.generate_motion_field(image_size=(17, 17), max_displacement=10.0, seed=0)
        magnitudes = np.sqrt(field[..., 0] ** 2 + field[..., 1] ** 2)
        # The function constructs dr = max_displacement * exp(-r/100), so all
        # vectors must have magnitude in [0, max_displacement].
        assert magnitudes.min() >= 0.0
        assert magnitudes.max() <= 10.0 + 1e-5


class TestGenerateCellMovement:
    def test_frame_count(self) -> None:
        frames, motion = gdd.generate_cell_movement(
            num_frames=3, image_size=(16, 16), num_cells=2, seed=0
        )
        assert len(frames) == 3
        for frame in frames:
            assert frame.shape == (16, 16)

    def test_motion_field_shape(self) -> None:
        _, motion = gdd.generate_cell_movement(
            num_frames=2, image_size=(16, 16), num_cells=1, seed=0
        )
        assert motion.shape == (16, 16, 2)
