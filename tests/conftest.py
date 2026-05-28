"""Shared pytest fixtures for the wholistic_registration test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded numpy random generator for deterministic tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def synthetic_image_2d(rng: np.random.Generator) -> np.ndarray:
    """A small 2D float32 image with known intensity range [0, 1000)."""
    return rng.uniform(0.0, 1000.0, size=(32, 32)).astype(np.float32)


@pytest.fixture
def synthetic_image_3d(rng: np.random.Generator) -> np.ndarray:
    """A small 3D float32 volume (Z, Y, X) with known intensity range."""
    return rng.uniform(0.0, 1000.0, size=(4, 16, 16)).astype(np.float32)
