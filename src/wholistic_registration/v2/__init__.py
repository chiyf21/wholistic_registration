"""
Wholistic Registration v2 - Clean, modular implementation.

A volumetric image registration pipeline for dual-channel microscopy data.
Supports ND2, Zarr, and TIFF inputs with OME-TIFF output.

Example:
    >>> from wholistic_registration.v2 import RegistrationPipeline, RegistrationConfig
    >>> config = RegistrationConfig.from_yaml("config.yaml")
    >>> pipeline = RegistrationPipeline(config)
    >>> pipeline.run()
"""

__version__ = "2.0.0"

from .config import RegistrationConfig, DownsampleConfig, ChannelConfig, PyramidConfig
from .pipeline import RegistrationPipeline
from .io import create_reader, OMETiffWriter

__all__ = [
    "RegistrationConfig",
    "DownsampleConfig", 
    "ChannelConfig",
    "PyramidConfig",
    "RegistrationPipeline",
    "create_reader",
    "OMETiffWriter",
]

