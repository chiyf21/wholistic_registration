"""wholistic_registration — whole-body cellular activity image registration.

Public API
----------
DefineParams        : build/save the configuration TOML for a registration run.
Registration_v3     : run the full 2D/3D non-rigid registration pipeline.
ReliableAnalysis    : run the reliability-mask analysis on registered output.

Example
-------
>>> from wholistic_registration import Registration_v3
>>> Registration_v3("path/to/config.toml", parallel=True)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wholistic_registration")
except PackageNotFoundError:  # editable install before metadata is generated
    __version__ = "0.0.0+unknown"

from .core.main_function import DefineParams, Registration_v3, ReliableAnalysis

__all__ = [
    "__version__",
    "DefineParams",
    "Registration_v3",
    "ReliableAnalysis",
]
