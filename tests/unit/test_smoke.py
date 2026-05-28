"""Smoke tests: the package imports and exposes its public API."""

from __future__ import annotations


def test_package_imports() -> None:
    import wholistic_registration

    assert hasattr(wholistic_registration, "__version__")
    assert isinstance(wholistic_registration.__version__, str)
    assert wholistic_registration.__version__


def test_public_api_callables() -> None:
    from wholistic_registration import (
        DefineParams,
        ReliableAnalysis,
        Registration_v3,
    )

    assert callable(DefineParams)
    assert callable(Registration_v3)
    assert callable(ReliableAnalysis)


def test_all_exports_resolvable() -> None:
    import wholistic_registration

    for name in wholistic_registration.__all__:
        assert hasattr(wholistic_registration, name), f"missing export: {name}"
