"""
Tests for rational_linkages.datasets loaders.

Run with:
    pytest tests/test_datasets.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rational_linkages.models import (
    bennett_ark24,
    cart_stl,
    collisions_free_6r,
    interp_4poses_6r,
    plane_fold_6r,
)
from rational_linkages import RationalMechanism


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PKL_LOADERS = [
    ("bennett_ark24", bennett_ark24),
    ("collisions_free_6r", collisions_free_6r),
    ("plane_fold_6r", plane_fold_6r),
    ("interp_4poses_6r", interp_4poses_6r),
]


# ---------------------------------------------------------------------------
# Pickle loaders — type and basic sanity
# ---------------------------------------------------------------------------

class TestPickleLoaders:

    @pytest.mark.parametrize("name,loader", PKL_LOADERS)
    def test_returns_rational_mechanism(self, name, loader):
        """Each loader must return a RationalMechanism instance."""
        result = loader()
        assert isinstance(result, RationalMechanism), (
            f"{name}() returned {type(result).__name__}, expected RationalMechanism"
        )

    @pytest.mark.parametrize("name,loader", PKL_LOADERS)
    def test_returns_new_object_each_call(self, name, loader):
        """Each call should deserialise a fresh object (no shared state)."""
        a = loader()
        b = loader()
        assert a is not b, f"{name}() returned the same object on two calls"

    @pytest.mark.parametrize("name,loader", PKL_LOADERS)
    def test_does_not_raise(self, name, loader):
        """Loading must not raise any exception."""
        try:
            loader()
        except Exception as exc:
            pytest.fail(f"{name}() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# STL loader
# ---------------------------------------------------------------------------

class TestCartStl:

    def test_returns_string(self):
        path = cart_stl()
        assert isinstance(path, str), f"cart_stl() should return str, got {type(path)}"

    def test_file_exists(self):
        path = cart_stl()
        assert os.path.isfile(path), f"cart_stl() path does not exist: {path}"

    def test_has_stl_extension(self):
        path = cart_stl()
        assert Path(path).suffix.lower() == ".stl", (
            f"cart_stl() should point to a .stl file, got: {path}"
        )

    def test_file_not_empty(self):
        path = cart_stl()
        assert os.path.getsize(path) > 0, f"cart_stl() file is empty: {path}"

    def test_returns_new_path_each_call(self):
        """Path string should be consistent across calls (same file)."""
        assert cart_stl() == cart_stl()


# ---------------------------------------------------------------------------
# importlib.resources resolution
# ---------------------------------------------------------------------------

class TestResourceResolution:
    """Ensure data files are bundled and accessible via importlib.resources."""

    def test_data_package_accessible(self):
        import importlib.resources
        pkg = importlib.resources.files("rational_linkages.data")
        assert pkg is not None

    @pytest.mark.parametrize("filename", [
        "bennett_ark24.pkl",
        "collisions_free_6r.pkl",
        "plane_fold_6r.pkl",
        "interp_4poses_6r.pkl",
        "cart.stl",
    ])
    def test_resource_file_exists(self, filename):
        import importlib.resources
        ref = importlib.resources.files("rational_linkages.data").joinpath(filename)
        with importlib.resources.as_file(ref) as path:
            assert path.exists(), (
                f"Bundled resource '{filename}' not found at {path}. "
                "Check [tool.setuptools.package-data] in pyproject.toml."
            )