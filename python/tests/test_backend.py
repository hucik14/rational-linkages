import pytest

from rational_linkages.backend import get_backend, is_symbolic, set_backend


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


class TestSetBackend:

    def test_set_numpy(self):
        set_backend("numpy")
        assert get_backend() == "numpy"

    def test_set_sympy(self):
        set_backend("sympy")
        assert get_backend() == "sympy"

    def test_set_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("jax")

    def test_set_empty_string_raises(self):
        with pytest.raises(ValueError):
            set_backend("")

    def test_returns_none(self):
        assert set_backend("numpy") is None


class TestGetBackend:

    def test_default_is_numpy(self):
        set_backend("numpy")
        assert get_backend() == "numpy"

    def test_returns_string(self):
        assert isinstance(get_backend(), str)

    def test_reflects_set_backend(self):
        set_backend("sympy")
        assert get_backend() == "sympy"
        set_backend("numpy")
        assert get_backend() == "numpy"


class TestIsSymbolic:

    def test_false_for_numpy(self):
        set_backend("numpy")
        assert is_symbolic() is False

    def test_true_for_sympy(self):
        set_backend("sympy")
        assert is_symbolic() is True

    def test_returns_bool(self):
        assert isinstance(is_symbolic(), bool)

    def test_toggles_with_set_backend(self):
        set_backend("sympy")
        assert is_symbolic() is True
        set_backend("numpy")
        assert is_symbolic() is False