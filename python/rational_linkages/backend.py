"""
Global backend configuration for rational_linkages.

Controls whether numeric (NumPy) or symbolic (SymPy) computation is used
across all classes. Call :func:`set_backend` once at the top of a script
before constructing any objects.

Examples
--------
.. code-block:: python

    import rational_linkages
    rational_linkages.set_backend("sympy")

    from rational_linkages import Quaternion
    from sympy import symbols

    a, b = symbols("a b", real=True)
    q = Quaternion([a, b, 0, 0])   # returns QuaternionSymbolic

"""

_BACKEND: str = "numpy"
_VALID_BACKENDS: tuple = ("numpy", "sympy")


def set_backend(name: str) -> None:
    """
    Set the global computation backend.

    Must be called before constructing any objects. Switching backends
    after objects have already been created leads to undefined behaviour
    when mixing instances from different backends.

    Parameters
    ----------
    name :
        Backend identifier. Either ``"numpy"`` (default) or ``"sympy"``.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised backend.

    Examples
    --------
    .. code-block:: python

        import rational_linkages

        rational_linkages.set_backend("sympy")   # enable symbolic computation
        rational_linkages.set_backend("numpy")   # restore numeric default
    """
    global _BACKEND
    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Valid options are: {_VALID_BACKENDS}"
        )
    _BACKEND = name


def get_backend() -> str:
    """
    Return the currently active backend name.

    Returns
    -------
    str
        ``"numpy"`` or ``"sympy"``.
    """
    return _BACKEND


def is_symbolic() -> bool:
    """
    Return whether the active backend is symbolic.

    Returns
    -------
    bool
        ``True`` if the active backend is ``"sympy"``.
    """
    return _BACKEND == "sympy"