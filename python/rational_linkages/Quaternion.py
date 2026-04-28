from typing import Optional, Sequence

import numpy

from .backend import is_symbolic


class Quaternion:
    """
    Quaternion representing a 4-dimensional number ``[w, x, y, z]``.

    Used as the building block of `DualQuaternion`, where the primal and
    dual parts are each a ``Quaternion``.

    By default, all computation is performed with NumPy (``float64``). When
    the global backend is set to ``"sympy"`` via
    :func:`~rational_linkages.set_backend`, construction transparently
    returns a :class:`~rational_linkages.QuaternionSymbolic` instance
    instead, with no change to the calling code required.

    Parameters
    ----------
    coeffs :
        Coefficients ``[w, x, y, z]``. If ``None``, the identity
        quaternion ``[1, 0, 0, 0]`` is constructed.

    Attributes
    ----------
    q : np.ndarray
        4-vector of quaternion coefficients ``[w, x, y, z]``.

    Raises
    ------
    ValueError
        If ``coeffs`` is not a 4-vector.

    Examples
    --------
    .. code-block:: python

        from rational_linkages import Quaternion


        identity = Quaternion()
        q = Quaternion([1.0, 2.0, 3.0, 4.0])
        print(identity)
        print(q)

    .. code-block:: python

        # Fully symbolic backend — set once at the top of your script

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import Quaternion
        from sympy import symbols


        a, b = symbols("a b", real=True)
        q = Quaternion([a, b, 0, 0])   # transparently returns QuaternionSymbolic

    """
    __hash__ = None  # mutable, not hashable by default; can be overridden in subclasses
    __array_priority__ = 20.0  # prioritizes Quaternion over numpy array __rmul__

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def __new__(cls, coeffs=None):
        """
        Intercept construction and return a
        :class:`~rational_linkages.QuaternionSymbolic` when the global
        backend is ``"sympy"``.

        Only applied when ``cls`` is exactly ``Quaternion``; subclass
        constructors are never redirected, preventing infinite recursion.

        Parameters
        ----------
        coeffs :
            Forwarded unchanged to ``__init__``.

        Returns
        -------
        Quaternion or QuaternionSymbolic
            A numeric or symbolic instance depending on the active backend.
        """
        if cls is Quaternion:
            symbolic = is_symbolic() or (
                    coeffs is not None
                    and any(hasattr(c, 'free_symbols') for c in coeffs)
            )
            if symbolic:
                from .QuaternionSymbolic import QuaternionSymbolic
                return object.__new__(QuaternionSymbolic)
        return object.__new__(cls)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, coeffs: Optional[Sequence[float]] = None):
        if coeffs is not None:
            if len(coeffs) != 4:
                raise ValueError("Quaternion: coeffs has to be 4-vector")
            try:
                self.q = numpy.asarray(coeffs, dtype=numpy.float64)
            except (TypeError, ValueError):
                self.q = numpy.asarray(coeffs, dtype=object)
        else:
            self.q = numpy.array([1.0, 0.0, 0.0, 0.0], dtype=numpy.float64)

    @property
    def real(self):
        return self.q[0]

    @property
    def imag(self):
        return self.q[1:]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """
        Get a coefficient by index.

        Parameters
        ----------
        idx :
            Index in range ``0..3``.

        Returns
        -------
        float
            Coefficient at ``idx``.
        """
        return self.q[idx]

    def __setitem__(self, idx, value):
        """
        Set a coefficient by index.

        Parameters
        ----------
        idx :
            Index in range ``0..3``.
        value :
            New coefficient value.
        """
        self.q[idx] = value

    def __len__(self):
        """Length of the quaternion, always 4."""
        return 4

    def __iter__(self):
        """Iterate over quaternions."""
        return iter(self.q)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        q = numpy.array2string(self.array(),
                               precision=16,
                               suppress_small=True,
                               separator=', ',
                               max_line_width=100000)
        return f"{self.__class__.__qualname__}({q})"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coordinates(self):
        """
        Return the coordinates of the quaternion.

        Returns
        -------
        numpy.ndarray
            4-vector of quaternion coordinates.
        """
        return self.q.copy()

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: "Quaternion") -> "Quaternion":
        """
        Add two quaternions.

        Parameters
        ----------
        other :
            Quaternion to add.

        Returns
        -------
        Quaternion
            Element-wise sum.
        """
        return self.__class__(self.q + other.q)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        """
        Subtract two quaternions.

        Parameters
        ----------
        other :
            Quaternion to subtract.

        Returns
        -------
        Quaternion
            Element-wise difference.
        """
        return self.__class__(self.q - other.q)

    def __mul__(self, other: "Quaternion | int | float") -> "Quaternion":
        """
        Multiply two quaternions, or scale by a scalar.

        Parameters
        ----------
        other :
            Quaternion, or a scalar ``int`` / ``float``.

        Returns
        -------
        Quaternion
            Hamilton product, or scalar-scaled quaternion.
        """
        if isinstance(other, Quaternion):
            w, x, y, z = self.q
            ow, ox, oy, oz = other.q
            return self.__class__(numpy.array([
                w * ow - x * ox - y * oy - z * oz,
                w * ox + x * ow + y * oz - z * oy,
                w * oy - x * oz + y * ow + z * ox,
                w * oz + x * oy - y * ox + z * ow,
            ]))
        else:
            return self.__class__(self.q * other)

    def __rmul__(self, other: "Quaternion | int | float") -> "Quaternion":
        """Scalar-on-left multiplication, delegates to ``__mul__``."""
        from numbers import Number  # lazy import
        if isinstance(other, Number):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(self, other: "Quaternion | int | float") -> "Quaternion":
        """
        Divide by a quaternion or scalar.

        Parameters
        ----------
        other :
            Quaternion, or a scalar ``int`` / ``float``.

        Returns
        -------
        Quaternion
            ``self * other.inv()`` for quaternions, or element-wise
            division for scalars.
        """
        if isinstance(other, Quaternion):
            return self * other.inv()
        return self.__class__(self.q / other)

    def __neg__(self) -> "Quaternion":
        """
        Negate the quaternion.

        Returns
        -------
        Quaternion
            Quaternion with all coefficients negated.
        """
        return self.__class__(-self.q)

    def __eq__(self, other: "Quaternion") -> bool:
        """
        Test coefficient-wise equality.

        Parameters
        ----------
        other :
            Quaternion to compare against.

        Returns
        -------
        bool
            ``True`` if all coefficients are equal.
        """
        return numpy.array_equal(self.array(), other.array())

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return coefficients as a numpy array.

        Returns
        -------
        np.ndarray
            4-vector ``[w, x, y, z]``.

        Examples
        --------

        .. code-block:: python
            from rational_linkages import Quaternion

            q = Quaternion([1, 2, 3, 4])
            q_array = q.array()
            print(q_array)  # np.array([1., 2., 3., 4.])

        """
        return self.q.copy()

    def conjugate(self) -> "Quaternion":
        """
        Quaternion conjugate.

        Returns
        -------
        Quaternion
            ``[w, -x, -y, -z]``.


        Examples
        --------

        .. code-block:: python
            from rational_linkages import Quaternion

            q = Quaternion([1, 2, 3, 4])
            q_conj = q.conjugate()
            print(q_conj)  # [1., -2., -3., -4.]

        """
        return self.__class__([self.q[0], -self.q[1], -self.q[2], -self.q[3]])

    def norm(self) -> float:
        """
        Quaternion norm (also called the Quadrance).

        Returns the squared length ``w² + x² + y² + z²``, not the
        Euclidean length. See `length` for the latter.

        Returns
        -------
        float
            Squared norm.

        Examples
        --------

        .. code-block:: python
            from rational_linkages import Quaternion

            q = Quaternion([1, 2, 3, 4])
            q_quadrance = q.norm()
            print(q_quadrance)

        """
        return float(numpy.dot(self.q, self.q))

    def length(self) -> float:
        """
        Euclidean length of the quaternion.

        Returns
        -------
        float
            ``sqrt(norm())``.

        Examples
        --------

        .. code-block:: python
            from rational_linkages import Quaternion

            q = Quaternion([1, 2, 3, 4])
            q_len = q.length()
            print(q_len)

        """
        return float(numpy.sqrt(self.norm()))

    def inv(self) -> "Quaternion":
        """
        Quaternion inverse.

        Returns
        -------
        Quaternion
            ``conjugate() / norm()``.

        Examples
        --------

        .. code-block:: python
            from rational_linkages import Quaternion

            q = Quaternion([1, 2, 3, 4])
            q_inv = q.inv()
            print(q_inv)

        """
        return self.__class__(self.conjugate().q / self.norm())

    def normalize(self):
        """
        Normalize the quaternion to unit length.

        Returns
        -------
        Quaternion
            Normalized quaternion with the same direction but length 1.

        Examples
        --------

        .. code-block:: python
            from rational_linkages import Quaternion


            q = Quaternion([2, 1, -2, 0])
            q_normalized = q.normalize()
            print(q_normalized)

        """
        return self / self.length()


    def eval(self, subs: dict) -> "Quaternion":
        """
        Evaluate the quaternion by substituting symbols with values.

        Parameters
        ----------
        subs : dict
            Dictionary mapping SymPy symbols to numeric values.

        Returns
        -------
        Quaternion
            New numeric quaternion with substitutions applied, backed by
            ``float64``.

        Examples
        --------
        .. code-block:: python
            from rational_linkages import Quaternion
            from sympy import symbols


            t = symbols("t")
            q1 = Quaternion([5*t, 3, 0, -t])
            q_numeric = q1.eval({t : 1})
            print(type(q_numeric))   # Quaternion (numeric)
            print(q_numeric)

        """
        evaluated = [float(v.subs(subs)) if hasattr(v, "subs") else float(v)
                     for v in self.q]
        return Quaternion(evaluated)

    def evalf(self) -> "Quaternion":
        """
        Placeholder for QuaternionSymbolic.evalf() method.

        Returns
        -------
        Quaternion
            Evaluated quaternion.
        """
        return self