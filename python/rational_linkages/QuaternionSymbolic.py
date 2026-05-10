from typing import Optional, Sequence

import numpy
import sympy

from .Quaternion import Quaternion


class QuaternionSymbolic(Quaternion):
    """
    Symbolic quaternion backed by SymPy expressions.

    Subclass of :class:`.Quaternion` for algebraic
    computation. Typically, not instantiated directly — when the global backend is
    set to ``"sympy"`` via :func:`.set_backend`,
    :class:`.Quaternion` transparently returns instances
    of this class via its ``__new__`` factory.

    All arithmetic operators return :class:`QuaternionSymbolic` instances.
    Products are simplified via :func:`sympy.expand` and inverses via
    :func:`sympy.simplify` for clean algebraic output.

    Parameters
    ----------
    coeffs :
        Coefficients ``[w, x, y, z]`` as SymPy expressions or plain
        numbers. If ``None``, the identity quaternion ``[1, 0, 0, 0]``
        is constructed.

    Attributes
    ----------
    q : numpy.ndarray
        Object-dtype array of SymPy expressions ``[w, x, y, z]``.

    Raises
    ------
    ValueError
        If ``coeffs`` is not a 4-vector.

    Examples
    --------
    .. code-block:: python

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import Quaternion
        from sympy import symbols

        a, b, c, d = symbols("a b c d", real=True)
        q1 = Quaternion([a, b, c, d])
        q2 = Quaternion([1, 0, 0, 0])   # identity

        print(q1 * q2)   # QuaternionSymbolic([a, b, c, d])
        print(q1.norm())  # a**2 + b**2 + c**2 + d**2

        rational_linkages.set_backend("numpy")   # restore default

    .. clear-namespace::

    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, coeffs: Optional[Sequence] = None):
        if coeffs is not None:
            if len(coeffs) != 4:
                raise ValueError("QuaternionSymbolic: coeffs has to be 4-vector")
            self.q = numpy.array(
                [sympy.sympify(v) for v in coeffs], dtype=object
            )
        else:
            self.q = numpy.array(
                [sympy.Integer(1), sympy.Integer(0),
                 sympy.Integer(0), sympy.Integer(0)],
                dtype=object,
            )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        entries = ", ".join(str(v) for v in self.q)
        return f"Qt([{entries}])"

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __mul__(self, other: "Quaternion | int | float") -> "QuaternionSymbolic":
        """
        Multiply two quaternions, or scale by a scalar.

        Results are simplified via :func:`sympy.expand`.

        Parameters
        ----------
        other :
            Quaternion, or a scalar ``int`` / ``float``.

        Returns
        -------
        QuaternionSymbolic
            Hamilton product, or scalar-scaled quaternion.
        """
        if isinstance(other, Quaternion):
            w, x, y, z = self.q
            ow, ox, oy, oz = other.q
            return self.__class__([
                sympy.expand(w * ow - x * ox - y * oy - z * oz),
                sympy.expand(w * ox + x * ow + y * oz - z * oy),
                sympy.expand(w * oy - x * oz + y * ow + z * ox),
                sympy.expand(w * oz + x * oy - y * ox + z * ow),
            ])
        else:
            return self.__class__(self.q * sympy.sympify(other))

    def __rmul__(self, other: "int | float | sympy.Expr") -> "QuaternionSymbolic":
        """Scalar-on-left multiplication, delegates to ``__mul__``."""
        from numbers import Number
        if isinstance(other, (Number, sympy.Basic)):
            return self.__mul__(other)
        return NotImplemented

    def __eq__(self, other: "Quaternion") -> bool:
        """
        Test coefficient-wise equality via symbolic simplification.

        Parameters
        ----------
        other :
            Quaternion to compare against.

        Returns
        -------
        bool
            ``True`` if all coefficient differences simplify to zero.
        """
        return all(sympy.simplify(a - b) == 0
                   for a, b in zip(self.q, other.q))

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return coefficients as an object-dtype numpy array.

        Returns
        -------
        numpy.ndarray
            Object-dtype 4-vector ``[w, x, y, z]`` of SymPy expressions.
        """
        return numpy.array(self.q, dtype=object)

    def norm(self) -> sympy.Expr:
        """
        Quaternion norm (also called the Quadrance).

        Returns the squared length ``w² + x² + y² + z²``, not the
        Euclidean length. See `length` for the latter.

        Returns
        -------
        sympy.Expr
            Squared norm as a SymPy expression.
        """
        return sympy.expand(sum(v ** 2 for v in self.q))

    def length(self) -> sympy.Expr:
        """
        Euclidean length of the quaternion.

        Returns
        -------
        sympy.Expr
            ``sqrt(norm())`` as a SymPy expression.
        """
        return sympy.sqrt(self.norm())

    def inv(self) -> "QuaternionSymbolic":
        """
        Quaternion inverse.

        Returns
        -------
        QuaternionSymbolic
            ``conjugate() / norm()``, with each component simplified.
        """
        n = self.norm()
        return self.__class__([sympy.simplify(v / n) for v in self.conjugate().q])

    def eval(self, subs: dict) -> "QuaternionSymbolic":
        """
        Evaluate the quaternion by substituting symbols with values.

        Parameters
        ----------
        subs : dict
            Dictionary mapping SymPy symbols to values.

        Returns
        -------
        QuaternionSymbolic
            New quaternion with substitutions applied.

        Examples
        --------
        .. code-block:: python

            import rational_linkages
            rational_linkages.set_backend("sympy")

            from rational_linkages import Quaternion
            from sympy import symbols

            a, b, c, d = symbols("a b c d", real=True)
            q1 = Quaternion([a, b, c, d])
            subs = {a: 1, b: -2, c: 0, d: 0}
            q1_evaluated = q1.eval(subs)
            print(q1_evaluated)

        .. clear-namespace::

        """
        return self.__class__([v.subs(subs) for v in self.q])


    def evalf(self):
        """
        Replace rational numbers by numerical ones.

        Returns
        -------
        Quaternion
            Evaluated numerically.
        """
        return Quaternion(
            numpy.array([val.evalf() for val in self.coordinates], dtype=numpy.float64))