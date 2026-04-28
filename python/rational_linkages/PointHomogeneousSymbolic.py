from typing import Optional, Sequence

import numpy
import sympy

from .PointHomogeneous import PointHomogeneous


class PointHomogeneousSymbolic(PointHomogeneous):
    """
    Symbolic point in projective space backed by SymPy expressions.

    Subclass of :class:`~rational_linkages.PointHomogeneous` for algebraic computation.
    Typically not instantiated directly — when the global backend is set to
    ``"sympy"`` via :func:`~rational_linkages.set_backend`,
    :class:`~rational_linkages.PointHomogeneous` transparently returns instances of this
    class via its ``__new__`` factory.

    Parameters
    ----------
    point :
        Sequence of homogeneous coordinates as SymPy expressions or plain
        numbers. If ``None``, the origin ``[1, 0, 0, 0]`` is constructed.

    Attributes
    ----------
    coordinates : np.ndarray
        Object-dtype array of SymPy expressions ``[w, x, y, ...]``.
    is_at_infinity : bool
        ``True`` when the homogeneous coordinate simplifies to zero.
    is_2d : bool
        ``True`` if the point is in ℙ² (3 homogeneous coordinates).
    is_3d : bool
        ``True`` if the point is in ℙ³ (4 homogeneous coordinates).

    Examples
    --------
    .. code-block:: python

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import PointHomogeneous
        from sympy import symbols

        w, x, y, z = symbols("w x y z", real=True)
        p = PointHomogeneous([w, x, y, z])    # transparently returns PointHomogeneousSymbolic
        print(p.normalize())        # PointHomogeneousSymbolic([1, x/w, y/w, z/w])

        rational_linkages.set_backend("numpy")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, point: Optional[Sequence] = None, rational: bool = False):

        # rationalize the input
        if rational:
            # check if any element is not already a sympy object
            if not all(isinstance(c, sympy.Basic) for c in point):
                point = [sympy.Rational(v) for v in point]

        self.is_rational = True
        # Bypass PointHomogeneous.__init__ bookkeeping that assumes float64; redo it
        # with symbolic-aware helpers.
        self.coordinates = self._initialize_coordinates(point)
        self._is_at_infinity = None
        self.is_2d = True if len(self.coordinates) == 3 else False
        self.is_3d = True if len(self.coordinates) == 4 else False
        self._normalized: Optional["PointHomogeneousSymbolic"] = None

    def _initialize_coordinates(self, point: Optional[Sequence]) -> numpy.ndarray:
        """
        Initialize the coordinate array with SymPy expressions.

        Parameters
        ----------
        point :
            Sequence of coordinates, or ``None`` for the origin in ℙ³.

        Returns
        -------
        np.ndarray
            Object-dtype array of SymPy expressions.
        """
        if point is None:
            return numpy.array(
                [sympy.Integer(1), sympy.Integer(0),
                 sympy.Integer(0), sympy.Integer(0)],
                dtype=object,
            )
        return numpy.array([sympy.sympify(v) for v in point], dtype=object)

    def _check_if_at_infinity(self) -> bool:
        """
        Return ``True`` if the homogeneous coordinate simplifies to zero.

        Returns
        -------
        bool
        """
        return sympy.simplify(self.coordinates[0]) == sympy.Integer(0)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        entries = ", ".join(str(v) for v in self.coordinates)
        return f"Pt([{entries}])"

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def norm(self):
        """
        Return the norm of the point.

        Returns
        -------
        sympy.Expr
             Square root of the sum of squares of the coordinates.
        """
        return sympy.sqrt(sum(v ** 2 for v in self.normalized_euclidean()))

    def normalize(self) -> "PointHomogeneousSymbolic":
        """
        Return the normalized point (homogeneous coordinate scaled to 1).

        Each coordinate is divided by ``w`` and simplified via
        :func:`sympy.simplify`. The result is cached after the first call.

        Returns
        -------
        PointHomogeneousSymbolic
            Point with leading coordinate equal to 1.

        Raises
        ------
        ValueError
            If the point is at infinity (``w`` simplifies to zero).
        """
        if self._normalized is None:
            if self.is_at_infinity:
                length = sympy.sqrt(sum(v ** 2 for v in self.coordinates))
                self._normalized = self.__class__(
                    [sympy.simplify(v / length) for v in self.coordinates]
                )
            else:
                w = self.coordinates[0]
                self._normalized = self.__class__(
                    [sympy.simplify(v / w) for v in self.coordinates]
                )
        return self._normalized

    def point2matrix(self) -> sympy.Matrix:
        """
        Convert to a homogeneous SE(3) matrix with identity rotation.

        Returns a SymPy Matrix rather than a NumPy array.

        Returns
        -------
        sympy.Matrix
            4×4 symbolic matrix.

        Raises
        ------
        ValueError
            If the coordinate length is not 3, 4, 12, or 13.
        """
        norm = self.normalize().array()
        mat = sympy.eye(4)

        if len(norm) == 3:
            mat[0, 0] = norm[0]
            mat[1, 0] = norm[1]
            mat[2, 0] = norm[2]
        elif len(norm) == 4:
            mat[0, 0] = norm[0]
            mat[1, 0] = norm[1]
            mat[2, 0] = norm[2]
            mat[3, 0] = norm[3]
        elif len(norm) == 12:
            for i, idx in enumerate(range(0, 12, 3)):
                mat[1, i] = norm[idx]
                mat[2, i] = norm[idx + 1]
                mat[3, i] = norm[idx + 2]
        elif len(norm) == 13:
            for i, idx in enumerate(range(1, 13, 3)):
                mat[1, i] = norm[idx]
                mat[2, i] = norm[idx + 1]
                mat[3, i] = norm[idx + 2]
        else:
            raise ValueError(
                "PointHomogeneousSymbolic.point2matrix: coordinate length must be 3, 4, 12, or 13"
            )
        return mat

    def eval(self, subs: dict) -> "PointHomogeneousSymbolic":
        """
        Evaluate the point by substituting symbols with values.

        Parameters
        ----------
        subs : dict
            Mapping of SymPy symbols to values.

        Returns
        -------
        PointHomogeneousSymbolic
            New point with substitutions applied.

        Examples
        --------
        .. code-block:: python

            import rational_linkages
            rational_linkages.set_backend("sympy")

            from rational_linkages import PointHomogeneous
            from sympy import symbols

            t = symbols("t")
            p = PointHomogeneous([1, t, 2*t, 0])
            p_eval = p.eval({t: 3})
            print(p_eval)   # PointHomogeneousSymbolic([1, 3, 6, 0])

            rational_linkages.set_backend("numpy")
        """
        return self.__class__([v.subs(subs) for v in self.coordinates])

    def evalf(self) -> numpy.ndarray:
        """
        Evaluate the point to floating-point numbers.

        Returns
        -------
        numpy.ndarray
        """
        return numpy.array([v.evalf() for v in self.coordinates], dtype=numpy.float64)

    def evalf_euclidean(self) -> numpy.ndarray:
        """
        Evaluate the Euclidean coordinates to floating-point numbers.

        Returns
        -------
        numpy.ndarray
        """
        evaluated = self.evalf()
        return evaluated[1:] / evaluated[0]