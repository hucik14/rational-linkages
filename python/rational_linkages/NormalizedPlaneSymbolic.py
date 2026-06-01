from typing import Sequence

import numpy
import sympy

from .NormalizedPlane import NormalizedPlane


class NormalizedPlaneSymbolic(NormalizedPlane):
    """
    Symbolic plane backed by SymPy expressions.

    Subclass of :class:`.NormalizedPlane` for algebraic
    computation. Typically not instantiated directly — when the global backend
    is set to ``"sympy"`` via :func:`.set_backend`,
    :class:`.NormalizedPlane` transparently returns instances
    of this class via its ``__new__`` factory.

    Parameters
    ----------
    normal :
        3-vector normal as SymPy expressions or plain numbers.
    point :
        3-vector point on the plane as SymPy expressions or plain numbers.

    Attributes
    ----------
    normal : numpy.ndarray
        Object-dtype unit normal ``[n1, n2, n3]`` of SymPy expressions.
    point : numpy.ndarray
        Object-dtype array ``[x, y, z]`` of SymPy expressions.
    oriented_distance : sympy.Expr
        Signed distance ``d = n · (-point)`` as a SymPy expression.
    coordinates : numpy.ndarray
        Object-dtype 4-vector ``[d, n1, n2, n3]``.

    Examples
    --------
    .. code-block:: python

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import NormalizedPlane
        from sympy import symbols

        a, b, c, d = symbols("a b c d", real=True)
        plane = NormalizedPlane([a, b, c], [d, 0, 0])
        print(plane.oriented_distance)

        rational_linkages.set_backend("numpy")

    .. clear-namespace::
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, normal: Sequence, point: Sequence):
        # Bypass NormalizedPlane.__init__ (which assumes float64) and redo
        # initialization with symbolic-aware helpers.
        self.point = numpy.array(
            [sympy.sympify(v) for v in point], dtype=object
        )

        n = numpy.array([sympy.sympify(v) for v in normal], dtype=object)
        all_numeric = all(
            isinstance(v, (int, float)) or
            (hasattr(v, 'free_symbols') and len(v.free_symbols) == 0)
            for v in n
        )
        if all_numeric:
            norm_sq = sympy.simplify(sum(v**2 for v in n))
            if norm_sq != sympy.Integer(1):
                norm = sympy.sqrt(norm_sq)
                n = numpy.array(
                    [sympy.simplify(v / norm) for v in n], dtype=object
                )
        self.normal = n
        self.oriented_distance = sympy.simplify(
            sum(-self.normal[i] * self.point[i] for i in range(3))
        )
        self.coordinates = numpy.array(
            [self.oriented_distance, self.normal[0], self.normal[1], self.normal[2]],
            dtype=object,
        )
        self._reflection_matrix = None
        self._reflection_tr = None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        entries = ", ".join(str(v) for v in self.coordinates)
        return f"Plane([{entries}])"

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other: "NormalizedPlane") -> bool:
        """
        Coefficient-wise equality via symbolic simplification.

        Parameters
        ----------
        other :
            Plane to compare against.

        Returns
        -------
        bool
            ``True`` if all coordinate differences simplify to zero.
        """
        return all(
            sympy.simplify(a - b) == sympy.Integer(0)
            for a, b in zip(self.coordinates, other.coordinates)
        )

    # ------------------------------------------------------------------
    # Cached properties (symbolic)
    # ------------------------------------------------------------------

    @property
    def reflection_matrix(self) -> numpy.ndarray:
        """
        3×3 symbolic Householder reflection matrix about the plane's normal.

        Cached after first access.

        Returns
        -------
        numpy.ndarray
            Object-dtype ``I - 2 * n ⊗ n``.
        """
        if self._reflection_matrix is None:
            eye = sympy.eye(3)
            outer = numpy.array(
                [[sympy.expand(self.normal[i] * self.normal[j])
                  for j in range(3)]
                 for i in range(3)],
                dtype=object,
            )
            self._reflection_matrix = numpy.array(
                [[sympy.simplify(eye[i, j] - 2 * outer[i, j])
                  for j in range(3)]
                 for i in range(3)],
                dtype=object,
            )
        return self._reflection_matrix

    @property
    def reflection_tr(self) -> numpy.ndarray:
        """
        4×4 symbolic homogeneous reflection transformation matrix.

        Cached after first access.

        Returns
        -------
        numpy.ndarray
            Object-dtype 4×4 array.
        """
        if self._reflection_tr is None:
            mat = numpy.array(sympy.eye(4).tolist(), dtype=object)
            ref = self.reflection_matrix
            for i in range(3):
                for j in range(3):
                    mat[i + 1, j + 1] = ref[i, j]
                mat[i + 1, 0] = sympy.simplify(
                    -2 * self.oriented_distance * self.normal[i]
                )
            self._reflection_tr = mat
        return self._reflection_tr

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return plane coordinates as an object-dtype NumPy array.

        Returns
        -------
        numpy.ndarray
            Object-dtype 4-vector ``[d, n1, n2, n3]`` of SymPy expressions.
        """
        return numpy.array(self.coordinates, dtype=object)

    def plane2dq_array(self) -> numpy.ndarray:
        """
        Embed the plane into dual quaternion space as an 8-vector.

        Maps ``[d, n]`` → ``[0, n1, n2, n3, d, 0, 0, 0]``.

        Returns
        -------
        numpy.ndarray
            Object-dtype 8-vector of SymPy expressions.
        """
        return numpy.array([
            sympy.Integer(0),
            self.normal[0],
            self.normal[1],
            self.normal[2],
            self.oriented_distance,
            sympy.Integer(0),
            sympy.Integer(0),
            sympy.Integer(0),
        ], dtype=object)

    def eval(self, subs: dict) -> "NormalizedPlaneSymbolic":
        """
        Evaluate the plane by substituting symbols with values.

        Parameters
        ----------
        subs : dict
            Mapping of SymPy symbols to values.

        Returns
        -------
        NormalizedPlaneSymbolic
            New plane with substitutions applied.

        Examples
        --------
        .. code-block:: python

            import rational_linkages
            rational_linkages.set_backend("sympy")

            from rational_linkages import NormalizedPlane
            from sympy import symbols

            t = symbols("t")
            plane = NormalizedPlane([0, 0, 1], [0, 0, t])
            plane_eval = plane.eval({t: 5})
            print(plane_eval)

            rational_linkages.set_backend("numpy")

        .. clear-namespace::
        """
        normal_eval = [v.subs(subs) for v in self.normal]
        point_eval = [v.subs(subs) for v in self.point]
        return self.__class__(normal_eval, point_eval)

    def evalf(self):
        """
        Replace rational numbers by numerical ones.

        Returns
        -------
        numpy.ndarray
            Float NumPy array of previous rational numbers.
        """
        from rational_linkages.utils import evaluate_numerically  # lazy import
        return evaluate_numerically(self)