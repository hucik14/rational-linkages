from typing import Optional, Sequence, Union
from warnings import warn

import numpy
import sympy

from .NormalizedLine import NormalizedLine


class NormalizedLineSymbolic(NormalizedLine):
    """
    Symbolic Plücker line backed by SymPy expressions.

    Subclass of :class:`~rational_linkages.NormalizedLine` for algebraic
    computation. Typically not instantiated directly — when the global backend
    is set to ``"sympy"`` via :func:`~rational_linkages.set_backend`,
    :class:`~rational_linkages.NormalizedLine` transparently returns instances
    of this class via its ``__new__`` factory.

    Parameters
    ----------
    unit_screw :
        6-vector of Plücker coordinates ``[l1, l2, l3, m1, m2, m3]`` as
        SymPy expressions or plain numbers. If ``None``, the Z-axis through
        the origin is constructed.

    Attributes
    ----------
    direction : numpy.ndarray
        Object-dtype array of SymPy expressions ``[l1, l2, l3]``.
    moment : numpy.ndarray
        Object-dtype array of SymPy expressions ``[m1, m2, m3]``.
    screw : numpy.ndarray
        Object-dtype 6-vector ``[direction | moment]``.

    Examples
    --------
    .. code-block:: python

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import NormalizedLine
        from sympy import symbols

        l1, l2, l3, m1, m2, m3 = symbols("l1 l2 l3 m1 m2 m3", real=True)
        line = NormalizedLine([l1, l2, l3, m1, m2, m3])
        print(line.direction)   # [l1, l2, l3]

        rational_linkages.set_backend("numpy")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, unit_screw: Optional[Sequence] = None):
        # Bypass NormalizedLine.__init__ (which assumes float64) and redo
        # initialization with symbolic-aware helpers.
        self.direction, self.moment = self._initialize_components(unit_screw)
        self.screw = numpy.concatenate((self.direction, self.moment))

    @classmethod
    def from_direction_and_point(cls, direction, point) -> "NormalizedLine":
        """
        Construct a ``NormalizedLine`` from the given ``direction`` and ``point``.

        Parameters
        ----------
        direction :
            3-vector of SymPy expressions or plain numbers representing the line's
            direction. Should be unit-length for correct behavior, but this is not
            enforced.
        point :
            3-vector of SymPy expressions or plain numbers representing a point
            on the line.

        Returns
        -------
        NormalizedLine

        """
        direction = numpy.array([sympy.sympify(v) for v in direction], dtype=object)
        point = numpy.array([sympy.sympify(v) for v in point], dtype=object)
        moment = numpy.array([
            -direction[1] * point[2] + direction[2] * point[1],
            -direction[2] * point[0] + direction[0] * point[2],
            -direction[0] * point[1] + direction[1] * point[0],
        ], dtype=object)
        return cls(numpy.concatenate((direction, moment)))

    @classmethod
    def from_direction_and_moment(cls, direction, moment) -> "NormalizedLine":
        """
        Construct a ``NormalizedLine`` from the given ``direction`` and ``moment``.

        Parameters
        ----------
        direction :
            3-vector of SymPy expressions or plain numbers representing the line's
            direction. Should be unit-length for correct behavior, but this is not
            enforced.
        moment :
            3-vector of SymPy expressions or plain numbers representing the line's
            moment.

        Returns
        -------
        NormalizedLine

        """
        direction = numpy.array([sympy.sympify(v) for v in direction], dtype=object)
        moment = numpy.array([sympy.sympify(v) for v in moment], dtype=object)
        return cls(numpy.concatenate((direction, moment)))

    @classmethod
    def from_two_points(cls,
                        pt0: Union["PointHomogeneous", Sequence[float]],
                        pt1: Union["PointHomogeneous", Sequence[float]]) -> "NormalizedLine":
        """
        Construct a ``NormalizedLine`` from the given ``pt0`` and ``pt1``.

        Parameters
        ----------
        pt0 :
            A :class:`~rational_linkages.PointHomogeneous` or a 3-vector of SymPy expressions
            or numbers representing a point on the line.
        pt1 :
            A :class:`~rational_linkages.PointHomogeneous` or a 3-vector of SymPy expressions
            or numbers representing a different point on the line.

        Returns
        -------
        NormalizedLine

        Raises
        ------
        ValueError
            If the two points are identical (i.e. the direction vector has zero
            norm).
        """
        from .PointHomogeneous import PointHomogeneous
        if isinstance(pt0, PointHomogeneous):
            pt0 = pt0.normalized_euclidean()
        else:
            pt0 = numpy.array([sympy.sympify(v) for v in pt0], dtype=object)
        if isinstance(pt1, PointHomogeneous):
            pt1 = pt1.normalized_euclidean()
        else:
            pt1 = numpy.array([sympy.sympify(v) for v in pt1], dtype=object)

        direction = pt1 - pt0
        if sum(v**2 for v in direction) == sympy.Integer(0):
            raise ValueError("Cannot construct a line from two identical points.")

        moment = numpy.array([
            -direction[1] * pt0[2] + direction[2] * pt0[1],
            -direction[2] * pt0[0] + direction[0] * pt0[2],
            -direction[0] * pt0[1] + direction[1] * pt0[0],
        ], dtype=object)
        return cls(numpy.concatenate((direction, moment)))

    def _initialize_components(
        self, unit_screw: Optional[Sequence]
    ) -> tuple:
        """
        Parse a 6-vector into symbolic direction and moment components.

        No normalization is applied — the caller is responsible for providing
        a unit-length direction when that is required.

        Parameters
        ----------
        unit_screw :
            6-vector of SymPy expressions, or ``None`` for the Z-axis.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(direction, moment)`` each an object-dtype 3-vector.

        Warns
        -----
        UserWarning
            If the direction vector simplifies to zero.
        """
        if unit_screw is None:
            direction = numpy.array(
                [sympy.Integer(0), sympy.Integer(0), sympy.Integer(1)],
                dtype=object,
            )
            moment = numpy.array(
                [sympy.Integer(0), sympy.Integer(0), sympy.Integer(0)],
                dtype=object,
            )
            return direction, moment

        arr = numpy.array([sympy.sympify(v) for v in unit_screw], dtype=object)
        direction = arr[0:3]
        moment = arr[3:6]

        all_numeric = all(
            isinstance(v, (int, float)) or
            (hasattr(v, 'free_symbols') and len(v.free_symbols) == 0)
            for v in direction
        )
        if all_numeric:
            norm_sq = sympy.simplify(sum(v**2 for v in direction))
            if norm_sq == sympy.Integer(0):
                warn(
                    "NormalizedLineSymbolic: direction vector has zero norm.",
                    UserWarning,
                    stacklevel=3,
                )
            elif norm_sq != sympy.Integer(1):
                norm = sympy.sqrt(norm_sq)
                direction = numpy.array(
                    [sympy.simplify(v / norm) for v in direction], dtype=object
                )
                moment = numpy.array(
                    [sympy.simplify(v / norm) for v in moment], dtype=object
                )

        return direction, moment

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        entries = ", ".join(str(v) for v in self.screw)
        return f"Ln([{entries}])"

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other: "NormalizedLine") -> bool:
        """
        Coefficient-wise equality via symbolic simplification.

        Parameters
        ----------
        other :
            Line to compare against.

        Returns
        -------
        bool
            ``True`` if all coordinate differences simplify to zero.
        """
        return all(
            sympy.simplify(a - b) == sympy.Integer(0)
            for a, b in zip(self.screw, other.screw)
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return screw coordinates as an object-dtype NumPy array.

        Returns
        -------
        numpy.ndarray
            Object-dtype 6-vector of SymPy expressions.
        """
        return numpy.array(self.screw, dtype=object)

    def line2dq_array(self) -> numpy.ndarray:
        """
        Embed the line into dual quaternion space as an 8-vector.

        Maps ``[l, m]`` → ``[0, l1, l2, l3, 0, -m1, -m2, -m3]``.

        Returns
        -------
        numpy.ndarray
            Object-dtype 8-vector of SymPy expressions.
        """
        return numpy.array([
            sympy.Integer(0),
            self.direction[0],
            self.direction[1],
            self.direction[2],
            sympy.Integer(0),
            -self.moment[0],
            -self.moment[1],
            -self.moment[2],
        ], dtype=object)

    def contains_point(
        self,
        point: Union["PointHomogeneous", numpy.ndarray, Sequence],
    ) -> bool:
        """
        Return ``True`` if *point* lies on the line (symbolic check).

        Parameters
        ----------
        point :
            A :class:`~rational_linkages.PointHomogeneous` or a 3-vector of SymPy
            expressions / numbers.

        Returns
        -------
        bool
        """
        from .PointHomogeneous import PointHomogeneous  # lazy import

        if isinstance(point, PointHomogeneous):
            pt = point.normalize().array()[1:]
        else:
            pt = numpy.array([sympy.sympify(v) for v in point], dtype=object)

        cross = numpy.array([
            pt[1] * self.direction[2] - pt[2] * self.direction[1],
            pt[2] * self.direction[0] - pt[0] * self.direction[2],
            pt[0] * self.direction[1] - pt[1] * self.direction[0],
        ], dtype=object)

        return all(
            sympy.simplify(cross[i] - self.moment[i]) == sympy.Integer(0)
            for i in range(3)
        )

    def eval(self, subs: dict) -> "NormalizedLineSymbolic":
        """
        Evaluate the line by substituting symbols with values.

        Parameters
        ----------
        subs : dict
            Mapping of SymPy symbols to values.

        Returns
        -------
        NormalizedLineSymbolic
            New line with substitutions applied.

        Examples
        --------
        .. code-block:: python

            import rational_linkages
            rational_linkages.set_backend("sympy")

            from rational_linkages import NormalizedLine
            from sympy import symbols

            t = symbols("t")
            line = NormalizedLine([0, 0, 1, t, 0, 0])
            line_eval = line.eval({t: 3})
            print(line_eval)

            rational_linkages.set_backend("numpy")
        """
        return self.__class__([v.subs(subs) for v in self.screw])

    def evaluate(self, param: float):
        """
        Evaluates the line by substituting single value of t.

        Parameters
        ----------
        param : float
            Parameter to evaluate.

        Returns
        -------
        NormalizedLineSymbolic
            New line with substitutions applied.
        """
        t = sympy.symbols('t')
        return self.__class__([v.subs({t: param}) for v in self.screw])

    def evalf(self):
        """
        Replace rational numbers by numerical ones.

        Returns
        -------
        NormalizedLine
            Return normalized line with rational numbers replaced by numerical ones.
        """
        return NormalizedLine(
            numpy.array([v.evalf() for v in self.coordinates], dtype=numpy.float64))