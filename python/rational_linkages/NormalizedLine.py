from typing import Optional, Sequence, Union
from warnings import warn

import numpy

from .backend import is_symbolic


class NormalizedLine:
    """
    Line in 3D space represented by Plücker coordinates (unit screw axis).

    Plücker coordinates ``[l1, l2, l3, m1, m2, m3]`` encode a line as a
    direction vector ``l`` (unit-length) and a moment vector ``m``, satisfying
    the Plücker condition ``l · m = 0``.

    By default, all computation is performed with NumPy (``float64``). When
    the global backend is set to ``"sympy"`` via
    :func:`~rational_linkages.set_backend`, construction transparently returns a
    :class:`~rational_linkages.NormalizedLineSymbolic` instance instead.

    Parameters
    ----------
    unit_screw :
        6-vector of Plücker coordinates ``[l1, l2, l3, m1, m2, m3]``. If
        ``None``, the Z-axis through the origin ``[0, 0, 1, 0, 0, 0]`` is
        constructed.

    Attributes
    ----------
    direction : numpy.ndarray
        Unit direction vector ``[l1, l2, l3]``.
    moment : numpy.ndarray
        Moment vector ``[m1, m2, m3]``.
    screw : numpy.ndarray
        Concatenation ``[direction | moment]``, length 6.

    Examples
    --------
    .. code-block:: python

        from rational_linkages import NormalizedLine

        line = NormalizedLine([1, 0, 0, 0, -2, 1])
        default = NormalizedLine()   # Z-axis through origin

    .. clear-namespace::

    .. code-block:: python

        from rational_linkages import NormalizedLine, PointHomogeneous

        p0 = PointHomogeneous([1, 1, 1, 1])
        p1 = PointHomogeneous([1, 3, 1, 1])
        line = NormalizedLine.from_two_points(p0, p1)

    .. clear-namespace::

    .. code-block:: python

        # Symbolic backend

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import NormalizedLine
        from sympy import symbols

        l1, l2, l3, m1, m2, m3 = symbols("l1 l2 l3 m1 m2 m3", real=True)
        line = NormalizedLine([l1, l2, l3, m1, m2, m3])

        rational_linkages.set_backend("numpy")

    .. clear-namespace::
    """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def __new__(cls, unit_screw=None):
        """
        Intercept construction and return a
        :class:`~rational_linkages.NormalizedLineSymbolic` when the global
        backend is ``"sympy"``.

        Only applied when ``cls`` is exactly ``NormalizedLine``; subclass
        constructors are never redirected, preventing infinite recursion.

        Parameters
        ----------
        unit_screw :
            Forwarded unchanged to ``__init__``.

        Returns
        -------
        NormalizedLine or NormalizedLineSymbolic
        """
        if cls is NormalizedLine:
            symbolic = is_symbolic() or (
                    unit_screw is not None
                    and any(hasattr(c, 'free_symbols') for c in unit_screw)
            )
            if symbolic:
                from .NormalizedLineSymbolic import NormalizedLineSymbolic
                return object.__new__(NormalizedLineSymbolic)
        return object.__new__(cls)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, unit_screw: Optional[Sequence[float]] = None):
        self.direction, self.moment = self._initialize_components(unit_screw)
        self.screw = numpy.concatenate((self.direction, self.moment))

    def _initialize_components(
        self, unit_screw: Optional[Sequence[float]]
    ) -> tuple:
        """
        Parse and normalize a 6-vector into direction and moment components.

        The direction vector is normalized to unit length if it is not already.
        The moment vector is scaled by the same factor.

        Parameters
        ----------
        unit_screw :
            6-vector ``[l1, l2, l3, m1, m2, m3]``, or ``None`` for the
            Z-axis through the origin.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(direction, moment)`` each a ``float64`` 3-vector.

        Warns
        -----
        UserWarning
            If the direction vector has zero norm.
        """
        if unit_screw is None:
            return (
                numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64),
                numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64),
            )

        try:
            arr = numpy.asarray(unit_screw, dtype=numpy.float64)
        except (TypeError, ValueError):
            arr = numpy.array(unit_screw, dtype=object)

        direction = arr[0:3]
        moment = arr[3:6]

        norm = numpy.linalg.norm(direction)
        if round(float(norm), 6) == 1.0:
            pass  # already unit length
        elif float(norm) > 1e-10:
            direction = direction / norm
            moment = moment / norm
        else:
            warn(
                "NormalizedLine: direction vector has zero norm.",
                UserWarning,
                stacklevel=3,
            )

        return direction, moment

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_two_points(
        cls,
        pt0: Union["PointHomogeneous", Sequence[float]],
        pt1: Union["PointHomogeneous", Sequence[float]],
    ) -> "NormalizedLine":
        """
        Construct a NormalizedLine from two points.

        Parameters
        ----------
        pt0 :
            First point — either a :class:`~rational_linkages.PointHomogeneous` or a
            3-vector ``[x, y, z]``.
        pt1 :
            Second point — either a :class:`~rational_linkages.PointHomogeneous` or a
            3-vector ``[x, y, z]``.

        Returns
        -------
        NormalizedLine

        Raises
        ------
        ValueError
            If the two points are identical.
        """
        if cls is NormalizedLine and is_symbolic():
            from .NormalizedLineSymbolic import NormalizedLineSymbolic
            return NormalizedLineSymbolic.from_two_points(pt0, pt1)
        from .PointHomogeneous import PointHomogeneous  # lazy import

        if isinstance(pt0, PointHomogeneous):
            pt0 = pt0.normalized_euclidean()
        else:
            pt0 = numpy.asarray(pt0, dtype=numpy.float64)

        if isinstance(pt1, PointHomogeneous):
            pt1 = pt1.normalized_euclidean()
        else:
            pt1 = numpy.asarray(pt1, dtype=numpy.float64)

        if numpy.allclose(pt0, pt1, rtol=1e-7):
            raise ValueError("NormalizedLine.from_two_points: points are identical.")

        direction = pt1 - pt0
        moment = numpy.cross(-direction, pt0)
        return cls(numpy.concatenate((direction, moment)))

    @classmethod
    def from_direction_and_point(
        cls,
        direction: Sequence[float],
        point: Sequence[float],
    ) -> "NormalizedLine":
        """
        Construct a NormalizedLine from a direction vector and a point on the line.

        Parameters
        ----------
        direction :
            3-vector direction ``[dx, dy, dz]``.
        point :
            3-vector point on the line ``[x, y, z]``.

        Returns
        -------
        NormalizedLine
        """
        if cls is NormalizedLine and is_symbolic():
            from .NormalizedLineSymbolic import NormalizedLineSymbolic
            return NormalizedLineSymbolic.from_direction_and_point(direction, point)
        try:
            direction = numpy.asarray(direction, dtype=numpy.float64)
            point = numpy.asarray(point, dtype=numpy.float64)
        except (TypeError, ValueError):
            direction = numpy.asarray(direction, dtype=object)
            point = numpy.asarray(point, dtype=object)
        moment = numpy.cross(-direction, point)
        return cls(numpy.concatenate((direction, moment)))

    @classmethod
    def from_direction_and_moment(
        cls,
        direction: Sequence[float],
        moment: Sequence[float],
    ) -> "NormalizedLine":
        """
        Construct a NormalizedLine directly from direction and moment vectors.

        Parameters
        ----------
        direction :
            3-vector direction ``[l1, l2, l3]``.
        moment :
            3-vector moment ``[m1, m2, m3]``.

        Returns
        -------
        NormalizedLine
        """
        if cls is NormalizedLine and is_symbolic():
            from .NormalizedLineSymbolic import NormalizedLineSymbolic
            return NormalizedLineSymbolic.from_direction_and_moment(direction, moment)
        try:
            direction = numpy.asarray(direction, dtype=numpy.float64)
            moment = numpy.asarray(moment, dtype=numpy.float64)
        except (TypeError, ValueError):
            direction = numpy.asarray(direction, dtype=object)
            moment = numpy.asarray(moment, dtype=object)
        return cls(numpy.concatenate((direction, moment)))

    @classmethod
    def from_dual_quaternion(cls, dq: "DualQuaternion") -> "NormalizedLine":
        """
        Construct a NormalizedLine from a DualQuaternion.

        Parameters
        ----------
        dq :
            Source dual quaternion representing a line.

        Returns
        -------
        NormalizedLine
        """
        return cls(dq.dq2screw())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coordinates(self):
        """
        Return the coordinates of line.

        Returns
        -------
        numpy.ndarray
            6-vector of Pluecker line coordinates (unit screw).
        """
        return self.screw.copy()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """Return the screw coordinate at *idx*."""
        return self.screw[idx]

    def __len__(self) -> int:
        """Number of screw coordinates, always 6."""
        return 6

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        s = numpy.array2string(
            self.screw,
            precision=10,
            suppress_small=True,
            separator=", ",
            max_line_width=100000,
        )
        return f"{self.__class__.__qualname__}({s})"

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other: "NormalizedLine") -> bool:
        """
        Coefficient-wise equality.

        Parameters
        ----------
        other :
            Line to compare against.

        Returns
        -------
        bool
        """
        return numpy.array_equal(self.screw, other.screw)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return screw coordinates as a NumPy array.

        Returns
        -------
        numpy.ndarray
            6-vector ``[direction | moment]``.
        """
        return self.screw.copy()

    def line2dq_array(self) -> numpy.ndarray:
        """
        Embed the line into dual quaternion space as an 8-vector.

        Maps ``[l, m]`` → ``[0, l1, l2, l3, 0, -m1, -m2, -m3]``.

        Returns
        -------
        numpy.ndarray
            8-vector.
        """
        return numpy.array([
            0,
            self.direction[0],
            self.direction[1],
            self.direction[2],
            0,
            -self.moment[0],
            -self.moment[1],
            -self.moment[2],
        ])

    def point_on_line(self, t: float = 0.0) -> numpy.ndarray:
        """
        Return a point on the line at parameter *t*.

        The principal point (``t = 0``) is ``direction × moment``.

        Parameters
        ----------
        t :
            Line parameter. Default ``0.0``.

        Returns
        -------
        numpy.ndarray
            3-vector ``[x, y, z]``.
        """
        principal_point = numpy.cross(self.direction, self.moment)
        return principal_point + t * self.direction

    def point_homogeneous(self) -> numpy.ndarray:
        """
        Return a homogeneous point on the Plücker line.

        Selects the row of the point quadric matrix with the largest absolute
        value in the first column.

        Returns
        -------
        numpy.ndarray
            4-vector ``[w, x, y, z]``.
        """
        l, m = self.direction, self.moment
        pt_quadric = numpy.array([
            [-l[0],    0,    m[2], -m[1]],
            [-l[1], -m[2],     0,  m[0]],
            [-l[2],  m[1], -m[0],     0],
        ])
        max_index = numpy.abs(pt_quadric[:, 0]).argmax()
        return pt_quadric[max_index, :]

    def get_point_param(
        self, point: Union[numpy.ndarray, Sequence[float]]
    ) -> float:
        """
        Return the line parameter *t* for a given point on the line.

        Parameters
        ----------
        point :
            3-vector ``[x, y, z]`` assumed to lie on the line.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the direction vector is zero (degenerate line).
        """
        point = numpy.asarray(point, dtype=numpy.float64)
        vec = point - self.point_on_line()
        for i in range(3):
            if self.direction[i] != 0.0:
                return vec[i] / self.direction[i]
        raise ValueError("NormalizedLine.get_point_param: direction vector is zero.")

    def contains_point(
        self,
        point: Union["PointHomogeneous", numpy.ndarray, Sequence[float]],
    ) -> bool:
        """
        Return ``True`` if *point* lies on the line.

        A new moment is computed from the point and direction; if it matches
        the stored moment the point is on the line.

        Parameters
        ----------
        point :
            A :class:`~rational_linkages.PointHomogeneous` or a 3-vector ``[x, y, z]``.

        Returns
        -------
        bool
        """
        from .PointHomogeneous import PointHomogeneous  # lazy import

        if isinstance(point, PointHomogeneous):
            point = point.normalized_euclidean()
        else:
            point = numpy.asarray(point, dtype=numpy.float64)

        return numpy.allclose(numpy.cross(point, self.direction), self.moment)

    def common_perpendicular_to_other_line(
        self, other: "NormalizedLine"
    ) -> tuple:
        """
        Compute the common perpendicular between this line and *other*.

        Returns the two foot-points, the distance, and the cosine of the
        angle between the lines. Falls back to the principal-point distance
        when the lines are parallel.

        Parameters
        ----------
        other :
            The second line.

        Returns
        -------
        tuple[list[numpy.ndarray], float, float]
            ``(points, distance, cos_angle)`` where ``points`` is a list of
            two 3-vectors.
        """
        points = [numpy.zeros(3), numpy.zeros(3)]
        cross = numpy.cross(self.direction, other.direction)
        cross_norm = numpy.linalg.norm(cross)

        if not numpy.isclose(cross_norm, 0.0, atol=1e-5):
            num0 = (
                numpy.cross(-self.moment, numpy.cross(other.direction, cross))
                + numpy.dot(self.direction, numpy.dot(other.moment, cross))
            )
            points[0] = num0 / cross_norm**2

            num1 = (
                numpy.cross(other.moment, numpy.cross(self.direction, cross))
                - numpy.dot(other.direction, numpy.dot(self.moment, cross))
            )
            points[1] = num1 / cross_norm**2

            distance = numpy.linalg.norm(points[0] - points[1])
            cos_angle = numpy.dot(self.direction, other.direction) / (
                numpy.linalg.norm(self.direction) * numpy.linalg.norm(other.direction)
            )
        else:
            points[0] = numpy.cross(self.direction, self.moment)
            points[1] = numpy.cross(other.direction, other.moment)
            distance = numpy.linalg.norm(points[0] - points[1])
            cos_angle = 1.0

        return points, distance, cos_angle

    def intersection_with_plane(self, plane: "NormalizedPlane") -> numpy.ndarray:
        """
        Return the homogeneous intersection point of the line with *plane*.

        See Pottmann et al., *Computational Line Geometry*, 2001, §3.4.2,
        eq. 2.17.

        Parameters
        ----------
        plane :
            The plane to intersect with.

        Returns
        -------
        numpy.ndarray
            4-vector ``[w, x, y, z]`` in homogeneous coordinates.
        """
        p0 = numpy.dot(plane.normal, self.direction)
        p_vec = (
            -plane.oriented_distance * self.direction
            + numpy.cross(plane.normal, self.moment)
        )
        return numpy.concatenate(([p0], p_vec))

    def eval(self, params: dict):
        """
        Placeholder for NormalizedLineSymbolic.eval(). Evaluates line with given parameters.

        Parameters
        ----------
        params : dict
            Dictionary of Sympy parameters and values to be evaluated for.

        Returns
        -------
        NormalizedLine
            Self.
        """
        return self

    def evaluate(self, param: float):
        """
        Placeholder for NormalizedLineSymbolic.eval(). Evaluates line with a given parameter.

        Returns
        -------
        NormalizedLine
            Self.
        """
        return self

    def evalf(self):
        """
        Placeholder for NormalizedLineSymbolic.evalf(). Returns numerical values.

        Returns
        -------
        NormalizedLine
            Self.
        """
        return self

    # ------------------------------------------------------------------
    # Conversion / plotting
    # ------------------------------------------------------------------

    def get_plot_data(self, interval: tuple) -> numpy.ndarray:
        """
        Return start-point and direction vector for 3-D plotting.

        Parameters
        ----------
        interval :
            ``(t_start, t_end)`` parameter range.

        Returns
        -------
        numpy.ndarray
            6-vector ``[x0, y0, z0, dx, dy, dz]``.
        """
        p0 = self.point_on_line(interval[0])
        p1 = self.point_on_line(interval[1])
        return numpy.concatenate((p0, p1 - p0))