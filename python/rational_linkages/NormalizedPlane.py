from typing import Optional, Sequence, Union

import numpy

from .backend import is_symbolic

PointHomogeneous = "PointHomogeneous"
NormalizedLine = "NormalizedLine"


class NormalizedPlane:
    """
    Plane in 3D space represented by a unit normal and an oriented distance.

    The plane is stored as ``[d, n1, n2, n3]`` where ``n`` is the unit normal
    and ``d = n · (-point)`` is the signed distance from the origin.

    By default, all computation is performed with NumPy (``float64``). When
    the global backend is set to ``"sympy"`` via
    :func:`.set_backend`, construction transparently returns a
    :class:`.NormalizedPlaneSymbolic` instance instead.

    Parameters
    ----------
    normal :
        3-vector normal to the plane. Normalized to unit length on construction.
    point :
        3-vector of any point lying on the plane.

    Attributes
    ----------
    normal : numpy.ndarray
        Unit normal vector ``[n1, n2, n3]``.
    point : numpy.ndarray
        A point on the plane ``[x, y, z]``.
    oriented_distance : float
        Signed distance from the origin, ``d = n · (-point)``.
    coordinates : numpy.ndarray
        4-vector ``[d, n1, n2, n3]``.

    Examples
    --------
    .. code-block:: python

        from rational_linkages import NormalizedPlane

        plane = NormalizedPlane([0, 0, 1], [0, 0, 5])   # z = 5

    .. clear-namespace::

    .. code-block:: python

        # Symbolic backend

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import NormalizedPlane
        from sympy import symbols

        a, b, c, d = symbols("a b c d", real=True)
        plane = NormalizedPlane([a, b, c], [d, 0, 0])

        rational_linkages.set_backend("numpy")

    .. clear-namespace::
    """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def __new__(cls, normal=None, point=None):
        """
        Intercept construction and return a
        :class:`.NormalizedPlaneSymbolic` when the global
        backend is ``"sympy"``.

        Only applied when ``cls`` is exactly ``NormalizedPlane``; subclass
        constructors are never redirected, preventing infinite recursion.

        Parameters
        ----------
        normal :
            Forwarded unchanged to ``__init__``.
        point :
            Forwarded unchanged to ``__init__``.

        Returns
        -------
        NormalizedPlane or NormalizedPlaneSymbolic
        """
        if cls is NormalizedPlane:
            symbolic = is_symbolic() or (
                    normal is not None
                    and any(hasattr(c, 'free_symbols') for c in normal)
            ) or (
                    point is not None
                    and any(hasattr(c, 'free_symbols') for c in point)
            )
            if symbolic:
                from .NormalizedPlaneSymbolic import NormalizedPlaneSymbolic  # lazy import
                return object.__new__(NormalizedPlaneSymbolic)
        return object.__new__(cls)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        normal: Sequence[float],
        point: Sequence[float],
    ):
        self.point = numpy.asarray(point, dtype=numpy.float64)
        n = numpy.asarray(normal, dtype=numpy.float64)
        self.normal = n / numpy.linalg.norm(n)
        self.oriented_distance = float(numpy.dot(self.normal, -self.point))
        self.coordinates = numpy.concatenate(
            ([self.oriented_distance], self.normal)
        )
        self._reflection_matrix: Optional[numpy.ndarray] = None
        self._reflection_tr: Optional[numpy.ndarray] = None

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_two_points_as_bisector(
        cls,
        point1: PointHomogeneous,
        point2: PointHomogeneous,
    ) -> "NormalizedPlane":
        """
        Construct the bisector plane of two points.

        The plane's foot-point is the midpoint of the two points and its
        normal is the vector connecting them.

        Parameters
        ----------
        point1 :
            First :class:`.PointHomogeneous`.
        point2 :
            Second :class:`.PointHomogeneous`.

        Returns
        -------
        NormalizedPlane
        """
        p1 = point1.normalized_euclidean()
        p2 = point2.normalized_euclidean()
        normal = p2 - p1
        midpoint = (p1 + p2) / 2.0
        return cls(normal, midpoint)

    @classmethod
    def from_three_points(
        cls,
        point0: PointHomogeneous,
        point1: PointHomogeneous,
        point2: PointHomogeneous,
    ) -> "NormalizedPlane":
        """
        Construct a plane through three non-collinear points.

        Parameters
        ----------
        point0 :
            First :class:`.PointHomogeneous`.
        point1 :
            Second :class:`.PointHomogeneous`.
        point2 :
            Third :class:`.PointHomogeneous`.

        Returns
        -------
        NormalizedPlane

        Raises
        ------
        ValueError
            If the three points are collinear.
        """
        p0 = point0.normalized_euclidean()
        p1 = point1.normalized_euclidean()
        p2 = point2.normalized_euclidean()
        normal = numpy.cross(p1 - p0, p2 - p0)
        if numpy.linalg.norm(normal) == 0.0:
            raise ValueError("NormalizedPlane.from_three_points: points are collinear.")
        return cls(normal, p0)

    @classmethod
    def from_line_and_point(
        cls,
        line: NormalizedLine,
        point: PointHomogeneous,
    ) -> "NormalizedPlane":
        """
        Construct the plane spanned by a line and a point not on the line.

        Parameters
        ----------
        line :
            A :class:`.NormalizedLine` contained in the plane.
        point :
            A :class:`.PointHomogeneous` contained in the plane.

        Returns
        -------
        NormalizedPlane

        Raises
        ------
        ValueError
            If *point* lies on *line*.
        """
        from .PointHomogeneous import PointHomogeneous  # lazy import

        if line.contains_point(point.normalized_euclidean()):
            raise ValueError(
                "NormalizedPlane.from_line_and_point: point lies on the line."
            )
        p1 = PointHomogeneous.from_3d_point(line.point_on_line(0.123456789))
        p2 = PointHomogeneous.from_3d_point(line.point_on_line(0.987654321))
        return cls.from_three_points(point, p1, p2)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """Return the plane coordinate at *idx*."""
        return self.coordinates[idx]

    def __len__(self) -> int:
        """Number of plane coordinates, always 4."""
        return 4

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        c = numpy.array2string(
            self.coordinates,
            precision=10,
            suppress_small=True,
            separator=", ",
            max_line_width=100000,
        )
        return f"{self.__class__.__qualname__}({c})"

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other: "NormalizedPlane") -> bool:
        """
        Coefficient-wise equality.

        Parameters
        ----------
        other :
            Plane to compare against.

        Returns
        -------
        bool
        """
        return numpy.array_equal(self.coordinates, other.coordinates)

    # ------------------------------------------------------------------
    # Cached properties
    # ------------------------------------------------------------------

    @property
    def reflection_matrix(self) -> numpy.ndarray:
        """
        3×3 Householder reflection matrix about the plane's normal.

        Cached after first access.

        Returns
        -------
        numpy.ndarray
            ``I - 2 * n ⊗ n``.
        """
        if self._reflection_matrix is None:
            self._reflection_matrix = (
                numpy.eye(3) - 2.0 * numpy.outer(self.normal, self.normal)
            )
        return self._reflection_matrix

    @property
    def reflection_tr(self) -> numpy.ndarray:
        """
        4×4 homogeneous reflection transformation matrix.

        Cached after first access.

        Returns
        -------
        numpy.ndarray
            4×4 array encoding the reflection about this plane.
        """
        if self._reflection_tr is None:
            mat = numpy.eye(4)
            mat[1:4, 1:4] = self.reflection_matrix
            mat[1:4, 0] = -2.0 * self.oriented_distance * self.normal
            self._reflection_tr = mat
        return self._reflection_tr

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return plane coordinates as a NumPy array.

        Returns
        -------
        numpy.ndarray
            4-vector ``[d, n1, n2, n3]``.
        """
        return self.coordinates.copy()

    def plane2dq_array(self) -> numpy.ndarray:
        """
        Embed the plane into dual quaternion space as an 8-vector.

        Maps ``[d, n]`` → ``[0, n1, n2, n3, d, 0, 0, 0]``.

        Returns
        -------
        numpy.ndarray
            8-vector.
        """
        return numpy.array([
            0,
            self.normal[0],
            self.normal[1],
            self.normal[2],
            self.oriented_distance,
            0,
            0,
            0,
        ])

    def intersection_with_plane(
        self, other: "NormalizedPlane"
    ) -> numpy.ndarray:
        """
        Return the Plücker screw coordinates of the line where two planes meet.

        Parameters
        ----------
        other :
            The second plane.

        Returns
        -------
        numpy.ndarray
            6-vector ``[direction | moment]``.
        """
        n1, n2 = self.normal, other.normal
        p1, p2 = self.point, other.point

        line_dir = numpy.cross(n1, n2)
        line_dir = line_dir / numpy.linalg.norm(line_dir)

        mat = numpy.stack([n1, n2, line_dir], axis=0)
        rhs = numpy.array([numpy.dot(n1, p1), numpy.dot(n2, p2), 0.0])
        line_point = numpy.linalg.lstsq(mat, rhs, rcond=None)[0]

        line_moment = numpy.cross(-line_dir, line_point)
        return numpy.concatenate((line_dir, line_moment))

    def intersection_with_line(
        self, line: NormalizedLine
    ) -> numpy.ndarray:
        """
        Return the homogeneous intersection point of the plane with a line.

        See Pottmann et al., *Computational Line Geometry*, 2001, §3.4.2,
        eq. 2.17.

        Parameters
        ----------
        line :
            The line to intersect with.

        Returns
        -------
        numpy.ndarray
            4-vector ``[w, x, y, z]`` in homogeneous coordinates.
        """
        p0 = numpy.dot(self.normal, line.direction)
        p_vec = (
            -self.oriented_distance * line.direction
            + numpy.cross(self.normal, line.moment)
        )
        return numpy.concatenate(([p0], p_vec))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def get_plot_data(
        self,
        xlim: tuple = (-1.0, 1.0),
        ylim: tuple = (-1.0, 1.0),
    ) -> tuple:
        """
        Return a meshgrid suitable for 3-D surface plotting.

        Parameters
        ----------
        xlim :
            ``(x_min, x_max)`` range. Default ``(-1, 1)``.
        ylim :
            ``(y_min, y_max)`` range. Default ``(-1, 1)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(x_pts, y_pts, z_pts)`` meshgrid arrays.
        """
        a, b, c = self.normal
        d = self.oriented_distance
        x = numpy.linspace(xlim[0], xlim[1], 5)
        y = numpy.linspace(ylim[0], ylim[1], 5)
        x_pts, y_pts = numpy.meshgrid(x, y)
        if numpy.isclose(c, 0.0):
            c = 1e-6
        z_pts = -(d + a * x_pts + b * y_pts) / c
        return x_pts, y_pts, z_pts