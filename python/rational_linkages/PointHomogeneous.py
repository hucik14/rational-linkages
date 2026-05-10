from typing import Optional, Sequence
from warnings import warn

import numpy

from .backend import is_symbolic


class PointHomogeneous:
    """
    Point in projective space with homogeneous coordinates.

    Homogeneous coordinates ``[w, x, y, z]`` represent points in ℙⁿ,
    including points at infinity (where ``w = 0``). The coordinate vector
    may have length 3 (ℙ²), 4 (ℙ³), or higher ℙⁿ for more abstract applications.

    By default, all computation is performed with NumPy (``float64``). When
    the global backend is set to ``"sympy"`` via
    :func:`.set_backend`, construction transparently
    returns a :class:`~rational_linkages.PointHomogeneousSymbolic` instance instead,
    with no change to the calling code required.

    Parameters
    ----------
    point :
        Sequence of homogeneous coordinates ``[w, x, y, ...]``. If
        ``None``, the origin in ℙ³ ``[1, 0, 0, 0]`` is constructed.

    Attributes
    ----------
    coordinates : numpy.ndarray
        1-D array of homogeneous coordinates.
    is_at_infinity : bool
        ``True`` when the homogeneous coordinate ``w`` is (numerically)
        zero, i.e. the point lies on the hyperplane at infinity.
    is_2d : bool
        ``True`` if the point is in ℙ² (3 homogeneous coordinates).
    is_3d : bool
        ``True`` if the point is in ℙ³ (4 homogeneous coordinates).

    Examples
    --------
    .. code-block:: python

        from rational_linkages import PointHomogeneous


        origin = PointHomogeneous()
        custom = PointHomogeneous([2.0, 1.0, -3.0, 4.0])
        origin_2d = PointHomogeneous.at_origin_in_2d()

    .. clear-namespace::

    .. code-block:: python

        # Symbolic backend

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import PointHomogeneous
        from sympy import symbols

        w, x, y, z = symbols("w x y z", real=True)
        p = PointHomogeneous([w, x, y, z])   # transparently returns PointHomogeneousSymbolic

        rational_linkages.set_backend("numpy")

    .. clear-namespace::

    """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def __new__(cls, point=None, rational: bool = False):
        """
        Intercept construction and return a
        :class:`~rational_linkages.PointHomogeneousSymbolic` when the global backend
        is ``"sympy"``.

        Only applied when ``cls`` is exactly ``PointHomogeneous``; subclass
        constructors are never redirected, preventing infinite recursion.

        Parameters
        ----------
        point :
            Forwarded unchanged to ``__init__``.
        rational :
            Keeps or forces rational values by using symbolic backend

        Returns
        -------
        PointHomogeneous or PointHomogeneousSymbolic
        """
        if cls is PointHomogeneous:
            symbolic = is_symbolic() or (
                    point is not None
                    and any(hasattr(c, 'free_symbols') for c in point)
            )
            if symbolic:
                from .PointHomogeneousSymbolic import PointHomogeneousSymbolic
                return object.__new__(PointHomogeneousSymbolic)
        return object.__new__(cls)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, point: Optional[Sequence[float]] = None, rational: bool = False):
        self.is_rational = rational
        self.coordinates = self._initialize_coordinates(point)
        self._is_at_infinity = None
        self.is_2d = True if len(self.coordinates) == 3 else False
        self.is_3d = True if len(self.coordinates) == 4 else False
        self._normalized: Optional["PointHomogeneous"] = None

    def _initialize_coordinates(self, point: Optional[Sequence[float]]) -> numpy.ndarray:
        """
        Initialize the coordinate array.

        Returns ``float64`` for numeric input, ``object`` dtype when SymPy
        expressions are detected (fallback path; prefer setting the backend
        to ``"sympy"`` explicitly).

        Parameters
        ----------
        point :
            Sequence of homogeneous coordinates, or ``None`` for the origin.

        Returns
        -------
        numpy.ndarray
        """
        if point is None:
            return numpy.array([1.0, 0.0, 0.0, 0.0], dtype=numpy.float64)
        try:
            return numpy.asarray(point, dtype=numpy.float64)
        except (TypeError, ValueError):
            return numpy.array(point, dtype=object)

    def _check_if_at_infinity(self) -> bool:
        """
        Return ``True`` if the homogeneous coordinate is numerically zero.

        Returns
        -------
        bool
        """
        return bool(numpy.isclose(float(self.coordinates[0]), 0.0, atol=1e-12))

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def at_origin_in_2d(cls) -> "PointHomogeneous":
        """
        Construct the origin in ℙ².

        Returns
        -------
        PointHomogeneous
        """
        return cls(numpy.array([1.0, 0.0, 0.0]))

    @classmethod
    def from_3d_point(cls, point: numpy.ndarray) -> "PointHomogeneous":
        """
        Construct a homogeneous point from a 3-vector.

        Parameters
        ----------
        point :
            3-vector ``[x, y, z]``.

        Returns
        -------
        PointHomogeneous

        Raises
        ------
        ValueError
            If ``point`` is not a 3-vector.
        """
        point = numpy.asarray(point)
        if len(point) != 3:
            raise ValueError("PointHomogeneous.from_3d_point: point must be a 3-vector")
        return cls(numpy.insert(point, 0, 1.0))

    @classmethod
    def from_dual_quaternion(cls, dq: "DualQuaternion") -> "PointHomogeneous":
        """
        Construct a homogeneous point from a dual quaternion.

        Extracts ``[dq[0], dq[5], dq[6], dq[7]]`` as ``[w, x, y, z]``.

        Parameters
        ----------
        dq :
            Source dual quaternion.

        Returns
        -------
        PointHomogeneous
        """
        return cls([dq[0], dq[5], dq[6], dq[7]])

    # ------------------------------------------------------------------
    # Indexing / iteration
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """Return the coordinate at *idx*."""
        return self.coordinates[idx]

    def __len__(self) -> int:
        """Number of homogeneous coordinates."""
        return len(self.coordinates)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_at_infinity(self) -> bool:
        if self._is_at_infinity is None:
            self._is_at_infinity = self._check_if_at_infinity()
        return self._is_at_infinity

    @property
    def x(self) -> float:
        """
        The x coordinate of the point.

        Returns
        -------
        float
        """
        if self.is_at_infinity:
            warn(
                "PointHomogeneous.x property accessed on a point at infinity "
                "— the result has no unique Euclidean representative.",
                UserWarning,
                stacklevel=2,
            )

        if self.is_2d or self.is_3d:
            return self.coordinates[1] / self.coordinates[0]
        else:
            raise ValueError("PointHomogeneous.x property is only defined for 2D and 3D points")

    @property
    def y(self) -> float:
        """
        The y coordinate of the point.

        Returns
        -------
        float
        """
        if self.is_at_infinity:
            warn(
                "PointHomogeneous.y property accessed on a point at infinity "
                "— the result has no unique Euclidean representative.",
                UserWarning,
                stacklevel=2,
            )
        if self.is_2d or self.is_3d:
            return self.coordinates[2] / self.coordinates[0]
        else:
            raise ValueError("PointHomogeneous.z property is only defined for 2D and 3D points")

    @property
    def z(self) -> float:
        """
        The z coordinate of the point.

        Returns
        -------
        float
        """
        if self.is_at_infinity:
            warn(
                "PointHomogeneous.y property accessed on a point at infinity "
                "— the result has no unique Euclidean representative.",
                UserWarning,
                stacklevel=2,
            )
        if self.is_3d:
            return self.coordinates[3] / self.coordinates[0]
        else:
            raise ValueError("PointHomogeneous.z property is only defined for 3D points")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        p = numpy.array2string(
            self.coordinates,
            precision=10,
            suppress_small=True,
            separator=", ",
            max_line_width=100000,
        )
        return f"{self.__class__.__qualname__}({p})"

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: "PointHomogeneous") -> "PointHomogeneous":
        """
        Element-wise sum of two points.

        Parameters
        ----------
        other :
            Point to add.

        Returns
        -------
        PointHomogeneous
        """
        return self.__class__(self.coordinates + other.coordinates)

    def __sub__(self, other: "PointHomogeneous") -> "PointHomogeneous":
        """
        Element-wise difference of two points.

        Parameters
        ----------
        other :
            Point to subtract.

        Returns
        -------
        PointHomogeneous
        """
        return self.__class__(self.coordinates - other.coordinates)

    def __mul__(self, other) -> "PointHomogeneous":
        """
        Scale the point by a scalar.

        Parameters
        ----------
        other :
            Scalar value.

        Returns
        -------
        PointHomogeneous

        Raises
        ------
        ValueError
            If ``other`` is a ``PointHomogeneous``.
        """
        if isinstance(other, PointHomogeneous):
            raise ValueError("PointHomogeneous: cannot multiply two points")
        return self.__class__(self.coordinates * other)

    def __rmul__(self, other) -> "PointHomogeneous":
        """Scalar-on-left multiplication, delegates to ``__mul__``."""
        return self.__mul__(other)

    def __truediv__(self, other) -> "PointHomogeneous":
        """
        Divide the point by a scalar.

        Parameters
        ----------
        other :
            Scalar value.

        Returns
        -------
        PointHomogeneous

        Raises
        ------
        ValueError
            If ``other`` is a ``PointHomogeneous``.
        """
        if isinstance(other, PointHomogeneous):
            raise ValueError("PointHomogeneous: cannot divide two points")
        return self.__class__(self.coordinates / other)

    def __eq__(self, other: "PointHomogeneous") -> bool:
        """
        Coefficient-wise equality.

        Parameters
        ----------
        other :
            Point to compare against.

        Returns
        -------
        bool
        """
        return numpy.array_equal(self.coordinates, other.coordinates)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return coordinates as a NumPy array.

        Returns
        -------
        numpy.ndarray
        """
        return self.coordinates.copy()

    def norm(self):
        """
        Return the norm of the point.

        Returns
        -------
        float
        """
        return numpy.linalg.norm(self.normalized_euclidean())

    def normalize(self) -> "PointHomogeneous":
        """
        Return the normalized point (homogeneous coordinate scaled to 1).

        The result is cached after the first call.

        Returns
        -------
        PointHomogeneous
            Point with ``coordinates / coordinates[0]``.

        """
        if self._normalized is None:
            if self.is_at_infinity:
                self._normalized = self.__class__(
                    self.coordinates / numpy.linalg.norm(self.coordinates)
                )
            else:
                self._normalized = self.__class__(
                    self.coordinates / self.coordinates[0]
                )
        return self._normalized

    def normalized_euclidean(self) -> numpy.ndarray:
        """
        Return the Euclidean (non-homogeneous) coordinates.

        Normalizes the point and drops the leading homogeneous coordinate,
        returning ``coordinates[1:]`` after scaling by ``1 / w``.

        Returns
        -------
        numpy.ndarray
            Array of length ``len(coordinates) - 1``.

        Warnings
        --------
        ValueError
            If the point is at infinity.
        """
        if self.is_at_infinity:
            warn(
                "PointHomogeneous.normalized_euclidean() called on a point at infinity — "
                "the result has no unique Euclidean representative.",
                UserWarning,
                stacklevel=2,
            )
        return self.normalize().array()[1:]

    def normalized_in_3d(self) -> numpy.ndarray:
        """
        Return the Euclidean coordinates (homogeneous coordinate dropped).

        .. deprecated::
            Use :meth:`normalized_euclidean` instead. This method will be
            removed in a future version.

        Returns
        -------
        numpy.ndarray
        """
        warn(
            "PointHomogeneous.normalized_in_3d() is deprecated and will be removed in a "
            "future version. Use normalized_euclidean() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.normalized_euclidean()

    # ------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------

    def point2matrix(self) -> numpy.ndarray:
        """
        Convert to a homogeneous SE(3) matrix with identity rotation.

        Follows the European convention: the first column carries the
        translation and the remaining columns carry the rotation (identity
        here).

        Returns
        -------
        numpy.ndarray
            4×4 array.

        Raises
        ------
        ValueError
            If the coordinate length is not 3, 4, 12, or 13.
        """
        norm = self.normalize().array()
        mat = numpy.eye(4)

        if len(norm) == 3:          # ℙ²
            mat[1:3, 0] = norm[1:3]
        elif len(norm) == 4:        # ℙ³
            mat[1:4, 0] = norm[1:4]
        elif len(norm) == 12:       # affine displacement in ℝ¹²
            mat[1:4, 0] = norm[0:3]
            mat[1:4, 1] = norm[3:6]
            mat[1:4, 2] = norm[6:9]
            mat[1:4, 3] = norm[9:12]
        elif len(norm) == 13:       # affine displacement in ℙ¹²
            mat[1:4, 0] = norm[1:4]
            mat[1:4, 1] = norm[4:7]
            mat[1:4, 2] = norm[7:10]
            mat[1:4, 3] = norm[10:13]
        else:
            raise ValueError(
                "PointHomogeneous.point2matrix: coordinate length must be 3, 4, 12, or 13"
            )
        return mat

    def point2dq_array(self) -> numpy.ndarray:
        """
        Embed the point into dual quaternion space.

        Maps ``[w, x, y, z]`` → ``[w, 0, 0, 0, 0, x, y, z]``.

        Returns
        -------
        numpy.ndarray
            8-vector.
        """
        c = self.coordinates
        return numpy.array([c[0], 0, 0, 0, 0, c[1], c[2], c[3]])

    def point2affine12d(self, map_alpha: "TransfMatrix") -> numpy.ndarray:
        """
        Map the point to 12-dimensional affine space.

        Parameters
        ----------
        map_alpha :
            SE(3) matrix object providing ``.t``, ``.n``, ``.o``, ``.a``
            column vectors.

        Returns
        -------
        numpy.ndarray
            12-vector.
        """
        c = self.coordinates
        return numpy.concatenate((
            c[0] * map_alpha.t,
            c[1] * map_alpha.n,
            c[2] * map_alpha.o,
            c[3] * map_alpha.a,
        ))

    def linear_interpolation(self, other: "PointHomogeneous", t: float = 0.5) -> "PointHomogeneous":
        """
        Linearly interpolate between two points.

        Parameters
        ----------
        other :
            Target point.
        t :
            Interpolation parameter in ``[0, 1]``. Default ``0.5``.

        Returns
        -------
        PointHomogeneous
            Interpolated point.
        """
        return self.__class__(self.coordinates * (1.0 - t) + other.coordinates * t)

    def get_plot_data(self) -> numpy.ndarray:
        """
        Return Euclidean coordinates for 3-D plotting.

        Returns
        -------
        numpy.ndarray
            ``float64`` array of shape ``(3,)``.
        """
        return numpy.array(self.normalized_euclidean(), dtype=numpy.float64)

    def evalf(self):
        """
        Evaluate the coordinates to floating-point numbers.

        Only relevant for PointHomogeneousSymbolic. Numeric version returns
        just self.

        Returns
        -------
        PointHomogeneous
            Self.
        """
        return self

    def evalf_euclidean(self) -> numpy.ndarray:
        """
        Evaluate the Euclidean coordinates to floating-point numbers.

        Only relevant for PointHomogeneousSymbolic. Numeric version returns
        just self.normalized_euclidean().

        Returns
        -------
        numpy.ndarray
            ``float64`` array of shape ``(3,)``.
        """
        return self.normalized_euclidean()

    def eval(self, params: dict):
        """
        Placeholder for PointHomogeneousSymbolic.eval(). Evaluates line with given parameters.

        Parameters
        ----------
        params : dict
            Dictionary of Sympy parameters and values to be evaluated for.

        Returns
        -------
        PointHomogeneous
            Self.
        """
        return self

    def evaluate(self, param: float):
        """
        Placeholder for PointHomogeneousSymbolic.eval(). Evaluates line with a given parameter.

        Returns
        -------
        PointHomogeneous
            Self.
        """
        return self


class PointOrbit:
    def __init__(self, point_center, radius_squared, t_interval):
        """
        Orbit of a point (its covering ball)
        """
        if not isinstance(point_center, PointHomogeneous):
            self.center = PointHomogeneous(point_center)
        else:
            self.center = point_center

        self.radius_squared = radius_squared

        self._radius = None

        self.t_interval = t_interval

    def __repr__(self):
        return f"PointOrbit(center={self.center}, radius_squared={self.radius_squared}, t_interval={self.t_interval})"

    @property
    def radius(self):
        if self._radius is None:
            self._radius = numpy.sqrt(self.radius_squared)
        return self._radius

    def get_plot_data_mpl(self) -> tuple:
        """
        Get data for plotting in 3D space

        :return: surface coordinates
        :rtype: tuple
        """
        if len(self.center.coordinates) == 4:
            # Create the 3D sphere representing the circle
            u = numpy.linspace(0, 2 * numpy.pi, 10)
            v = numpy.linspace(0, numpy.pi, 10)

            x = (self.radius * numpy.outer(numpy.cos(u), numpy.sin(v))
                 + self.center.normalized_euclidean()[0])
            y = (self.radius * numpy.outer(numpy.sin(u), numpy.sin(v))
                 + self.center.normalized_euclidean()[1])
            z = (self.radius * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))
                 + self.center.normalized_euclidean()[2])
        else:
            raise ValueError("Cannot plot ball due to incompatible dimension.")

        return x, y, z

    def get_plot_data(self) -> tuple:
        """
        Get data for plotting in 3D space

        :return: center and radius
        :rtype: tuple
        """
        center = tuple(self.center.normalized_euclidean())
        return center, self.radius