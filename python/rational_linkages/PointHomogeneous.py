import numpy as np

from typing import Optional, Sequence
from sympy import Rational

from .TransfMatrix import TransfMatrix

# Forward declarations for class names
DualQuaternion = "DualQuaternion"


class PointHomogeneous:
    """
    Points in projective space with homogeneous coordinates.

    Homogeneous coordinates are used to represent points, including points at infinity.
    The first row of the point array (index 0) stores the homogeneous coordinates.

    :ivar coordinates: Array of floats representing the homogeneous coordinates of the
        point.
    :ivar is_at_infinity: Indicates whether the point is at infinity (coordinates[0] is
        close to 0).

    :examples:

    .. testcode:: [point_homogeneous_example1]

        # Create points in projective space

        from rational_linkages import PointHomogeneous


        origin_point_3D = PointHomogeneous()
        origin_point_2D = PointHomogeneous.at_origin_in_2d()
        custom_point = PointHomogeneous([2.0, 3.0, 4.0, 1.0])

    .. testcleanup:: [point_homogeneous_example1]

        del PointHomogeneous
        del origin_point_3D, origin_point_2D, custom_point

    """

    def __init__(self,
                 point: Optional[Sequence[float]] = None,
                 rational: bool = False):
        """
        Class to store points in PR3 or PR2

        Homogeneous coordinates are stored in the first row of the point array (index 0)
        :param point: array or list of floats
        :param bool rational: flag to indicate if the point shall be treated as rational
            i.e. using SymPy expressions
        """
        self.is_rational = rational
        self.is_expression = False

        self.coordinates = self._initialize_coordinates(point)
        self.is_at_infinity = self._check_if_at_infinity()
        self.coordinates_normalized = self.normalize() if not (
            self.is_at_infinity) else None

        self.orbit = None

    def _initialize_coordinates(self, point: Optional[Sequence[float]]) -> np.ndarray:
        """
        Initialize the coordinates of the point

        If None, create point at origin in PR3, otherwise convert the point to float. If
        the point is an expression, it will be stored as a SymPy object.

        :param point: array or list of floats

        :return: array of floats or Sympy objects
        :rtype: np.ndarray
        """
        if point is None:  # Origin point in PR3
            return np.array([1, 0, 0, 0], dtype='float64')

        if self.is_rational:
            return np.array([Rational(coord) for coord in point], dtype=object)

        # try to convert the point to float, if it is an expression, it will fail
        try:
            return np.asarray(point, dtype='float64')
        except Exception:
            self.is_expression = True
            self.is_rational = True
            return np.array(point, dtype=object)

    def _check_if_at_infinity(self) -> bool:
        """
        Check if the point is at infinity

        :return: True if the point is at infinity, False otherwise
        :rtype: bool
        """
        if self.is_expression:
            return self.coordinates[0] == 0
        elif self.is_rational:
            return self.coordinates[0] == 0
        else:
            return np.isclose(self.coordinates[0], 0.0, atol=1e-12)

    @classmethod
    def at_origin_in_2d(cls):
        """
        Create homogeneous point at origin in 2D

        :return: PointHomogeneous
        """
        point = np.zeros(3)
        point[0] = 1
        return cls(point)

    @classmethod
    def from_3d_point(cls, point: np.ndarray):
        """
        Create homogeneous point from 3D point

        :param point: 3D point
        :return: PointHomogeneous
        """
        point = np.asarray(point)
        if len(point) != 3:
            raise ValueError("PointHomogeneous: point has to be 3D")
        point = np.insert(point, 0, 1)
        return cls(point)

    @classmethod
    def from_dual_quaternion(cls, dq: "DualQuaternion"):
        """
        Create homogeneous point from dual quaternion

        :param dq: DualQuaternion
        :return: PointHomogeneous
        """

        p0 = dq[0]
        p1 = dq[5]
        p2 = dq[6]
        p3 = dq[7]
        return cls([p0, p1, p2, p3])

    def __getitem__(self, idx):
        """
        Get specified coordinate from point
        :param idx: coordinate index
        :return: float
        """
        return self.coordinates[idx]

    def __repr__(self):
        """
        Print point
        :return:
        """
        p = np.array2string(self.array(),
                            precision=10,
                            suppress_small=True,
                            separator=', ',
                            max_line_width=100000)
        return f"{p}"

    def __add__(self, other: "PointHomogeneous"):
        """
        Add two points

        :param PointHomogeneous other: Other point

        :return: two points added together
        :rtype: PointHomogeneous
        """
        return PointHomogeneous(self.coordinates + other.coordinates)

    def __mul__(self, other):
        """
        Multiply point by scalar
        :param other: float
        :return: array of floats
        """
        if isinstance(other, PointHomogeneous):
            raise ValueError("PointHomogeneous: cannot multiply two points")
        return PointHomogeneous(self.coordinates * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Divide point by scalar
        :param other: float
        :return: array of floats
        """
        if isinstance(other, PointHomogeneous):
            raise ValueError("PointHomogeneous: cannot divide two points")
        return PointHomogeneous(self.coordinates / other)

    def __sub__(self, other):
        """
        Subtract two points

        :param PointHomogeneous other: Other point

        :return: two points subtracted
        :rtype: PointHomogeneous
        """
        return PointHomogeneous(self.coordinates - other.coordinates)

    def array(self) -> np.ndarray:
        """
        Return point as numpy array
        :return:
        """
        return self.coordinates

    def normalize(self) -> np.ndarray:
        """
        Normalize the point
        :return: 4x1 array
        """
        if self.is_at_infinity:
            return self.coordinates / np.linalg.norm(self.coordinates)
        else:
            return self.coordinates / self.coordinates[0]

    def normalized_in_3d(self) -> np.ndarray:
        """
        Normalize the point and return its 3D coordinates
        :return: 3x1 array
        """
        return self.coordinates_normalized[1:]

    def point2matrix(self) -> np.ndarray:
        """
        Convert point to homogeneous SE3 matrix with identity in rotation part

        This methods follows the European convention for SE3 matrices, i.e. the first
        column of the matrix is translation and the rotation part is represented by
        the remaining 3 columns.

        :return: 4x4 array
        :rtype: np.ndarray
        """
        mat = np.eye(4)
        if len(self.coordinates_normalized) == 3:  # point in PR2
            mat[1:3, 0] = self.coordinates_normalized[1:3]
        elif len(self.coordinates_normalized) == 4:
            mat[1:4, 0] = self.coordinates_normalized[1:4]

        # affine displacement in R12
        elif len(self.coordinates_normalized) == 12:
            mat[1:4, 0] = self.coordinates_normalized[0:3]
            mat[1:4, 1] = self.coordinates_normalized[3:6]
            mat[1:4, 2] = self.coordinates_normalized[6:9]
            mat[1:4, 3] = self.coordinates_normalized[9:12]

        # affine displacement in PR12
        elif len(self.coordinates_normalized) == 13:
            mat[1:4, 0] = self.coordinates_normalized[1:4]
            mat[1:4, 1] = self.coordinates_normalized[4:7]
            mat[1:4, 2] = self.coordinates_normalized[7:10]
            mat[1:4, 3] = self.coordinates_normalized[10:13]
        else:
            raise ValueError("PointHomogeneous: point has to be in PR2 or PR3")

        return mat

    def point2dq_array(self) -> np.ndarray:
        """
        Embed point to dual quaternion space

        :return: np.array of shape (8,)
        """
        return np.array(
            [
                self.coordinates[0],
                0,
                0,
                0,
                0,
                self.coordinates[1],
                self.coordinates[2],
                self.coordinates[3],
            ]
        )

    def point2affine12d(self, map_alpha: TransfMatrix) -> np.array:
        """
        Map point to 12D affine space

        :param TransfMatrix map_alpha: SE3matrix object that maps point to 12D affine
            space

        :return: 12D affine point
        :rtype: np.array
        """

        x = self.coordinates
        point0 = x[0] * map_alpha.t

        point1 = x[1] * map_alpha.n
        point2 = x[2] * map_alpha.o
        point3 = x[3] * map_alpha.a

        return np.concatenate((point0, point1, point2, point3))

    def linear_interpolation(self, other, t: float = 0.5) -> "PointHomogeneous":
        """
        Linear interpolation between two points

        :param other: PointHomogeneous
        :param t: parameter of interpolation in range [0, 1]
        :return: PointHomogeneous
        """
        return PointHomogeneous(self.coordinates * (1 - t) + other.coordinates * t)

    def get_plot_data(self) -> np.ndarray:
        """
        Get data for plotting in 3D space

        :return: np.ndarray of shape (3, 1)
        """
        return np.array(self.normalized_in_3d(), dtype="float64")

    def evaluate(self, t_param: float) -> 'PointHomogeneous':
        """
        Evaluate the point at the given parameter

        :param float t_param: parameter

        :return: evaluated point with float elements
        :rtype: PointHomogeneous
        """
        from sympy import Expr, Symbol, Number

        t = Symbol("t")

        point_expr = [Expr(coord) if not isinstance(coord, Number) else coord
                      for coord in self.coordinates]
        point = [coord.subs(t, t_param).evalf().args[0]
                 if not isinstance(coord, Number) else coord
                 for coord in point_expr]
        return PointHomogeneous(np.asarray(point, dtype="float64"))

    def get_point_orbit(self,
                        acting_center: "PointHomogeneous",
                        acting_radius: float,
                        metric: "AffineMetric",
                        ) -> tuple[np.ndarray, float]:
        """
        Get point orbit

        Equation from Schroecker and Webber, Guaranteed collision detection with
        toleranced motions, 2014, eq. 4.

        :param PointHomogeneous acting_center: center of the acting ball
        :param float acting_radius: squared radius of the orbit ball
        :param AffineMetric metric: metric of the curve

        :return: point center and radius squared
        :rtype: tuple[np.ndarray, float]
        """
        point_center = acting_center.point2matrix() @ self.coordinates_normalized

        coords_3d = self.normalized_in_3d()

        radius_squared = acting_radius * (1/metric.total_mass + np.sum([(coord ** 2 / metric.inertia_eigen_vals[i]) for i, coord in enumerate(coords_3d)]))

        return point_center, radius_squared


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

    @property
    def radius(self):
        if self._radius is None:
            self._radius = np.sqrt(self.radius_squared)
        return self._radius

    def get_plot_data_mpl(self) -> tuple:
        """
        Get data for plotting in 3D space

        :return: surface coordinates
        :rtype: tuple
        """
        if len(self.center.coordinates) == 4:
            # Create the 3D sphere representing the circle
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)

            x = (self.radius * np.outer(np.cos(u), np.sin(v))
                 + self.center.normalized_in_3d()[0])
            y = (self.radius * np.outer(np.sin(u), np.sin(v))
                 + self.center.normalized_in_3d()[1])
            z = (self.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                 + self.center.normalized_in_3d()[2])
        else:
            raise ValueError("Cannot plot ball due to incompatible dimension.")

        return x, y, z

    def get_plot_data(self) -> tuple:
        """
        Get data for plotting in 3D space

        :return: center and radius
        :rtype: tuple
        """
        center = tuple(self.center.normalized_in_3d())
        return center, self.radius
