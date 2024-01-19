from math import isclose
import numpy as np

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

    .. code-block:: python
        :caption: Create points in projective space

        from rational_linkages import PointHomogeneous


        origin_point_3D = PointHomogeneous()
        origin_point_2D = PointHomogeneous.at_origin_in_2d()
        custom_point = PointHomogeneous([2.0, 3.0, 4.0, 1.0])
    """

    def __init__(self, point=None):
        """
        Class to store points in PR3 or PR2

        Homogeneous coordinates are stored in the first row of the point array (index 0)
        :param point: array or list of floats
        """
        from sympy import Expr

        self.is_real = True

        if point is None:  # point in the origin in PR3
            self.coordinates = np.array([1, 0, 0, 0])
        elif any(isinstance(element, Expr) for element in point):
            try:
                self.coordinates = np.asarray(point, dtype='float64')
            except Exception:
                self.coordinates = point
                self.is_real = False
        else:
            self.coordinates = np.asarray(point, dtype='float64')

        if self.is_real and isclose(self.coordinates[0], 0.0):  # point at infinity
            self.is_at_infinity = True
            self.coordinates_normalized = None
        elif self.is_real:
            self.is_at_infinity = False
            self.coordinates_normalized = self.normalize()
        else:
            self.is_at_infinity = False
            self.coordinates_normalized = None

        #if len(self.coordinates_normalized) == 4:  # point in PR3
        #    self.as_dq_array = self.point2dq_array()

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
        return f"{self.coordinates}"

    def __add__(self, other):
        """
        Add two points
        :param other: Point
        :return: array of floats
        """
        return PointHomogeneous(self.coordinates + other.coordinates)

    def __sub__(self, other):
        """
        Subtract two points
        :param other: Point
        :return: array of floats
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
        return self.normalized_in_3d()

    def evaluate(self, t_param: float) -> 'PointHomogeneous':
        """
        Evaluate the point at the given parameter

        :param float t_param: parameter

        :return: evaluated point with float elements
        :rtype: PointHomogeneous
        """
        from sympy import Expr, Symbol

        t = Symbol("t")

        point = [Expr(self.coordinates[i]).subs(t, t_param)
                 for i in range(len(self.coordinates))]
        point = [point[j].args[0] for j in range(len(point))]
        return PointHomogeneous(np.asarray(point, dtype="float64"))
