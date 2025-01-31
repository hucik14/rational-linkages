import numpy as np

from .DualQuaternion import DualQuaternion
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve


class AffineMetric:
    """
    Class of affine metric in R12

    :references:
        M. Hofer, "Variational Motion Design in the Presence of Obstacles",
        dissertation thesis (2004), Page 7, Equation 2.4

        Schroecker, Weber, "Guaranteed collision detection with toleranced
        motions", Computer Aided Geometric Design (2014), Equation 3. DOI:
        http://dx.doi.org/10.1016/j.cagd.2014.08.001

    """
    def __init__(self, motion_curve: RationalCurve, points: list[PointHomogeneous]):
        """
        Construct the affine metric of a motion from the given points in the 3D space

        :param motion_curve: RationalCurve - rational curve representing the motion
        :param points: list[PointHomogeneous] - points in the 3D space
        """
        self.motion_curve = motion_curve
        self.points = points
        self.number_of_points = len(points)

        # By Hofer
        self.pose_distance_matrix = self.create_affine_metric()

        # By Schroecker, Weber
        self.inertia_matrix = np.sum([p[0] ** 2 * np.outer(p[1:], p[1:])
                                      for p in self.points], axis=0)
        self.inertia_eigen_vals = np.linalg.eigvals(self.inertia_matrix)
        self.total_mass = np.sum([p[0] for p in self.points])

    def __repr__(self):
        return f"{self.pose_distance_matrix}"

    def create_affine_metric(self) -> np.ndarray:
        """
        Create the affine metric of the motion

        This function computes the metric matrix for a homogeneous 3D point based on
        the formulation from M. Hofer's dissertation thesis titled "Variational Motion
        Design in the Presence of Obstacles", specifically on page 7, equation 2.4.

        :return: affine metric matrix in R12x12
        :rtype: np.ndarray

        :references:
            M. Hofer, "Variational Motion Design in the Presence of Obstacles",
            dissertation thesis (2004), Page 7, Equation 2.4

        """
        metric_matrix = np.zeros((12, 12))
        for i in range(self.number_of_points):
            metric_matrix += self.get_point_metric_matrix(self.points[i])
        return metric_matrix

    @staticmethod
    def get_point_metric_matrix(point: PointHomogeneous) -> np.ndarray:
        """
        Get the metric matrix of the given point

        :param point: PointHomogeneous - point in the 3D space

        :return: metric matrix of a single point in R12x12
        :rtype: np.ndarray

        :references:
            M. Hofer, "Variational Motion Design in the Presence of Obstacles",
            dissertation thesis (2004), Page 7, Equation 2.4

        """
        p = point.normalized_in_3d()
        i = np.eye(3)

        m00 = i
        m01 = p[0] * i
        m02 = p[1] * i
        m03 = p[2] * i

        m11 = p[0] ** 2 * i
        m12 = p[0] * p[1] * i
        m13 = p[0] * p[2] * i

        m22 = p[1] ** 2 * i
        m23 = p[1] * p[2] * i

        m33 = p[2] ** 2 * i

        metric_matrix = np.block([[m00, m01, m02, m03],
                                  [m01, m11, m12, m13],
                                  [m02, m12, m22, m23],
                                  [m03, m13, m23, m33]])

        return metric_matrix

    def get_curve_transformations(self) -> list[DualQuaternion]:
        """
        Get the transformations of the curve

        :return: transformations of the curve
        :rtype: list[DualQuaternion]
        """

        # tranformation at -1
        dq_1 = self.motion_curve.evaluate(-1)
        # transformation at infinity
        dq_inf = self.motion_curve.evaluate(0, inverted_part=True)

        return [DualQuaternion(dq_1), DualQuaternion(dq_inf)]

    def distance_via_matrix(self, a: DualQuaternion, b: DualQuaternion) -> float:
        """
        Distance between two affine displacements

        :param DualQuaternion a: displacement
        :param DualQuaternion b: displacement

        :return: distance between a and b
        :rtype: float
        """
        a12 = a.as_12d_vector()
        b12 = b.as_12d_vector()
        ab = a12 - b12
        return np.sqrt(ab @ self.pose_distance_matrix @ ab)

    def squared_distance_pr12_points(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Squared distance between two points in R12

        :param np.ndarray a: point in PR12
        :param np.ndarray b: point in PR12

        :return: squared distance between a and b
        :rtype: float
        """
        a12 = a[1:]
        b12 = b[1:]

        ab = a12 - b12
        return ab @ self.pose_distance_matrix @ ab

    def distance(self, a: DualQuaternion, b: DualQuaternion) -> float:
        """
        Distance between two affine displacements

        :param DualQuaternion a: displacement
        :param DualQuaternion b: displacement

        :return: float - distance between a and b
        :rtype: float
        """
        return np.sqrt(self.inner_product(a, b))

    def squared_distance(self, a: DualQuaternion, b: DualQuaternion) -> float:
        """
        Squared distance between two affine displacements

        :param DualQuaternion a: displacement
        :param DualQuaternion b: displacement

        :return: float - squared distance between a and b
        :rtype: float
        """
        if abs(a[0]) > 1e-10 and abs(b[0]) > 1e-10:
            a = a / a[0]
            b = b / b[0]
        return self.inner_product(a, b)

    def inner_product(self, a: DualQuaternion, b: DualQuaternion):
        """
        Inner product of two DualQuaternions in the affine space

        It is calculated as the sum of usual dot products of acted points, after the two
        dual quaternions act on the points that define the metric.

        :param DualQuaternion a: displacement
        :param DualQuaternion b: displacement

        :return: inner product of dq_a and dq_b
        :rtype: float
        """
        inner_product = 0
        for i in range(self.number_of_points):
            a_point = a.act(self.points[i])
            b_point = b.act(self.points[i])

            scalar = np.dot(a_point.normalized_in_3d() - b_point.normalized_in_3d(),
                            a_point.normalized_in_3d() - b_point.normalized_in_3d())
            inner_product += scalar

        return inner_product
