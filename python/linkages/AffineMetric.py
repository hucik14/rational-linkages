import numpy as np

from RationalCurve import RationalCurve
from PointHomogeneous import PointHomogeneous
from DualQuaternion import DualQuaternion
from AffineMatrix import AffineMatrix


class AffineMetric:
    """
    Class of affine metric in R12

    For more information, see M. Hofer's dissertation thesis titled "Variational Motion
    Design in the Presence of Obstacles", section 2.2 on page 6.
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

        self.matrix = self.create_affine_metric()

    def __repr__(self):
        return f"{self.matrix}"

    def create_affine_metric(self) -> np.ndarray:
        """
        Create the affine metric of the motion

        This function computes the metric matrix for a homogeneous 3D point based on
        the formulation from M. Hofer's dissertation thesis titled "Variational Motion
        Design in the Presence of Obstacles", specifically on page 7, equation 2.4.

        :return: np.ndarray - affine metric matrix in R12x12

        :references:
            - M. Hofer, "Variational Motion Design in the Presence of Obstacles",
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

        :return: np.ndarray - metric matrix of a single point in R12x12

        :references:
            - M. Hofer, "Variational Motion Design in the Presence of Obstacles",
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

        :return: list[DualQuaternion] - transformations of the curve
        """

        # tranformation at -1
        dq_1 = self.motion_curve.evaluate(-1)
        # transformation at 1
        dq_2 = self.motion_curve.evaluate(1)
        # transformation at infinity
        dq_inf = self.motion_curve.evaluate(0, inverted_part=True)

        return [DualQuaternion(dq_1), DualQuaternion(dq_2), DualQuaternion(dq_inf)]

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Distance between two affine displacements

        :param a:
        :param b:
        :return: float - distance between a and b
        """
        return np.sqrt(self.inner_product(a - b, a - b))

    def inner_product(self, dq_a: DualQuaternion, dq_b: DualQuaternion):
        """
        Inner product of two DualQuaternions

        :param dq_a: DualQuaternion
        :param dq_b: DualQuaternion

        :return: float - inner product of dq_a and dq_b
        """
        inner_product = np.zeros(4)
        for i in range(self.number_of_points):
            a_point = dq_a.act(self.points[i])
            b_point = dq_b.act(self.points[i])

            point = np.dot(a_point.array(), b_point.array())
            inner_product += point

        return inner_product
