import numpy as np

from .DualQuaternion import DualQuaternion
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve


class AffineMetric:
    """Affine metric for a motion defined by a rational curve.

    The affine metric aggregates point-wise metric contributions of a set of
    homogeneous 3D points and provides distances and inner-products for
    affine displacements represented as dual quaternions.

    References
    ----------
    Hofer, "Variational Motion Design in the Presence of Obstacles",
        dissertation thesis (2004), Page 7, Equation 2.4

    Schroecker, Weber, "Guaranteed collision detection with toleranced
        motions", Computer Aided Geometric Design (2014), Equation 3.
        http://dx.doi.org/10.1016/j.cagd.2014.08.001
    """
    def __init__(self, motion_curve: RationalCurve, points: list[PointHomogeneous]):
        """Initialize the affine metric for a motion.

        Parameters
        ----------
        motion_curve
            Rational curve representing the motion.
        points
            Points in 3D space given as homogeneous points that define the
            metric (each element is a :class:`PointHomogeneous`).
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
        """Compute the aggregate affine metric matrix for the motion.

        The metric is computed by summing the per-point metric contributions of
        the homogeneous 3D points stored in ``self.points`` (see
        :meth:`get_point_metric_matrix`). The returned matrix operates on 12D
        representations of affine displacements.

        Returns
        -------
        metric_matrix
            The affine metric matrix in R^{12x12}.
        """
        metric_matrix = np.zeros((12, 12))
        for i in range(self.number_of_points):
            metric_matrix += self.get_point_metric_matrix(self.points[i])
        return metric_matrix

    @staticmethod
    def get_point_metric_matrix(point: PointHomogeneous) -> np.ndarray:
        """Return the 12x12 metric matrix contribution of a single point.

        Parameters
        ----------
        point
            A homogeneous 3D point whose metric contribution is desired.

        Returns
        -------
        metric_matrix
            The metric matrix contribution for the supplied point.

        Notes
        -----
        The implementation follows the formulation in Hofer (2004), page 7,
        equation 2.4.
        """
        p = point.normalized_euclidean()
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
        """Return two representative transformations of the rational curve.

        The method evaluates the stored motion curve at -1 and at infinity
        (implemented as evaluation with ``inverted_part=True`` at parameter 0)
        and returns the corresponding dual quaternions.

        Returns
        -------
        list
            A two-element list containing the transformation at -1 and the
            transformation at infinity respectively.
        """

        # tranformation at -1
        dq_1 = self.motion_curve.evaluate(-1)
        # transformation at infinity
        dq_inf = self.motion_curve.evaluate(0, inverted_part=True)

        return [DualQuaternion(dq_1), DualQuaternion(dq_inf)]

    def distance_via_matrix(self, a: DualQuaternion, b: DualQuaternion) -> float:
        """Compute the metric distance between two affine displacements.

        Parameters
        ----------
        a, b
            Affine displacements represented as dual quaternions.

        Returns
        -------
        float
            The distance between ``a`` and ``b`` using the precomputed
            ``self.pose_distance_matrix``.
        """
        a12 = a.as_12d_vector()
        b12 = b.as_12d_vector()
        ab = a12 - b12
        return np.sqrt(ab @ self.pose_distance_matrix @ ab)

    def squared_distance_pr12_points(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute squared distance between two points given in projective R12.

        Parameters
        ----------
        a, b
            Projective R12 points where the first component is the homogeneous
            scale and the remaining 12 components form the 12D representation
            (this function uses ``a[1:]`` and ``b[1:]`` internally).

        Returns
        -------
        float
            The squared distance between the two supplied PR12 points.
        """
        a12 = a[1:]
        b12 = b[1:]

        ab = a12 - b12
        return ab @ self.pose_distance_matrix @ ab

    def distance(self, a: DualQuaternion, b: DualQuaternion) -> float:
        """Compute the Euclidean distance between two affine displacements.

        Parameters
        ----------
        a, b
            Affine displacements represented as dual quaternions.

        Returns
        -------
        float
            The Euclidean distance computed as the square root of the inner
            product between the two displacements.
        """
        return np.sqrt(self.inner_product(a, b))

    def squared_distance(self, a: DualQuaternion, b: DualQuaternion) -> float:
        """Return the squared distance between two affine displacements.

        Parameters
        ----------
        a, b
            Affine displacements represented as dual quaternions. If both
            have non-zero scalar parts they will be normalized by their scalar
            components before computing the inner product.

        Returns
        -------
        float
            The squared distance between ``a`` and ``b``.
        """
        if abs(a[0]) > 1e-10 and abs(b[0]) > 1e-10:
            a = a / a[0]
            b = b / b[0]
        return self.inner_product(a, b)

    def inner_product(self, a: DualQuaternion, b: DualQuaternion):
        """Compute the inner product of two dual quaternions w.r.t. this metric.

        The inner product is computed by acting both dual quaternions on each of
        the reference points that define the metric and summing the squared
        Euclidean distances between the resulting acted points.

        Parameters
        ----------
        a, b
            Affine displacements represented as dual quaternions.

        Returns
        -------
        float
            The inner product value (a non-negative scalar).
        """
        inner_product = 0
        for i in range(self.number_of_points):
            a_point = a.act(self.points[i])
            b_point = b.act(self.points[i])

            scalar = np.dot(a_point.normalized_euclidean() - b_point.normalized_euclidean(),
                            a_point.normalized_euclidean() - b_point.normalized_euclidean())
            inner_product += scalar

        return inner_product
