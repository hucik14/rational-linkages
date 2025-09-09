from copy import deepcopy

import numpy as np
import sympy as sp

from .DualQuaternion import DualQuaternion
from .MiniBall import MiniBall
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve


class RationalBezier(RationalCurve):
    """
    Class representing rational Bezier curves in n-dimensional space.

    :examples:

    .. testcode:: [rationalbezier_example1]

        # Create a rational Bezier curve from control points

        # part of Limancon of Pascal

        from rational_linkages import RationalBezier, PointHomogeneous
        import numpy as np


        control_points = [PointHomogeneous(np.array([4.,  0., -2.,  4.])),
                          PointHomogeneous(np.array([0.,  1., -2.,  0.])),
                          PointHomogeneous(np.array([1.33333333, 2.66666667, 0., 1.33333333])),
                          PointHomogeneous(np.array([0., 1., 2., 0.])),
                          PointHomogeneous(np.array([4., 0., 2., 4.]))]
        bezier_curve = RationalBezier(control_points)

    .. testcleanup:: [rationalbezier_example1]

        del RationalBezier, PointHomogeneous, np
        del control_points, bezier_curve
    """

    def __init__(self,
                 control_points: list[PointHomogeneous]):
        """
        Initializes a RationalBezier object with the provided control points.

        :param list[PointHomogeneous] control_points: control points of the curve
        """
        super().__init__(self.get_polynomials_from_control_points(control_points))

        self.control_points = control_points
        self._ball = None

    @property
    def ball(self):
        """
        Get the smallest ball enclosing the control points of the curve
        """
        if self._ball is None:
            self._ball = MiniBall(self.control_points, metric=self.metric)
        return self._ball

    def get_polynomials_from_control_points(self,
                                            control_points: list[PointHomogeneous]
                                            ) -> (list[sp.Poly]):
        """
        Calculate the coefficients of the parametric equations of the curve from
        the control points.

        :param control_points: list[PointHomogeneous] - control points of the curve

        :return: np.array - coefficients of the parametric equations of the curve
        :rtype: list[sp.Poly]
        """
        t = sp.Symbol("t")
        degree = len(control_points) - 1
        dimension = control_points[0].coordinates.size - 1

        # Calculate the Bernstein basis polynomials and construct the Bezier curve
        bernstein_basis = self.get_bernstein_polynomial_equations(t, degree=degree)
        bezier_curve = [0] * (dimension + 1)
        for i in range(degree + 1):
            bezier_curve += bernstein_basis[i] * control_points[i].array()

        # Convert the Bezier curve to a set of polynomials
        bezier_polynomials = [
            sp.Poly(bezier_curve[i], t) for i in range(dimension + 1)]
        return bezier_polynomials

    @staticmethod
    def get_numerical_coeffs(control_points: list[PointHomogeneous]
                             ) -> np.ndarray:
        """
        Get the numerical coefficients of the Bezier curve
        """
        from scipy.special import comb  # lazy import

        control_pts = np.array([point.array() for point in control_points])
        degree = len(control_points) - 1

        mat = np.zeros((degree + 1, degree + 1))

        for j in range(degree + 1):
            for k in range(j + 1):
                mat[j, k] = comb(degree, k) * comb(degree - k, j - k) * (-1)**(j - k)

        return mat.dot(control_pts).T


    def get_plot_data(self, interval: tuple = (0, 1), steps: int = 50) -> tuple:
        """
        Get the data to plot the curve in 2D or 3D, based on the dimension of the curve

        :param interval: tuple - interval of the parameter t
        :param steps: int - number of discrete steps in the interval to plot the curve

        :return: x, y, z coordinates of the curve and x_cp, y_cp, z_cp
            coordinates of the control points
        :rtype: tuple
        """
        # perform superclass coordinates acquisition (only the curve)
        x, y, z = super().get_plot_data(interval, steps)

        if self.is_motion:
            points = [DualQuaternion(point.array()).dq2point_via_matrix()
                      for point in self.control_points]

        elif self.is_affine_motion:
            points = [point.coordinates[1:4]/point.coordinates[0]
                      for point in self.control_points]

        else:
            points = [self.control_points[i].normalized_in_3d()
                      for i in range(self.degree + 1)]

        x_cp, y_cp, z_cp = zip(*points)

        return x, y, z, x_cp, y_cp, z_cp

    def check_for_control_points_at_infinity(self):
        """
        Check if there is a control point at infinity

        :return: bool - True if there is a control point at infinity, False otherwise
        """
        return any(point.is_at_infinity for point in self.control_points)

    def check_for_negative_weights(self):
        """
        Check if there are negative weights in the control points

        :return: bool - True if there are negative weights, False otherwise
        """
        return any(point.coordinates[0] < 0 for point in self.control_points)


class BezierSegment:
    """
    Bezier curves that reparameterizes a motion curve in split segments.
    """
    def __init__(self,
                 control_points: list[PointHomogeneous],
                 t_param: tuple[bool, list[float]] = (False, [0, 1]),
                 metric: "AffineMetric" = None):
        """
        Initializes a BezierSegment object with the provided control points.

        :param control_points: list[PointHomogeneous] - control points of the curve
        :param t_param: tuple[bool, list[float]] - True if the Bezier curve is
            interpolation inverse part of reparameterized motion curve, False otherwise;
            list of two floats representing the original parameter interval of the
            motion curve
        """
        self.control_points = control_points
        self.t_param_of_motion_curve = t_param
        self._metric = metric

        self._ball = None
        self._curve = None

    @property
    def curve(self):
        """
        Get the Bezier curve
        """
        if self._curve is None:
            self._curve = RationalBezier(self.control_points)
        return self._curve

    @property
    def ball(self):
        """
        Get the smallest ball enclosing the control points of the curve
        """
        if self._ball is None:
            self._ball = MiniBall(self.control_points, metric=self.metric)
        return self._ball

    @property
    def metric(self):
        """
        Define a metric in R12 for the mechanism.

        This metric is used for collision detection.
        """
        if self._metric is None:
            return "euclidean"
        else:
            return self._metric

    @metric.setter
    def metric(self, metric: "AffineMetric"):
        from .AffineMetric import AffineMetric  # lazy import

        if isinstance(metric, AffineMetric):
            self._metric = metric
        elif metric == "euclidean" or metric is None:
            self._metric = None
        else:
            raise TypeError("The 'metric' property must be of type 'AffineMetric'")

    def split_de_casteljau(self,
                           t: float = 0.5,
                           ) -> tuple:
        """
        Split the curve at the given parameter value t

        :param float t: parameter value to split the curve at

        :return: tuple - two new Bezier curves
        :rtype: tuple
        """
        control_points = deepcopy(self.control_points)

        left_curve = [control_points[0]]
        right_curve = [control_points[-1]]

        # Perform De Casteljau subdivision until only two points remain
        while len(control_points) > 1:
            new_points = []

            # Compute linear interpolations between adjacent control points
            for i in range(len(control_points) - 1):
                new_points.append(
                    control_points[i].linear_interpolation(control_points[i + 1], t))

            # Append the first point of the new segment to the left curve
            left_curve.append(new_points[0])
            # Insert the last point of a new segment at the beginning of the right curve
            right_curve.insert(0, new_points[-1])

            # Update control points for the next iteration
            control_points = new_points

        mid_t = t * (self.t_param_of_motion_curve[1][0]
                     + self.t_param_of_motion_curve[1][1])

        new_t_left = (self.t_param_of_motion_curve[0],
                      [self.t_param_of_motion_curve[1][0], mid_t])
        new_t_right = (self.t_param_of_motion_curve[0],
                       [mid_t, self.t_param_of_motion_curve[1][1]])

        return (BezierSegment(left_curve, t_param=new_t_left, metric=self.metric),
                BezierSegment(right_curve, t_param=new_t_right, metric=self.metric))

    def check_for_control_points_at_infinity(self):
        """
        Check if there is a control point at infinity

        :return: bool - True if there is a control point at infinity, False otherwise
        """
        return any(point.is_at_infinity for point in self.control_points)

    def check_for_negative_weights(self):
        """
        Check if there are negative weights in the control points

        :return: bool - True if there are negative weights, False otherwise
        """
        return any(point.coordinates[0] < 0 for point in self.control_points)


class RationalSoo(RationalCurve):
    """
    Class representing rational Gauss-Legendre curves in n-dimensional space.

    The class implements the Gauss-Legendre curves to represent curved link segments.
    Gauss-Legendre curves, introduced in :footcite:t:`Moon2023`, have the property that
    the control polygon approximates the curve closely, and therefore can be used
    for collision detection, instead of using the curve polynomials.
    """
    def __init__(self,
                 control_points: list[PointHomogeneous]):
        """
        Initializes a RationalBezier object with the provided control points.

        :param list[PointHomogeneous] control_points: control points of the curve
        """
        super().__init__(self.get_poly_from_control_points(control_points))
        self.control_points = control_points

    def get_poly_from_control_points(self,
                                     control_points: list[PointHomogeneous]
                                     ) -> (list[sp.Poly]):
        """
        Calculate the coefficients of the parametric equations of the curve from
        the control points.

        :param control_points: list[PointHomogeneous] - control points of the curve

        :return: np.array - coefficients of the parametric equations of the curve
        :rtype: list[sp.Poly]
        """
        t = sp.Symbol("t")

        deg = len(control_points) - 1
        dim = control_points[0].coordinates.size

        taus, weights = np.polynomial.legendre.leggauss(deg)
        lagrange_basis = self.lagrange_basis(taus, t, weights)

        integrated_basis = []
        for base in lagrange_basis:
            # integrate from -1 to t
            integrated_basis.append(sp.integrate(base, (t, -1, t)) - 0.5)
        integrated_basis.insert(0, 0.5)
        integrated_basis.append(-0.5)

        gauss_legendre_basis = []
        for i in range(len(integrated_basis) - 1):
            gauss_legendre_basis.append(integrated_basis[i] - integrated_basis[i + 1])

        gl_curve = [0] * dim
        for i in range(len(control_points)):
            gl_curve += gauss_legendre_basis[i] * control_points[i].array()

        return [sp.Poly(gl_curve[i], t, greedy=False) for i in range(dim)]

    @classmethod
    def from_two_points(cls,
                        p0: PointHomogeneous,
                        p1: PointHomogeneous,
                        degree: int = 2) -> "RationalSoo":
        """
        Create a RationalSoo curve from two points.

        The other control points will be added based on the given degree.

        :param PointHomogeneous p0: first point
        :param PointHomogeneous p1: second point
        :param int degree: degree of the curve (default is 2)

        :return: the resulting Gauss-Legendre curve
        :rtype: RationalSoo
        """
        control_points = RationalSoo.control_points_between_two_points(p0, p1, degree)
        return cls(control_points)

    @staticmethod
    def control_points_between_two_points(p0: PointHomogeneous,
                                          p1: PointHomogeneous,
                                          degree: int = 2) -> list[PointHomogeneous]:
        """
        Generate control points for a Gauss-Legendre curve between two points.

        :param PointHomogeneous p0: first point
        :param PointHomogeneous p1: second point
        :param int degree: degree of the curve (default is 2)

        :return: list of control points
        :rtype: list[PointHomogeneous]
        """
        if degree < 2:
            raise ValueError("Degree must be at least 2 for a Gauss-Legendre curve.")

        control_points = [p0]
        for i in range(degree - 1):
            # create intermediate control points
            control_points.append(p0.linear_interpolation(p1, (i + 1) / degree))

        control_points.append(p1)

        return control_points

    @staticmethod
    def lagrange_basis(tau, symbol, weights):
        """
        Generate all Lagrange basis polynomials in symbolic form.

        :param tau: array-like9
        :param symbol: sympy.Symbol
        :param weights: array-like

        :return: list of polynomials
        :rtype: list[sp.Poly]
        """
        n = len(tau)
        basis = []

        for j in range(n):
            basis_j = 1
            for i in range(n):
                if i != j:
                    basis_j *= (symbol - tau[i]) / (tau[j] - tau[i])
            basis.append(basis_j / weights[j])

        return basis

    def get_plot_data(self,
                      interval: tuple = (-1, 1),
                      steps: int = 50) -> tuple:
        """
        Get the data to plot the curve in 3D.
        """
        # perform superclass coordinates
        x, y, z = super().get_plot_data(interval=interval)

        points = [point.normalized_in_3d() for point in self.control_points]

        x_cp, y_cp, z_cp = zip(*points)

        return x, y, z, x_cp, y_cp, z_cp
