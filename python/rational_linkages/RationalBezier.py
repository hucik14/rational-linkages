from copy import deepcopy

import sympy as sp

from .MiniBall import MiniBall
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve
from .DualQuaternion import DualQuaternion


class RationalBezier(RationalCurve):
    """
    Class representing rational Bezier curves in n-dimensional space.

    :examples:

    .. testcode::

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
    """

    def __init__(self,
                 control_points: list[PointHomogeneous]):
        """
        Initializes a RationalBezier object with the provided control points.

        :param list[PointHomogeneous] control_points: control points of the curve
        """
        super().__init__(self.get_coeffs_from_control_points(control_points))

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

    def get_coeffs_from_control_points(self,
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


class BezierSegment(RationalBezier):
    """
    Bezier curves that reparameterizes a motion curve in split segments.
    """
    def __init__(self,
                 control_points: list[PointHomogeneous],
                 t_param: tuple[bool, list[float]] = (False, [0, 1])):
        """
        Initializes a BezierSegment object with the provided control points.

        :param control_points: list[PointHomogeneous] - control points of the curve
        :param t_param: tuple[bool, list[float]] - True if the Bezier curve is
            interpolation inverse part of reparameterized motion curve, False otherwise;
            list of two floats representing the original parameter interval of the
            motion curve
        """
        super().__init__(control_points)

        self._ball = None
        self.t_param_of_motion_curve = t_param

    @property
    def ball(self):
        """
        Get the smallest ball enclosing the control points of the curve
        """
        if self._ball is None:
            self._ball = MiniBall(self.control_points, metric=self.metric)
        return self._ball

    def simplify_to_control_points(self):
        """
        Convert to BezierSegmentControlPoints class object to avoid init of sympy.Poly

        :return: Bezier segment as BezierSegmentControlPoints object
        :rtype: BezierSegmentControlPoints
        """
        return BezierSegmentControlPoints(self.control_points,
                                          t_param=self.t_param_of_motion_curve)

    def split_de_casteljau(self,
                           t: float = 0.5) -> tuple:
        """
        Split the curve at the given parameter value t

        :param float t: parameter value to split the curve at

        :return: tuple - two new Bezier curves
        :rtype: tuple
        """
        # get control points of the new curves
        left_cps, right_cps = BezierSegmentControlPoints(
            self.control_points,
            t_param=self.t_param_of_motion_curve).split_de_casteljau(t)
        return left_cps.as_bezier_segment(), right_cps.as_bezier_segment()


class BezierSegmentControlPoints:
    """
    Class representing control points of a Bezier curve segment
    """
    def __init__(self,
                 control_points: list[PointHomogeneous],
                 t_param: tuple[bool, list[float]] = (False, [0, 1]),
                 ):
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

        return (BezierSegmentControlPoints(left_curve, t_param=new_t_left),
                BezierSegmentControlPoints(right_curve, t_param=new_t_right))

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

    def return_as_bezier_segment(self):
        """
        Return the control points as a BezierSegment object

        :return: BezierSegment - Bezier curve segment
        :rtype: BezierSegment
        """
        return BezierSegment(self.control_points, t_param=self.t_param_of_motion_curve)
