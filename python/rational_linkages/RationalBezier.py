from copy import deepcopy

import sympy as sp

from .MiniBall import MiniBall
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve


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

    def __init__(self, control_points: list[PointHomogeneous], reparametrization:
    bool = False):
        """
        Initializes a RationalBezier object with the provided control points.

        :param list[PointHomogeneous] control_points: control points of the curve
        """
        super().__init__(
            self.get_coeffs_from_control_points(control_points, reparametrization=reparametrization)
        )

        self.control_points = control_points

        # Calculate the bounding ball of the control points
        self.ball = MiniBall(self.control_points)

    def get_coeffs_from_control_points(
        self, control_points: list[PointHomogeneous], reparametrization) -> (
            list[sp.Poly]):
        """
        Calculate the coefficients of the parametric equations of the curve from
        the control points.

        :param control_points:
        :param mapped: bool - True if the curve is mapped to the [-1,1] interval,
            False keeps [0,1] interval
        :return: np.array - coefficients of the parametric equations of the curve
        """
        t = sp.Symbol("t")
        degree = len(control_points) - 1
        dimension = control_points[0].coordinates.size - 1

        # Calculate the Bernstein basis polynomials and construct the Bezier curve
        bernstein_basis = self.get_bernstein_polynomial_equations(
            t, reparametrization=reparametrization, degree=degree
        )
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

        :return: tuple - (x, y, z) coordinates of the curve
        """
        # perform superclass coordinates acquisition (only the curve)
        x, y, z = super().get_plot_data(interval, steps)

        # plot the control points
        x_cp, y_cp, z_cp = zip(
            *[self.control_points[i].normalized_in_3d() for i in range(self.degree + 1)]
        )
        return x, y, z, x_cp, y_cp, z_cp

    def check_for_control_points_at_infinity(self):
        """
        Check if there is a control point at infinity

        :return: bool - True if there is a control point at infinity, False otherwise
        """
        return any(
            self.control_points[i].is_at_infinity for i in range(self.degree + 1)
        )

    def split_de_casteljau(self, t=0.5):
        """
        Split the curve at the given parameter value t

        :param t: float - parameter value to split the curve at
        :return: tuple - two new Bezier curves
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
                    control_points[i].linear_interpolation(control_points[i + 1], t)
                )

            # Append the first point of the new segment to the left curve
            left_curve.append(new_points[0])
            # Insert the last point of a new segment at the beginning of the right curve
            right_curve.insert(0, new_points[-1])

            # Update control points for the next iteration
            control_points = new_points

        return RationalBezier(left_curve), RationalBezier(right_curve)
