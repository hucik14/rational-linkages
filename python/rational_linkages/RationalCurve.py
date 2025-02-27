from copy import deepcopy
from typing import Union

from scipy.integrate import quad

import numpy as np
import sympy as sp

from .PointHomogeneous import PointHomogeneous
from .DualQuaternion import DualQuaternion
from .Quaternion import Quaternion

MotionFactorization = "MotionFactorization"


class RationalCurve:
    """
    Class representing rational curves in n-dimensional space, where the first row is
    homogeneous coordinate equation.

    This class allows you to work with rational curves defined by parametric equations.

    :ivar coeffs: Coefficients of parametric equations of the curve.
    :ivar dimension: The dimension of the curve, excluding the homogeneous coordinate.
    :ivar degree: The degree of the curve.
    :ivar symbolic: Symbolic expressions for the parametric equations of the curve.
    :ivar set_of_polynomials: A set of polynomials representing the curve.
    :ivar symbolic_inversed: Symbolic expressions for the parametric equations of the
        inversed curve.
    :ivar set_of_polynomials_inversed: A set of polynomials representing the inversed
        curve.
    :ivar is_motion: True if the curve is a motion curve, False otherwise.

    :examples:

    .. testcode:: [rationalcurve_example1]

        # Limancon of Pascal -- from polynomial equations


        import sympy as sp
        from rational_linkages import RationalCurve

        a = 1
        b = 0.5
        t = sp.Symbol('t')
        eq0 = sp.Poly((1+t**2)**2, t)
        eq1 = sp.Poly(b*(1-t**2)*(1+t**2) + a*(1-t**2)**2, t)
        eq2 = sp.Poly(2*b*t*(1+t**2) + 2*a*t*(1-t**2), t)
        curve = RationalCurve([eq0, eq1, eq2, eq0])

    .. testcleanup:: [rationalcurve_example1]

        del RationalCurve, sp
        del a, b, t, eq0, eq1, eq2, curve

    .. testcode:: [rationalcurve_example2]

        # From coefficients

        import numpy as np
        from rational_linkages import RationalCurve


        curve = RationalCurve.from_coeffs(np.array([[1., 0., 2., 0., 1.], [0.5, 0., -2., 0., 1.5], [0., -1., 0., 3., 0.], [1., 0., 2., 0., 1.]]))

    .. testcleanup:: [rationalcurve_example2]

        del RationalCurve, np, curve
    """

    def __init__(self,
                 polynomials: list[sp.Poly],
                 coeffs: Union[np.array, sp.Matrix] = None,
                 metric: "AffineMetric" = None):
        """
        Initializes a RationalCurve object with the provided coefficients.

        :param polynomials: list of polynomial equations of the curve
        :param coeffs: coefficients of the curve
        """

        self.set_of_polynomials = polynomials

        self.dimension = len(self.set_of_polynomials) - 1
        # Get the degree of the curve
        self.degree = 1
        for i in range(len(polynomials)):
            self.degree = max(self.degree, self.set_of_polynomials[i].degree())

        self._coeffs = coeffs
        self._symbolic = None

        self.coeffs_inversed = self.inverse_coeffs()
        self._symbolic_inversed = None
        self._set_of_polynomials_inversed = None

        # check if the curve is a motion curve
        self.is_motion = self.dimension == 7
        self.is_affine_motion = self.dimension == 12

        self._metric = metric

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
        from .AffineMetric import AffineMetric  # inner import

        if isinstance(metric, AffineMetric):
            self._metric = metric
        elif metric == "euclidean" or metric is None:
            self._metric = None
        else:
            raise TypeError("The 'metric' property must be of type 'AffineMetric'")

    @property
    def symbolic(self):
        """
        Get the vector symbolic expressions of the curve

        :return: list of symbolic expressions
        :rtype: list
        """
        if self._symbolic is None:
            self._symbolic, _ = self.get_symbolic_expressions(self.coeffs)

        return self._symbolic

    @property
    def symbolic_inversed(self):
        """
        Get the vector symbolic expressions of the inversed curve

        :return: list of symbolic expressions
        :rtype: list
        """
        if self._symbolic_inversed is None:
            self._symbolic_inversed, self._set_of_polynomials_inversed \
                = self.get_symbolic_expressions(self.coeffs_inversed)

        return self._symbolic_inversed

    @property
    def set_of_polynomials_inversed(self):
        """
        Get the set of polynomials representing the inversed curve

        :return: list of symbolic expressions
        :rtype: list
        """
        if self._set_of_polynomials_inversed is None:
            self._symbolic_inversed, self._set_of_polynomials_inversed \
                = self.get_symbolic_expressions(self.coeffs_inversed)

        return self._set_of_polynomials_inversed

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = self.get_coeffs()
        return self._coeffs

    @classmethod
    def from_coeffs(cls, coeffs: Union[np.ndarray, sp.Matrix]) -> "RationalCurve":
        """
        Construct rational curve from coefficients

        :param Union[np.ndarray, sp.Matrix] coeffs: coefficients of the curve

        :returns: RationalCurve object from coefficients
        :rtype: RationalCurve
        """
        _, polynomials = cls.get_symbolic_expressions(coeffs)
        return cls(polynomials, coeffs)

    @classmethod
    def from_two_quaternions(cls,
                             rot: Quaternion,
                             transl: Quaternion) -> "RationalCurve":
        """
        Construct rational curve from rotational and transl. parts given as equations.

        The rotation and translation has to be given as vectorial quaternions, i.e.
        the real parts are zero.

        :rot: Quaternion - rotation part of the
        :transl: Quaternion - translational part of the curve

        :returns: RationalCurve object from rotational and translational parts
        :rtype: RationalCurve

        :raises ValueError: if the rotation and translation parts are not quaternionic
        """
        if len(rot.array()) != 4 or len(transl.array()) != 4:
            raise ValueError("The rotation and translation parts have to be "
                             "quaternionic polynomials")

        t = sp.Symbol('t')

        polynomials = np.concatenate((rot.array(), (-1/2) * (transl * rot).array()))

        # if one of the elements is not a sympy object, convert it
        polynomials = [sp.Poly(poly, t) for poly in polynomials]

        return cls(polynomials)

    @staticmethod
    def get_symbolic_expressions(coeffs: Union[np.ndarray, sp.Matrix]
                                 ) -> tuple[list, list[sp.Poly]]:
        """
        Add a symbolic variable to the matrix of coefficients that describes the curve

        :param Union[np.ndarray, sp.Matrix] coeffs: coefficients of the curve

        :return: tuple of symbolic expressions list and list of sympy polynomials
        :rtype: tuple[list, list[sp.Poly]]

        :raises ValueError: if the coefficients are not a numpy array or sympy matrix
        """
        symbolic_expressions = []
        polynomials = []
        t = sp.Symbol("t")

        if isinstance(coeffs, np.ndarray):
            dim = len(coeffs)
        elif isinstance(coeffs, sp.Matrix):
            dim = coeffs.rows
        else:
            raise ValueError("The coefficients must be a numpy array or sympy matrix")

        for i in range(dim):
            # Extract coefficients from the current row of the coefficient matrix and
            # create symbolic expressions
            row_coefficients = reversed(coeffs[i, :])
            symbolic_row_coeffs = [
                coefficient * t**j for j, coefficient in enumerate(row_coefficients)
            ]
            symbolic_expressions.append(sum(symbolic_row_coeffs))
            polynomials.append(sp.Poly(symbolic_expressions[i], t))

        return symbolic_expressions, polynomials

    def get_coeffs(self) -> np.ndarray:
        """
        Get the coefficients of the symbolic polynomial equations

        :return: np.array of coefficients
        :rtype: np.ndarray
        """
        # Obtain the coefficients
        coeffs = np.zeros((self.dimension + 1, self.degree + 1))
        for i in range(self.dimension + 1):
            # to fill all coeffs, check if the degree of the equation is the same
            # as the curve
            if len(self.set_of_polynomials[i].all_coeffs()) == self.degree + 1:
                coeffs[i, :] = np.array(self.set_of_polynomials[i].all_coeffs())
            else:  # if the degree of the equation is lower than the curve, check
                # the difference
                if not (self.set_of_polynomials[i].all_coeffs() == [0.0]
                        or self.set_of_polynomials[i].all_coeffs() == [0]):
                    # if the equation is not zero, fill the coeffs
                    degree_of_eq = self.set_of_polynomials[i].degree()
                    coeffs[i, self.degree - degree_of_eq :] = np.array(
                        self.set_of_polynomials[i].all_coeffs()
                    )
        return coeffs

    def __repr__(self):
        return f"RationalCurve({self.symbolic})"

    def curve2bezier_control_points(self,
                                    reparametrization: bool = False
                                    ) -> list[PointHomogeneous]:
        """
        Convert a curve to a Bezier curve using the Bernstein polynomials

        :param bool reparametrization: if True, the curve is mapped to the [-1,1]

        :return: list of Bezier control points
        :rtype: list[PointHomogeneous]
        """
        t = sp.Symbol("t")

        # Get the symbolic variables in the form of x00, x01, ... based on degree
        # of curve and dimension of space
        points = [
            [sp.Symbol("x%d_%d" % (i, j)) for j in range(self.dimension + 1)]
            for i in range(self.degree + 1)
        ]
        points_flattened = [var for variables in points for var in variables]

        # Get the Bernstein polynomial equations and Bernstein basis
        expression_list = self.get_bernstein_polynomial_equations(t, reparametrization=reparametrization)
        bernstein_basis = [0] * (self.dimension + 1)
        for i in range(self.dimension + 1):
            for j in range(self.degree + 1):
                bernstein_basis[i] += expression_list[j] * points[j][i]

        # Get the coefficients of the equations
        equations_coeffs = [
            sp.Poly((bernstein_basis[i] - self.symbolic[i]), t, greedy=False).all_coeffs()
            for i in range(self.dimension + 1)
        ]
        # Flatten the list
        equations_coeffs = [coeff for coeffs in equations_coeffs for coeff in coeffs]

        # Solve the equations
        points_sol = sp.linsolve(equations_coeffs, points_flattened)
        # Convert the solutions to numpy arrays (get points)
        points_array = np.array(points_sol.args[0], dtype="float64").reshape(
            self.degree + 1, self.dimension + 1)

        points_objects = [PointHomogeneous()] * (self.degree + 1)
        for i in range(self.degree + 1):
            points_objects[i] = PointHomogeneous(points_array[i, :])

        return points_objects

    def get_bernstein_polynomial_equations(self,
                                           t_var: sp.Symbol,
                                           reparametrization: bool = False,
                                           degree: int = None
                                           ) -> list:
        """
        Generate the Bernstein polynomial equation

        :param sp.Symbol t_var: symbolic variable
        :param bool reparametrization: if True, the curve is mapped to the [-1,1]
        :param degree: int - degree of the polynomial, if None (not specified),
            the degree of the curve is used

        :return: list of symbolic expressions
        :rtype: list
        """
        if degree is None:
            degree = self.degree

        if not reparametrization:
            t = t_var
        else:
            # Mapping of t to the interval [-1, 1]
            t = (t_var + 1) / 2

            # NOT WORKING because sp.Poly cannot handle
            # t = 1/sp.tan(t_var/2)
            # t = 1/t_var

        # Initialize the polynomial expression list
        expr = []
        # Generate the polynomial expression using the Bernstein polynomials
        for i in range(degree + 1):
            expr.append(sp.binomial(degree, i) * t**i * (1 - t) ** (degree - i))

        return expr

    def inverse_coeffs(self) -> np.ndarray:
        """
        Get the coefficients of the inverse curve

        :return: np.array of inversed coefficients
        :rtype: np.ndarray
        """
        inverse_coeffs = np.zeros((self.dimension + 1, self.degree + 1))
        for i in range(self.dimension + 1):
            inverse_coeffs[i, :] = self.coeffs[i, :][::-1]

        return inverse_coeffs

    def inverse_curve(self) -> "RationalCurve":
        """
        Get the inverse curve

        :return: inversed rational curve
        :rtype: RationalCurve
        """
        return RationalCurve.from_coeffs(self.inverse_coeffs())

    def curve(self) -> "RationalCurve":
        """
        Get the rational curve (itself) - suitable for subclasses, returns the
        superclass object

        :return: RationalCurve
        :rtype: RationalCurve
        """
        return RationalCurve(self.set_of_polynomials)

    def extract_expressions(self) -> list:
        """
        Extract the expressions of the curve

        :return: list of expressions of the curve (avoiding sp.Poly class)
        :rtype: list
        """
        return [self.set_of_polynomials[i].expr
                for i in range(len(self.set_of_polynomials))]

    def evaluate(self, t_param: Union[float, np.ndarray],
                 inverted_part: bool = False) -> np.ndarray:
        """
        Evaluate the curve for given t and return in the form of dual quaternion vector

        :param float, np.ndarray t_param: parameter of the motion curve
        :param bool inverted_part: if True, return the inverted part of the curve

        :return: pose of the curve as a dual quaternion vector
        :rtype: np.ndarray
        """
        t = sp.Symbol("t")
        if inverted_part:
            return np.array(
                [
                    self.set_of_polynomials_inversed[i].subs(t, t_param).evalf()
                    for i in range(len(self.set_of_polynomials_inversed))
                ],
                dtype="float64",
            )
        else:
            return np.array(
                [
                    self.set_of_polynomials[i].subs(t, t_param).evalf()
                    for i in range(len(self.set_of_polynomials))
                ],
                dtype="float64",
            )

    def evaluate_as_matrix(self, t_param, inverted_part: bool = False) -> np.ndarray:
        """
        Evaluate the curve for given t and return in the form of a transformation matrix

        :param float t_param: parameter of the motion curve
        :param bool inverted_part: if True, return the inverted part of the curve

        :return: pose of the curve as a matrix
        :rtype: np.ndarray
        """
        from .DualQuaternion import DualQuaternion

        dq = DualQuaternion(self.evaluate(t_param, inverted_part))
        return dq.dq2matrix()

    def factorize(self, use_rationals: bool = False) -> list[MotionFactorization]:
        """
        Factorize the curve into motion factorizations

        :param bool use_rationals: if True, force the factorization in QQ to return
            rational numbers

        :return: list of MotionFactorization objects
        :rtype: list[MotionFactorization]
        """
        if type(self) != RationalCurve:
            raise TypeError("Can factorize only for a rational curve or motion "
                            "factorization")

        from .FactorizationProvider import FactorizationProvider

        factorization_provider = FactorizationProvider(use_rationals=use_rationals)
        return factorization_provider.factorize_motion_curve(self)

    def get_plot_data(self, interval: Union[str, tuple] = (0, 1), steps: int = 50) -> (
            tuple)[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the data to plot the curve in 3D

        :param Union[str, tuple] interval: interval of the parameter t, if 'closed',
            the closed-loop curve is parametrized using tangent half-angle substitution
        :param int steps: number of numerical steps in the interval

        :return: tuple of np.ndarray - (x, y, z) coordinates of the curve
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        t = sp.Symbol("t")

        if interval == 'closed':
            # tangent half-angle substitution for closed curves
            t_space = np.tan(np.linspace(-np.pi/2, np.pi/2, steps + 1))
        else:
            t_space = np.linspace(interval[0], interval[1], steps)

        # make a copy of the polynomials and append the homogeneous coordinate
        # to the Z-equation place in the list if in 2D, so later z = 1
        polynoms = deepcopy(self.set_of_polynomials)
        if self.dimension == 2:
            polynoms.append(sp.Poly(polynoms[0], t))

        # plot the curve
        curve_points = [PointHomogeneous()] * steps
        for i in range(steps):
            point = self.evaluate(t_space[i])

            # if it is a pose in SE3, convert it to a point via matrix mapping
            if self.is_motion:
                point = DualQuaternion(point).dq2point_via_matrix()
                point = np.concatenate((np.array([1]), point))
            elif self.is_affine_motion:
                point = point[:4]

            curve_points[i] = PointHomogeneous([point[0], point[-3], point[-2], point[-1]])
        x, y, z = zip(*[curve_points[i].normalized_in_3d() for i in range(steps)])
        return x, y, z

    def get_curve_in_pr12(self) -> "RationalCurve":
        """
        Get the representation of the curve in PR12

        :return: curve in PR12
        :rtype: RationalCurve
        """
        if not self.is_motion:
            raise ValueError("The curve is not a motion curve, cannot convert to PR12")

        t = sp.Symbol("t")

        # convert the motion curve to a dual quaternion, then map to matrix
        curve_matrix = DualQuaternion(self.symbolic).dq2matrix(normalize=False)

        # save the not normalized coordinate
        curve_p = curve_matrix[0, 0]
        # transpose the matrix so the flatten() provides right order (vector by vector)
        curve_r12 = curve_matrix[1:4, :].T.flatten()

        # create PR12 vector of polynomial equations
        curve_pr12 = np.concatenate((np.array([curve_p]), curve_r12))
        curve_poly = [sp.Poly(curve, t) for curve in curve_pr12]

        return RationalCurve(curve_poly)

    def split_in_beziers(self,
                         min_splits: int = 0) -> list["BezierSegment"]:
        """
        Split the curve into Bezier curves with positive weights of control points.

        The curve is split into Bezier curves using the De Casteljau algorithm.

        :param int min_splits: minimal number of splits to be performed

        :return: list of RationalBezier objects
        :rtype: list[BezierSegment]
        """
        if not self.is_motion:
            raise ValueError("Not a motion curve, cannot split into Bezier curves.")

        from .RationalBezier import BezierSegment  # inner import

        curve = self.get_curve_in_pr12()

        # obtain Bezier curves for the curve and its reparametrized inverse part
        bezier_curve_segments = [
            # reparametrize the curve from the intervals [-1, 1]
            BezierSegment(curve.curve2bezier_control_points(reparametrization=True),
                          t_param=(False, [-1.0, 1.0]),
                          metric=self.metric),
            BezierSegment(curve.inverse_curve().curve2bezier_control_points(
                reparametrization=True),
                t_param=(True, [-1.0, 1.0]),
                metric=self.metric)
        ]

        # split the Bezier curves until all control points have positive weights
        # or no weights at infinity, or the minimal number of splits is reached
        while True:
            new_segments = [
                part for b_curve in bezier_curve_segments
                for part in (
                    b_curve.split_de_casteljau()
                    if b_curve.check_for_control_points_at_infinity()
                       or b_curve.check_for_negative_weights() else [b_curve])
            ]

            # if all control points have positive weights and no weights at infinity,
            # but the minimal number of splits is not reached, continue splitting
            if not any(
                    b_curve.check_for_control_points_at_infinity()
                    or b_curve.check_for_negative_weights()
                    for b_curve in new_segments):
                if len(new_segments) < min_splits:
                    new_segments = [part for b_curve in new_segments
                                    for part in b_curve.split_de_casteljau()]
                else:
                    bezier_curve_segments = new_segments
                    break

            bezier_curve_segments = new_segments

        return bezier_curve_segments

    def get_path_length(self, num_of_points: int = 100) -> float:
        """
        Get the length of the curve path

        Evaluates the curve in the given number of points and sums the distances between.

        :param int num_of_points: number of discrete points to evaluate the curve

        :return: length of the curve path
        :rtype: float
        """
        t_space = np.tan(np.linspace(-np.pi/2, np.pi/2, num_of_points))
        poses = [self.evaluate(t) for t in t_space]

        points = [DualQuaternion(p).dq2point_via_matrix()
                  for p in poses]

        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    def split_in_equal_segments(self,
                                interval: list[float],
                                point_to_act_on: PointHomogeneous = PointHomogeneous(),
                                num_segments: int = 10,) -> list[float]:
        """
        Find the t values that split the curve into equal segments in given interval

        Perform the arc length parameterization of the curve to split it into equal
        segments. The method uses the bisection method to find the t values.

        :param list[float] interval: interval of the parameter t
        :param PointHomogeneous point_to_act_on: point to act on
        :param int num_segments: number of segments to split the curve into

        :return: list of t values that split the curve into equal segments
        :rtype: list[float]

        :raises ValueError: if the interval is not in the form [a, b] where a < b
        :raises ValueError: if the interval values are identical
        :raises ValueError: if the number of segments is less than 1
        """
        if interval[0] > interval[1]:
            raise ValueError("The interval must be in the form [a, b] where a < b")
        elif interval[0] == interval[1]:
            raise ValueError("The interval values are identical")
        elif num_segments < 2:
            raise ValueError("The number of segments must be greater than 1")

        t = sp.Symbol('t')

        curve_dq = DualQuaternion(self.symbolic)
        point_path = curve_dq.act(point_to_act_on)

        dx_dt = sp.diff(point_path[1] / point_path[0], t)
        dy_dt = sp.diff(point_path[2] / point_path[0], t)
        dz_dt = sp.diff(point_path[3] / point_path[0], t)

        integrand = sp.sqrt(dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2)
        integrand_func = sp.lambdify(t, integrand, 'numpy')

        arc_length, _ = quad(integrand_func, interval[0], interval[1])
        desired_segment_length = arc_length / num_segments

        t_vals = [interval[0]]
        for i in range(num_segments - 1):
            b = self._bisection_find_t(t_vals[-1],
                                       interval,
                                       desired_segment_length,
                                       integrand_func)
            t_vals.append(b)
        t_vals.append(interval[1])

        return t_vals

    @staticmethod
    def _bisection_find_t(section_start: float,
                          curve_interval: list[float],
                          segment_length_target: float,
                          integrand_func: callable,
                          tolerance: float = 1e-14):
        """
        Find the t value that splits the curve into given segment length using bisection

        :param float section_start: start of the section
        :param list[float] curve_interval: interval of the parameter t
        :param float segment_length_target: target segment length
        :param callable integrand_func: integrand function
        :param float tolerance: tolerance of the bisection method

        :return: t value that splits the curve into given segment length
        :rtype: float
        """
        # initial lower and upper bounds
        low = section_start
        high = curve_interval[1]  # start with the upper bound

        # ensure the segment length at 'high' is greater than the target
        while True:
            segment_length_high, _ = quad(integrand_func, section_start, high)
            if segment_length_high >= segment_length_target:
                break
            high += 0.1 * (curve_interval[1] - curve_interval[0])

        # bisection
        while high - low > tolerance:
            mid = (low + high) / 2
            segment_length_mid, _ = quad(integrand_func, section_start, mid)

            if segment_length_mid < segment_length_target:
                low = mid
            else:
                high = mid

        t_val = (low + high) / 2
        return t_val

    def study_quadric_check(self) -> np.ndarray:
        """
        Calculate the error of the curve from the Study quadric

        :return: coefficients error of the curve from the Study quadric
        :rtype: np.ndarray
        """
        poly_list = [np.polynomial.Polynomial(self.coeffs[i, :][::-1])
                     for i in range(8)]

        study_quadric = (poly_list[0] * poly_list[4] + poly_list[1] * poly_list[5]
                         + poly_list[2] * poly_list[6] + poly_list[3] * poly_list[7])

        return study_quadric.coef

    def is_on_study_quadric(self):
        """
        Check if the curve is a motion curve on the study quadric

        :return: True if the curve is a motion curve, False otherwise
        :rtype: bool
        """
        study_quadric_error = self.study_quadric_check()

        return sum(abs(study_quadric_error)) < 1e-10


