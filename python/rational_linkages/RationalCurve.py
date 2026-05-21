from copy import deepcopy
from typing import Union

import numpy
import sympy

from .DualQuaternion import DualQuaternion
from .PointHomogeneous import PointHomogeneous
from .Quaternion import Quaternion


MotionFactorization = "MotionFactorization"
AffineMetric = "AffineMetric"
BezierSegment = "BezierSegment"


class RationalCurve:
    """
    Class representing rational curves in n-dimensional space, where the first row is
    homogeneous coordinate equation.

    This class allows you to work with rational curves defined by parametric equations.

    Attributes
    ----------
    coeffs : numpy.ndarray or sympy.Matrix
        Coefficients of parametric equations of the curve.
    dimension : int
        The dimension of the curve, excluding the homogeneous coordinate.
    degree : int
        The degree of the curve.
    symbolic : list
        Symbolic expressions for the parametric equations of the curve.
    set_of_polynomials : list of sympy.Poly
        A set of polynomials representing the curve.
    symbolic_inversed : list
        Symbolic expressions for the parametric equations of the inversed curve.
    set_of_polynomials_inversed : list of sympy.Poly
        A set of polynomials representing the inversed curve.
    is_motion : bool
        True if the curve is a motion curve, False otherwise.

    Examples
    --------
    .. code-block:: python

        # From symbolic equations

        from rational_linkages import RationalCurve, Plotter
        from sympy import symbols, Poly

        t = symbols('t')

        c = [t ** 2 + 3, -2*t, 2, 0, 0, 1, t, 0,]

        c = RationalCurve([Poly(p, t) for p in c])

        p = Plotter(backend='matplotlib', arrows_length=0.05)
        p.plot(c, interval='closed', with_poses=True)
        p.show()

    .. clear-namespace

    .. code-block:: python

        # Limancon of Pascal -- from polynomial equations
        import sympy
        from rational_linkages import RationalCurve

        a = 1
        b = 0.5
        t = sympy.Symbol('t')
        eq0 = sympy.Poly((1+t**2)**2, t)
        eq1 = sympy.Poly(b*(1-t**2)*(1+t**2) + a*(1-t**2)**2, t)
        eq2 = sympy.Poly(2*b*t*(1+t**2) + 2*a*t*(1-t**2), t)
        curve = RationalCurve([eq0, eq1, eq2, eq0])

    .. clear-namespace

    .. code-block:: python

        # From coefficients
        import numpy
        from rational_linkages import RationalCurve

        curve = RationalCurve.from_coeffs(numpy.array([[1., 0., 2., 0., 1.], [0.5, 0., -2., 0., 1.5], [0., -1., 0., 3., 0.], [1., 0., 2., 0., 1.]]))

    .. clear-namespace

    """
    def __init__(self,
                 polynomials: list[sympy.Poly],
                 coeffs: Union[numpy.ndarray, sympy.Matrix] = None,
                 metric: AffineMetric = None):
        """
        Initialize a RationalCurve object with the provided coefficients.

        Parameters
        ----------
        polynomials : list of sympy.Poly
            List of polynomial equations of the curve.
        coeffs : numpy.ndarray or sympy.Matrix, optional
            Coefficients of the curve. Default is None.
        metric : AffineMetric, optional
            Metric for the curve. Default is None.
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
        Get or set the metric in R12 for the mechanism.

        This metric is used for collision detection.

        Returns
        -------
        AffineMetric or str or None
            The metric used for collision detection. Returns 'euclidean' if not set.
        """
        if self._metric is None:
            return "euclidean"
        else:
            return self._metric

    @metric.setter
    def metric(self, metric: AffineMetric):
        """
        Set the metric for the mechanism.

        Parameters
        ----------
        metric : AffineMetric, str, or None
            The metric to use. If 'euclidean' or None, resets to default.

        Raises
        ------
        TypeError
            If the metric is not of type AffineMetric, 'euclidean', or None.
        """
        from rational_linkages.AffineMetric import AffineMetric  # lazy import

        if isinstance(metric, AffineMetric):
            self._metric = metric
        elif metric == "euclidean" or metric is None:
            self._metric = None
        else:
            raise TypeError("The 'metric' property must be of type 'AffineMetric'")

    @property
    def symbolic(self):
        """
        Get the vector symbolic expressions of the curve.

        Returns
        -------
        list
            List of symbolic expressions for the curve.
        """
        if self._symbolic is None:
            self._symbolic, _ = self.get_symbolic_expressions(self.coeffs)

        return self._symbolic

    @property
    def symbolic_inversed(self):
        """
        Get the vector symbolic expressions of the inversed curve.

        Returns
        -------
        list
            List of symbolic expressions for the inversed curve.
        """
        if self._symbolic_inversed is None:
            self._symbolic_inversed, self._set_of_polynomials_inversed \
                = self.get_symbolic_expressions(self.coeffs_inversed)

        return self._symbolic_inversed

    @property
    def set_of_polynomials_inversed(self):
        """
        Get the set of polynomials representing the inversed curve.

        Returns
        -------
        list of sympy.Poly
            List of polynomials for the inversed curve.
        """
        if self._set_of_polynomials_inversed is None:
            self._symbolic_inversed, self._set_of_polynomials_inversed \
                = self.get_symbolic_expressions(self.coeffs_inversed)

        return self._set_of_polynomials_inversed

    @property
    def coeffs(self):
        """
        Get the coefficients of the curve.

        Returns
        -------
        numpy.ndarray
            Coefficient matrix of the curve.
        """
        if self._coeffs is None:
            self._coeffs = self.get_coeffs()
        return self._coeffs

    @classmethod
    def from_coeffs(cls, coeffs: Union[numpy.ndarray, sympy.Matrix]) -> "RationalCurve":
        """
        Construct a RationalCurve from coefficients.

        Parameters
        ----------
        coeffs : numpy.ndarray or sympy.Matrix
            Coefficients of the curve.

        Returns
        -------
        RationalCurve
            RationalCurve object from coefficients.
        """
        _, polynomials = cls.get_symbolic_expressions(coeffs)
        return cls(polynomials, coeffs)

    @classmethod
    def from_two_quaternions(cls,
                             rot: Quaternion,
                             transl: Quaternion) -> "RationalCurve":
        """
        Construct a RationalCurve from rotational and translational parts given as equations.

        The rotation and translation must be given as vectorial quaternions (real parts zero).

        Parameters
        ----------
        rot : Quaternion
            Rotation part of the curve.
        transl : Quaternion
            Translational part of the curve.

        Returns
        -------
        RationalCurve
            RationalCurve object from rotational and translational parts.

        Raises
        ------
        ValueError
            If the rotation and translation parts are not quaternionic.
        """
        if len(rot.array()) != 4 or len(transl.array()) != 4:
            raise ValueError("The rotation and translation parts have to be "
                             "quaternionic polynomials")

        t = sympy.Symbol('t')

        polynomials = numpy.concatenate((rot.array(), (-1/2) * (transl * rot).array()))

        # if one of the elements is not a sympy object, convert it
        polynomials = [sympy.Poly(poly, t) for poly in polynomials]

        return cls(polynomials)

    @staticmethod
    def get_symbolic_expressions(coeffs: Union[numpy.ndarray, sympy.Matrix]
                                 ) -> tuple[list, list[sympy.Poly]]:
        """
        Add a symbolic variable to the matrix of coefficients that describes the curve.

        Parameters
        ----------
        coeffs : numpy.ndarray or sympy.Matrix
            Coefficients of the curve.

        Returns
        -------
        tuple
            Tuple of (list of symbolic expressions, list of sympy.Poly polynomials).

        Raises
        ------
        ValueError
            If the coefficients are not a numpy array or sympy matrix.
        """
        symbolic_expressions = []
        polynomials = []
        t = sympy.Symbol("t")

        if isinstance(coeffs, numpy.ndarray):
            dim = len(coeffs)
        elif isinstance(coeffs, sympy.Matrix):
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
            polynomials.append(sympy.Poly(symbolic_expressions[i], t))

        return symbolic_expressions, polynomials

    def get_coeffs(self) -> numpy.ndarray:
        """
        Get the coefficients of the symbolic polynomial equations.

        Returns
        -------
        numpy.ndarray
            Coefficient matrix of the curve.
        """
        # Obtain the coefficients
        coeffs = numpy.zeros((self.dimension + 1, self.degree + 1))
        for i in range(self.dimension + 1):
            # to fill all coeffs, check if the degree of the equation is the same
            # as the curve
            if len(self.set_of_polynomials[i].all_coeffs()) == self.degree + 1:
                coeffs[i, :] = numpy.array(self.set_of_polynomials[i].all_coeffs())
            else:  # if the degree of the equation is lower than the curve, check
                # the difference
                if not (self.set_of_polynomials[i].all_coeffs() == [0.0]
                        or self.set_of_polynomials[i].all_coeffs() == [0]):
                    # if the equation is not zero, fill the coeffs
                    degree_of_eq = self.set_of_polynomials[i].degree()
                    coeffs[i, self.degree - degree_of_eq :] = numpy.array(
                        self.set_of_polynomials[i].all_coeffs()
                    )
        return coeffs

    def __repr__(self):
        """
        Return a string representation of the RationalCurve.

        Returns
        -------
        str
            String representation of the curve.
        """
        return f"RationalCurve({self.symbolic})"

    def curve2bezier_control_points(self,
                                    reparametrization: bool = False
                                    ) -> list[PointHomogeneous]:
        """
        Convert a curve to a Bezier curve using the Bernstein polynomials.

        Parameters
        ----------
        reparametrization : bool, optional
            If True, the curve is mapped to the [-1, 1] interval. Default is False.

        Returns
        -------
        list of PointHomogeneous
            List of Bezier control points.
        """
        t = sympy.Symbol("t")

        # Get the symbolic variables in the form of x00, x01, ... based on degree
        # of curve and dimension of space
        points = [
            [sympy.Symbol("x%d_%d" % (i, j)) for j in range(self.dimension + 1)]
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
            sympy.Poly((bernstein_basis[i] - self.symbolic[i]), t, greedy=False).all_coeffs()
            for i in range(self.dimension + 1)
        ]
        # Flatten the list
        equations_coeffs = [coeff for coeffs in equations_coeffs for coeff in coeffs]

        # Solve the equations
        points_sol = sympy.linsolve(equations_coeffs, points_flattened)
        # Convert the solutions to numpy arrays (get points)
        points_array = numpy.array(points_sol.args[0], dtype="float64").reshape(
            self.degree + 1, self.dimension + 1)

        points_objects = [PointHomogeneous()] * (self.degree + 1)
        for i in range(self.degree + 1):
            points_objects[i] = PointHomogeneous(points_array[i, :])

        return points_objects

    def get_bernstein_polynomial_equations(self,
                                           t_var: sympy.Symbol,
                                           reparametrization: bool = False,
                                           degree: int = None
                                           ) -> list:
        """
        Generate the Bernstein polynomial equations.

        Parameters
        ----------
        t_var : sympy.Symbol
            Symbolic variable for the parameter.
        reparametrization : bool, optional
            If True, the curve is mapped to the [-1, 1] interval. Default is False.
        degree : int, optional
            Degree of the polynomial. If None, uses the degree of the curve.

        Returns
        -------
        list
            List of symbolic expressions for the Bernstein basis.
        """
        if degree is None:
            degree = self.degree

        if not reparametrization:
            t = t_var
        else:
            # Mapping of t to the interval [-1, 1]
            t = (t_var + 1) / 2

            # NOT WORKING because sympy.Poly cannot handle
            # t = 1/sympy.tan(t_var/2)
            # t = 1/t_var

        # Initialize the polynomial expression list
        expr = []
        # Generate the polynomial expression using the Bernstein polynomials
        for i in range(degree + 1):
            expr.append(sympy.binomial(degree, i) * t**i * (1 - t) ** (degree - i))

        return expr

    def inverse_coeffs(self) -> numpy.ndarray:
        """
        Get the coefficients of the inverse curve.

        Returns
        -------
        numpy.ndarray
            Coefficient matrix of the inverse curve.
        """
        inverse_coeffs = numpy.zeros((self.dimension + 1, self.degree + 1))
        for i in range(self.dimension + 1):
            inverse_coeffs[i, :] = self.coeffs[i, :][::-1]

        return inverse_coeffs

    def inverse_curve(self) -> "RationalCurve":
        """
        Get the inverse curve.

        Returns
        -------
        RationalCurve
            Inversed rational curve.
        """
        return RationalCurve.from_coeffs(self.inverse_coeffs())

    def curve(self) -> "RationalCurve":
        """
        Get the rational curve (itself).

        Returns
        -------
        RationalCurve
            The curve itself (for subclass compatibility).
        """
        return RationalCurve(self.set_of_polynomials)

    def extract_expressions(self) -> list:
        """
        Extract the expressions of the curve (avoiding sympy.Poly class).

        Returns
        -------
        list
            List of expressions of the curve.
        """
        return [self.set_of_polynomials[i].expr
                for i in range(len(self.set_of_polynomials))]

    def evaluate(self,
                 t_param: Union[float, numpy.ndarray],
                 inverted_part: bool = False,
                 numerically: bool = True) -> numpy.ndarray:
        """
        Evaluate the curve for a given parameter and return as a dual quaternion vector.

        Parameters
        ----------
        t_param : float or numpy.ndarray
            Parameter value(s) for the curve.
        inverted_part : bool, optional
            If True, return the inverted part of the curve. Default is False.
        numerically : bool, optional
            If True, returns numerical (numpy) values, otherwise sympy

        Returns
        -------
        numpy.ndarray
            Pose of the curve as a dual quaternion vector.
        """
        t = sympy.Symbol("t")
        if numerically:
            if inverted_part:
                return numpy.array(
                    [self.set_of_polynomials_inversed[i].as_expr().subs(t, t_param).evalf()
                     for i in range(len(self.set_of_polynomials_inversed))],
                    dtype="float64",
                )
            else:
                return numpy.array(
                    [self.set_of_polynomials[i].as_expr().subs(t, t_param).evalf()
                     for i in range(len(self.set_of_polynomials))],
                    dtype="float64",
                )
        else:  # sympy
            if inverted_part:
                return numpy.array(
                    [self.set_of_polynomials_inversed[i].as_expr().subs(t, t_param)
                     for i in range(len(self.set_of_polynomials_inversed))],
                    dtype=object,
                )
            else:
                return numpy.array(
                    [self.set_of_polynomials[i].as_expr().subs(t, t_param)
                     for i in range(len(self.set_of_polynomials))],
                    dtype=object,
                )

    def evaluate_as_matrix(self, t_param, inverted_part: bool = False) -> numpy.ndarray:
        """
        Evaluate the curve for a given parameter and return as a transformation matrix.

        Parameters
        ----------
        t_param : float
            Parameter value for the curve.
        inverted_part : bool, optional
            If True, return the inverted part of the curve. Default is False.

        Returns
        -------
        numpy.ndarray
            Pose of the curve as a transformation matrix.
        """
        from .DualQuaternion import DualQuaternion

        dq = DualQuaternion(self.evaluate(t_param, inverted_part))
        return dq.dq2matrix()

    def factorize(self, use_rationals: bool = False) -> list[MotionFactorization]:
        """
        Factorize the curve into motion factorizations.

        Parameters
        ----------
        use_rationals : bool, optional
            If True, force the factorization in QQ to return rational numbers. Default is False.

        Returns
        -------
        list of MotionFactorization
            List of MotionFactorization objects.
        """
        if type(self) is not RationalCurve:
            raise TypeError("Can factorize only for a rational curve or motion "
                            "factorization")

        from .FactorizationProvider import FactorizationProvider

        factorization_provider = FactorizationProvider(use_rationals=use_rationals)
        return factorization_provider.factorize_motion_curve(self)

    def get_plot_data(self, interval: Union[str, tuple] = (0, 1), steps: int = 50) -> tuple:
        """
        Get the data to plot the curve in 3D.

        Parameters
        ----------
        interval : str or tuple, optional
            Interval of the parameter t. If 'closed', the closed-loop curve is parametrized using tangent half-angle substitution. Default is (0, 1).
        steps : int, optional
            Number of numerical steps in the interval. Default is 50.

        Returns
        -------
        tuple of numpy.ndarray
            (x, y, z) coordinates of the curve.
        """
        t = sympy.Symbol("t")

        if interval == 'closed':
            # tangent half-angle substitution for closed curves
            t_space = numpy.tan(numpy.linspace(-numpy.pi/2, numpy.pi/2, steps + 1))
        else:
            t_space = numpy.linspace(interval[0], interval[1], steps)

        # make a copy of the polynomials and append the homogeneous coordinate
        # to the Z-equation place in the list if in 2D, so later z = 1
        polynoms = deepcopy(self.set_of_polynomials)
        if self.dimension == 2:
            polynoms.append(sympy.Poly(polynoms[0], t))

        # plot the curve
        curve_points = [PointHomogeneous()] * steps
        for i in range(steps):
            point = self.evaluate(t_space[i])

            # if it is a pose in SE3, convert it to a point via matrix mapping
            if self.is_motion:
                point = DualQuaternion(point).dq2point_via_matrix()
                point = numpy.concatenate((numpy.array([1]), point))
            elif self.is_affine_motion:
                point = point[:4]

            curve_points[i] = PointHomogeneous([point[0], point[-3], point[-2], point[-1]])
        x, y, z = zip(*[curve_points[i].normalized_euclidean() for i in range(steps)])
        return x, y, z

    def get_curve_in_pr12(self) -> "RationalCurve":
        """
        Get the representation of the curve in PR12.

        Returns
        -------
        RationalCurve
            Curve in PR12.

        Raises
        ------
        ValueError
            If the curve is not a motion curve.
        """
        if not self.is_motion:
            raise ValueError("The curve is not a motion curve, cannot convert to PR12")

        t = sympy.Symbol("t")

        # convert the motion curve to a dual quaternion, then map to matrix
        curve_matrix = DualQuaternion(self.symbolic).dq2matrix(normalize=False)

        # save the not normalized coordinate
        curve_p = curve_matrix[0, 0]
        # transpose the matrix so the flatten() provides right order (vector by vector)
        curve_r12 = curve_matrix[1:4, :].T.flatten()

        # create PR12 vector of polynomial equations
        curve_pr12 = numpy.concatenate((numpy.array([curve_p]), curve_r12))
        curve_poly = [sympy.Poly(curve, t) for curve in curve_pr12]

        return RationalCurve(curve_poly)

    def split_in_beziers(self,
                         min_splits: int = 0) -> list[BezierSegment]:
        """
        Split the curve into Bezier curves with positive weights of control points.

        The curve is split into Bezier curves using the De Casteljau algorithm.

        Parameters
        ----------
        min_splits : int, optional
            Minimal number of splits to be performed. Default is 0.

        Returns
        -------
        list of BezierSegment
            List of BezierSegment objects.
        """
        if not self.is_motion:
            raise ValueError("Not a motion curve, cannot split into Bezier curves.")

        from .RationalBezier import BezierSegment  # lazy import

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
        Get the length of the curve path.

        Evaluates the curve in the given number of points and sums the distances between.

        Parameters
        ----------
        num_of_points : int, optional
            Number of discrete points to evaluate the curve. Default is 100.

        Returns
        -------
        float
            Length of the curve path.
        """
        t_space = numpy.tan(numpy.linspace(-numpy.pi/2, numpy.pi/2, num_of_points))
        poses = [self.evaluate(t) for t in t_space]

        points = [DualQuaternion(p).dq2point_via_matrix()
                  for p in poses]

        return numpy.sum(numpy.linalg.norm(numpy.diff(points, axis=0), axis=1))

    def split_in_equal_segments(self,
                                interval: list[float],
                                point_to_act_on: PointHomogeneous = PointHomogeneous(),
                                num_segments: int = 10,) -> list[float]:
        """
        Find the t values that split the curve into equal segments in a given interval.

        Performs arc length parameterization of the curve to split it into equal segments. Uses the bisection method to find the t values.

        Parameters
        ----------
        interval : list of float
            Interval of the parameter t.
        point_to_act_on : PointHomogeneous, optional
            Point to act on. Default is PointHomogeneous().
        num_segments : int, optional
            Number of segments to split the curve into. Default is 10.

        Returns
        -------
        list of float
            List of t values that split the curve into equal segments.

        Raises
        ------
        ValueError
            If the interval is not in the form [a, b] where a < b, or if the interval values are identical, or if the number of segments is less than 2.
        RuntimeError
            If scipy is not installed.
        """
        try:
            from scipy.integrate import quad  # lazy import
        except ImportError:
            raise RuntimeError("Scipy import failed. Check the package installation.")

        if interval[0] > interval[1]:
            raise ValueError("The interval must be in the form [a, b] where a < b")
        elif interval[0] == interval[1]:
            raise ValueError("The interval values are identical")
        elif num_segments < 2:
            raise ValueError("The number of segments must be greater than 1")

        t = sympy.Symbol('t')

        curve_dq = DualQuaternion(self.symbolic)
        point_path = curve_dq.act(point_to_act_on)

        dx_dt = sympy.diff(point_path[1] / point_path[0], t)
        dy_dt = sympy.diff(point_path[2] / point_path[0], t)
        dz_dt = sympy.diff(point_path[3] / point_path[0], t)

        integrand = sympy.sqrt(dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2)
        integrand_func = sympy.lambdify(t, integrand, 'numpy')

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
        Find the t value that splits the curve into a given segment length using bisection.

        Parameters
        ----------
        section_start : float
            Start of the section.
        curve_interval : list of float
            Interval of the parameter t.
        segment_length_target : float
            Target segment length.
        integrand_func : callable
            Integrand function.
        tolerance : float, optional
            Tolerance of the bisection method. Default is 1e-14.

        Returns
        -------
        float
            t value that splits the curve into the given segment length.

        Raises
        ------
        RuntimeError
            If scipy is not installed.
        """
        try:
            from scipy.integrate import quad  # lazy import
        except ImportError:
            raise RuntimeError("Scipy import failed. Check the package installation.")

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

    def study_quadric_check(self) -> numpy.ndarray:
        """
        Calculate the error of the curve from the Study quadric

        :return: coefficients error of the curve from the Study quadric
        :rtype: numpy.ndarray
        """
        poly_list = [numpy.polynomial.Polynomial(self.coeffs[i, :][::-1])
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


