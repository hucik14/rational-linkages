from copy import deepcopy

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from PointHomogeneous import PointHomogeneous


MotionFactorization = "MotionFactorization"

class RationalCurve:
    """
    Class representing rational curves in n-dimensional space, where the first row is
    homogeneous coordinate equation.

    This class allows you to work with rational curves defined by parametric equations.

    Attributes:
        coeffs (np.array): Coefficients of parametric equations of the curve.
        dimension (int): The dimension of the curve, excluding the homogeneous
        coordinate.
        degree (int): The degree of the curve.
        symbolic (list): Symbolic expressions for the parametric equations of the curve.
        set_of_polynomials (list): A set of polynomials representing the curve.

    Args:
        polynomials (list): A list of polynomials representing the curve.

    Example:
        Limancon of Pascal:
        >>> a = 1
        >>> b = 0.5
        >>> t = sp.Symbol('t')
        >>> eq0 = sp.Poly((1+t**2)**2, t, domain='QQ')
        >>> eq1 = sp.Poly(b*(1-t**2)*(1+t**2) + a*(1-t**2)**2, t, domain='QQ')
        >>> eq2 = sp.Poly(2*b*t*(1+t**2) + 2*a*t*(1-t**2), t, domain='QQ')
        >>> curve = RationalCurve([eq0, eq1, eq2, eq0])

        or from polynomials:
        >>> obj = RationalCurve.from_coeffs(np.array([[1., 0., 2., 0., 1.], [0.5, 0., -2., 0., 1.5], [0., -1., 0., 3., 0.], [1., 0., 2., 0., 1.]]))
    """

    def __init__(self, polynomials: list[sp.Poly]):
        """
        Initializes a RationalCurve object with the provided coefficients.

        :param polynomials: list of polynomial equations of the curve
        """
        self.set_of_polynomials = polynomials

        self.dimension = len(self.set_of_polynomials) - 1
        # Get the degree of the curve
        self.degree = 1
        for i in range(len(polynomials)):
            self.degree = max(self.degree, self.set_of_polynomials[i].degree())

        self.coeffs = self.get_coeffs()
        self.symbolic, _ = self.get_symbolic_expressions(self.coeffs)

        self.coeffs_inversed = self.inverse_coeffs()
        self.symbolic_inversed, self.set_of_polynomials_inversed = self.get_symbolic_expressions(self.coeffs_inversed)

    @classmethod
    def from_coeffs(cls, coeffs: np.ndarray) -> "RationalCurve":
        """
        Construct rational curve from coefficients

        :coeffs: np.ndarray - coefficients of the curve

        :returns: RationalCurve
        """
        _, polynomials = cls.get_symbolic_expressions(coeffs)
        return cls(polynomials)

    @staticmethod
    def get_symbolic_expressions(coeffs: np.ndarray) -> tuple[list, list[sp.Poly]]:
        """
        Add a symbolic variable to the matrix of coefficients that describes the curve

        :param coeffs: np.ndarray - coefficients of the curve

        :return: tuple of symbolic expressions list and list of sympy polynomials
        """
        symbolic_expressions = []
        polynomials = []
        t = sp.Symbol("t")

        dimension = len(coeffs) - 1

        for i in range(dimension + 1):
            # Extract coefficients from the current row of the coefficient matrix and
            # create symbolic expressions
            row_coefficients = reversed(coeffs[i, :])
            symbolic_row_coeffs = [
                coefficient * t**j for j, coefficient in enumerate(row_coefficients)
            ]
            symbolic_expressions.append(sum(symbolic_row_coeffs))
            polynomials.append(sp.Poly(symbolic_expressions[i], t, domain="QQ"))

        return symbolic_expressions, polynomials

    def get_coeffs(self) -> np.ndarray:
        """
        Get the coefficients of the symbolic polynomial equations

        :param polynomials: list of sympy polynomials

        :return: np.array of coefficients
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
                if not self.set_of_polynomials[i].all_coeffs() == [
                    0
                ]:  # if the equation is not zero, fill the coeffs
                    degree_of_eq = self.set_of_polynomials[i].degree()
                    coeffs[i, self.degree - degree_of_eq :] = np.array(
                        self.set_of_polynomials[i].all_coeffs()
                    )
        return coeffs

    def __repr__(self):
        return f"RationalCurve({self.symbolic})"

    def curve2bezier(self, reparametrization: bool = False) -> list[PointHomogeneous]:
        """
        Convert a curve to a Bezier curve using the Bernstein polynomials

        :param reparametrization: bool - if True, the curve is mapped to the [-1,1]

        :return: list of Bezier control points
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
            sp.Poly((bernstein_basis[i] - self.symbolic[i]), t).all_coeffs()
            for i in range(self.dimension + 1)
        ]
        # Flatten the list
        equations_coeffs = [coeff for coeffs in equations_coeffs for coeff in coeffs]

        # Solve the equations
        points_sol = sp.linsolve(equations_coeffs, points_flattened)
        # Convert the solutions to numpy arrays (get points)
        points_array = np.array(points_sol.args[0]).reshape(
            self.degree + 1, self.dimension + 1
        )
        points_objects = [PointHomogeneous()] * (self.degree + 1)
        for i in range(self.degree + 1):
            points_objects[i] = PointHomogeneous(points_array[i, :])

        return points_objects

    def get_bernstein_polynomial_equations(
        self, t_var: sp.Symbol, reparametrization: bool = False, degree: int = None
    ) -> list:
        """
        Generate the Bernstein polynomial equation

        :param t_var: symbolic variable
        :param reparametrization: a function that maps the interval
        :param degree: int - degree of the polynomial, if None (not specified),
            the degree of the curve is used

        :return: list of symbolic expressions
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
            polynomial_expr = sp.binomial(degree, i) * t**i * (1 - t) ** (degree - i)
            expr.append(sp.simplify(polynomial_expr))

        return expr

    def inverse_coeffs(self) -> np.array:
        """
        Get the coefficients of the inverse curve

        :return: np.array of inversed coefficients
        """
        inverse_coeffs = np.zeros((self.dimension + 1, self.degree + 1))
        for i in range(self.dimension + 1):
            inverse_coeffs[i, :] = self.coeffs[i, :][::-1]

        return inverse_coeffs

    def inverse_curve(self) -> "RationalCurve":
        """
        Get the inverse curve

        :return: RationalCurve
        """
        return RationalCurve.from_coeffs(self.inverse_coeffs())

    def extract_expressions(self) -> list:
        """
        Extract the expressions of the curve

        :return: list of expressions of the curve (avoiding sp.Poly class)
        :rtype: list
        """
        return [self.set_of_polynomials[i].expr
                for i in range(len(self.set_of_polynomials))]

    def evaluate(self, t_param, inverted_part: bool = False) -> np.ndarray:
        """
        Evaluate the curve for given t and return in the form of dual quaternion vector

        :param t_param: float - parameter of the motion curve
        :param inverted_part: bool - if True, return the inverted part of the curve

        :return: pose of the curve as a dual quaternion vector
        """
        t = sp.Symbol("t")
        if inverted_part:
            return np.array(
                [
                    self.set_of_polynomials_inversed[i].subs(t, t_param)
                    for i in range(len(self.set_of_polynomials_inversed))
                ],
                dtype="float64",
            )
        else:
            return np.array(
                [
                    self.set_of_polynomials[i].subs(t, t_param)
                    for i in range(len(self.set_of_polynomials))
                ],
                dtype="float64",
            )

    def evaluate_as_matrix(self, t_param, inverted_part: bool = False) -> np.ndarray:
        """
        Evaluate the curve for given t and return in the form of a transformation matrix

        :param t_param: float - parameter of the motion curve
        :param inverted_part: bool - if True, return the inverted part of the curve

        :return: pose of the curve as a matrix
        """
        from DualQuaternion import DualQuaternion

        dq = DualQuaternion(self.evaluate(t_param, inverted_part))
        return dq.dq2matrix()

    def factorize(self) -> list[MotionFactorization]:
        """
        Factorize the curve into motion factorizations

        :return: list of MotionFactorization objects
        """
        from FactorizationProvider import FactorizationProvider

        factorization_provider = FactorizationProvider()
        return factorization_provider.factorize_motion_curve(self)

    def plot(self, interval=(-1, 1), steps=50, ax=None, line_style=None) -> plt.axes:
        """
        Plot the curve in 2D or 3D, based on the dimension of the curve

        :param interval: tuple - interval of the parameter t
        :param steps: int - number of steps in the interval
        :param ax: existing matplotlib axis
        :param line_style: str - line style of the plot

        :return: matplotlib axis
        """
        # check dimension
        if not (self.dimension == 2 or self.dimension == 3):
            raise ValueError("Cannot plot curves in more than 3 dimensions")
        else:
            t = sp.Symbol("t")

            # make a copy of the polynomials and append the homogeneous coordinate
            # to the Z-equation place in the list
            # if in 2D, so later z = 1
            polynoms = deepcopy(self.set_of_polynomials)
            if self.dimension == 2:
                polynoms.append(sp.Poly(polynoms[0], t, domain="QQ"))

            # plot the curve
            t_space = np.linspace(interval[0], interval[1], steps)
            curve_points = [PointHomogeneous()] * steps
            for i in range(steps):
                curve_points[i] = PointHomogeneous(
                    np.array(
                        [
                            polynoms[0].subs(t, t_space[i]),
                            polynoms[1].subs(t, t_space[i]),
                            polynoms[2].subs(t, t_space[i]),
                            polynoms[3].subs(t, t_space[i]),
                        ]
                    )
                )

            if ax is None:
                ax = plt.figure().add_subplot(projection="3d")
            else:
                ax = ax
            x, y, z = zip(*[curve_points[i].normalized_in_3d() for i in range(steps)])

            if line_style is None:
                ax.plot(x, y, z)
            else:
                ax.plot(x, y, z, line_style)

        return ax

    def get_plot_data(self, interval: tuple = (0, 1), steps: int = 50) -> (
            tuple)[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the data to plot the curve in 3D

        :param interval: interval of the parameter t
        :param steps: number of numerical steps in the interval

        :return: tuple of np.ndarray - (x, y, z) coordinates of the curve
        """
        t = sp.Symbol("t")
        t_space = np.linspace(interval[0], interval[1], steps)

        # make a copy of the polynomials and append the homogeneous coordinate
        # to the Z-equation place in the list if in 2D, so later z = 1
        polynoms = deepcopy(self.set_of_polynomials)
        if self.dimension == 2:
            polynoms.append(sp.Poly(polynoms[0], t, domain="QQ"))

        # plot the curve
        curve_points = [PointHomogeneous()] * steps
        for i in range(steps):
            point = self.evaluate(t_space[i])
            curve_points[i] = PointHomogeneous([point[0], point[-3], point[-2], point[-1]])
        x, y, z = zip(*[curve_points[i].normalized_in_3d() for i in range(steps)])
        return x, y, z

