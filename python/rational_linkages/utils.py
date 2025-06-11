# This file contains utility functions that are used in the rational_linkages package.


def dq_algebraic2vector(ugly_expression: list) -> list:
    """
    Convert an algebraic expression to a vector.

    Converts an algebraic equation in terms of i, j, k, epsilon to an 8-vector
    representation with coefficients [p0, p1, p2, p3, p4, p5, p6, p7].

    :param list ugly_expression: An algebraic equation in terms of i, j, k, epsilon.

    :return: 8-vector representation of the algebraic equation
    :rtype: list
    """
    from sympy import symbols, expand  # inner import
    i, j, k, epsilon = symbols('i j k epsilon')

    expr = expand(ugly_expression)

    basis = [0, i, j, k]

    primal = expr.coeff(epsilon, 0)
    dual = expr.coeff(epsilon)

    primal_coeffs = [primal.coeff(b) for b in basis]
    dual_coeffs = [dual.coeff(b) for b in basis]

    return primal_coeffs + dual_coeffs

def extract_coeffs(expr, var, deg: int, expand: bool = True):
    """
    Extracts the coefficients of a polynomial expression.

    :param sympy.Expr expr: Polynomial expression.
    :param sympy.Symbol var: Variable to extract coefficients with respect to.
    :param int deg: Degree of the polynomial.
    :param bool expand: Expand the expression before extracting coefficients.

    :return: List of coefficients of the polynomial.
    :rtype: list
    """
    if expand:
        from sympy import expand  # inner import
        expr = expand(expr)
    return [expr.coeff(var, i) for i in range(deg, -1, -1)]

def color_rgba(color: str, transparency: float = 1.0) -> tuple:
    """
    Convert a common color name to RGB tuple.

    :param str color: color name or shortcut
    :param float transparency: transparency value

    :return: RGBA color scheme
    :rtype: tuple
    """
    color_map = {
        'red': (1, 0, 0),
        'r': (1, 0, 0),
        'green': (0, 1, 0),
        'g': (0, 1, 0),
        'blue': (0, 0, 1),
        'b': (0, 0, 1),
        'yellow': (1, 1, 0),
        'y': (1, 1, 0),
        'cyan': (0, 1, 1),
        'c': (0, 1, 1),
        'magenta': (1, 0, 1),
        'm': (1, 0, 1),
        'black': (0, 0, 0),
        'k': (0, 0, 0),
        'white': (1, 1, 1),
        'w': (1, 1, 1),
        'orange': (1, 0.5, 0),
        'purple': (0.5, 0, 0.5),
        'pink': (1, 0.75, 0.8),
        'brown': (0.65, 0.16, 0.16),
        'gray': (0.5, 0.5, 0.5),
        'grey': (0.5, 0.5, 0.5)
    }
    return (*color_map.get(color, (1, 0, 0)), transparency)

def sum_of_squares(list_of_values: list) -> float:
    """
    Calculate the sum of squares of values in given list.

    :param list list_of_values: List of values.

    :return: Sum of squares of the values.
    :rtype: float
    """
    return sum([value**2 for value in list_of_values])


def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed.
    """
    from importlib.metadata import distribution

    try:
        distribution(package_name)
        return True
    except ImportError:
        return False
