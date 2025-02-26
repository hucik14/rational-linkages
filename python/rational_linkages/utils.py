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
