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
    from sympy import expand, symbols  # lazy import
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
        from sympy import expand  # lazy import
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
    from importlib.metadata import distribution  # lazy import

    try:
        distribution(package_name)
        return True
    except ImportError:
        return False


def tr_from_dh_rationally(t_theta, di, ai, t_alpha):
    """
    Create transformation matrix from DH parameters using Sympy in rational form.

    The input shall be rational numbers, including the angles which are expected
    to be parameters of tangent half-angle substitution, i.e., t_theta = tan(theta/2)
    and t_alpha = tan(alpha/2).

    :param sp.Rational t_theta: DH parameter theta in tangent half-angle form
    :param sp.Rational di: DH parameter d, the offset along Z axis
    :param sp.Rational ai: DH parameter a, the length along X axis
    :param sp.Rational t_alpha: DH parameter alpha in tangent half-angle form

    :return: 4x4 transformation matrix
    :rtype: sp.Matrix
    """
    from sympy import Matrix, eye, Expr  # lazy import

    if not all(isinstance(param, Expr) for param in [t_theta, di, ai, t_alpha]):
        raise ValueError("All parameters must be of type sympy objects (Expr).")

    s_th = 2*t_theta / (1 + t_theta**2)
    c_th = (1 - t_theta**2) / (1 + t_theta**2)
    s_al = 2*t_alpha / (1 + t_alpha**2)
    c_al = (1 - t_alpha**2) / (1 + t_alpha**2)

    mat = eye(4)
    mat[1:4, 0] = Matrix([ai * c_th, ai * s_th, di])
    mat[1, 1:4] = Matrix([[c_th, -s_th * c_al, s_th * s_al]])
    mat[2, 1:4] = Matrix([[s_th, c_th * c_al, -c_th * s_al]])
    mat[3, 1:4] = Matrix([[0, s_al, c_al]])
    return mat


def normalized_line_rationally(point, direction):
    """
    Create a normalized Plücker line from a point and a direction using Sympy.

    The input shall be rational numbers, i.e. Sympy objects.

    :param sp.Rational point:
    :param sp.Rational direction:

    :return: 6-vector representing the Plücker line
    :rtype: sp.Matrix
    """
    from sympy import Matrix, Expr  # lazy import

    if not all(isinstance(param, Expr) for param in point + direction):
        raise ValueError("All parameters must be of type sympy objects (Expr).")

    dir = Matrix(direction)
    pt = Matrix(point)
    mom = (-1 * dir).cross(pt)
    return Matrix.vstack(dir, mom)