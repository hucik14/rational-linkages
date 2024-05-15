from .RationalMechanism import RationalMechanism
from .Linkage import LineSegment

import sympy
import numpy as np


class SingularityAnalysis:
    """
    Singularity analysis algorithm of collision-free linkages by :footcite:t:`Li2020`.
    """
    def __init__(self):
        pass

    def check_singularity(self, mechanism: RationalMechanism):
        """
        Check for singularity in the mechanism.

        :param RationalMechanism mechanism: The mechanism to check for singularity
        """
        # check for singularity
        jacobian = self.get_jacobian(mechanism.segments)

        def get_submatrices(matrix):
            submatrices = []
            for row_to_remove in range(matrix.rows):
                for col_to_remove in range(matrix.cols):
                    # Create a submatrix by removing one row and one column
                    submatrix = matrix.minor_submatrix(row_to_remove, col_to_remove)
                    submatrices.append(submatrix)
            return submatrices

        def sum_of_squared_determinants(matrix):
            submatrices = get_submatrices(matrix)[0:2]
            return sum(submatrix.det() ** 2 for submatrix in submatrices)

        sum_det = sum_of_squared_determinants(jacobian)

        t = sympy.Symbol('t')
        p = sum_det

        first_derivative = sympy.diff(p, t)
        first_derivative = sympy.simplify(first_derivative)

        f = sympy.lambdify(t, first_derivative, 'numpy')
        #coeffs = sympy.Poly(first_derivative).all_coeffs()
        #print("Coeffs:", coeffs)

        t_space = np.linspace(-1, 1, 10)

        y = f(t_space)
        print("Y:", y)

        second_derivative = sympy.diff(first_derivative, t)
        # Find the critical points
        critical_points = sympy.nsolve(first_derivative, t)


        # Determine which critical points are local minima
        local_minima = [point for point in critical_points if
                        second_derivative.subs(t, point).evalf() > 0]

        print("Local minima:", local_minima)
        return local_minima

    def get_jacobian(self, segments: list[LineSegment]):
        """
        Get the algebraic Jacobian matrix of the mechanism.

        :param list[LineSegment] segments: The line segments of the mechanism.
        """
        algebraic_plucker_coords = [joint.equation
                                    for joint in segments if joint.type == 'j']

        # normalization



        jacobian = sympy.Matrix.zeros(6, len(algebraic_plucker_coords))
        for i, plucker_line in enumerate(algebraic_plucker_coords):
            jacobian[:, i] = [eq for eq in plucker_line.screw]

        return jacobian



