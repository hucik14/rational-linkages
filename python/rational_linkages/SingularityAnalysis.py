from .RationalMechanism import RationalMechanism
from .Linkage import LineSegment

import sympy


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
            submatrices = get_submatrices(matrix)
            return sum(submatrix.det() ** 2 for submatrix in submatrices)

        sum_det = sum_of_squared_determinants(jacobian)

        return sum_det

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
            jacobian[:, i] = plucker_line.screw

        return jacobian
