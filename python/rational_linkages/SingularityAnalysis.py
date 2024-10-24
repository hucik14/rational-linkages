from .RationalMechanism import RationalMechanism
from .Linkage import LineSegment

import sympy
from itertools import combinations


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

        def sum_of_squared_determinants(matrix):
            submatrices = SingularityAnalysis.get_submatrices(matrix, 3, 3)
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

    @staticmethod
    def get_submatrices(matrix, target_rows, target_cols):
        submatrices = []
        rows, cols = matrix.shape  # Get the dimensions of the matrix

        # Check if the input matrix has enough rows and columns
        if rows < target_rows or cols < target_cols:
            raise ValueError(
                f"Input matrix must have at least {target_rows} rows and {target_cols} columns")

        # Generate all possible combinations of rows to keep (if rows > target_rows)
        for row_indices_to_keep in combinations(range(rows), target_rows):
            submatrix_rows = matrix[row_indices_to_keep,
                             :]  # Keep the specified rows

            # Generate all possible combinations of columns to keep (if cols > target_cols)
            for col_indices_to_keep in combinations(range(cols), target_cols):
                submatrix = submatrix_rows[:,
                            col_indices_to_keep]  # Keep the specified columns
                submatrices.append(submatrix)

        return submatrices
