from sympy import Matrix

from .Linkage import LineSegment
from .RationalMechanism import RationalMechanism


class SingularityAnalysis:
    """
    Perform singularity analysis for collision-free linkages.

    This class implements the singularity analysis algorithm as described in
    :footcite:t:`Li2020`.

    .. footbibliography::

    """
    def __init__(self):
        """
        Initialize the SingularityAnalysis class.
        """
        pass

    def check_singularity(self, mechanism: RationalMechanism):
        """
        Check for singularity in the given mechanism.

        Parameters
        ----------
        mechanism : RationalMechanism
            The mechanism to check for singularity.

        Returns
        -------
        sympy.Basic
            The sum of squared determinants of the Jacobian submatrices.
        """
        # check for singularity
        jacobian = self.get_jacobian(mechanism.segments)

        def get_submatrices(matrix):
            """
            Generate all submatrices by removing one row and one column.

            Parameters
            ----------
            matrix : sympy.Matrix
                The input matrix.

            Returns
            -------
            list of sympy.Matrix
                A list of submatrices.
            """
            submatrices = []
            for row_to_remove in range(matrix.rows):
                for col_to_remove in range(matrix.cols):
                    # Create a submatrix by removing one row and one column
                    submatrix = matrix.minor_submatrix(row_to_remove, col_to_remove)
                    submatrices.append(submatrix)
            return submatrices

        def sum_of_squared_determinants(matrix):
            """
            Compute the sum of squared determinants of all submatrices.

            Parameters
            ----------
            matrix : sympy.Matrix
                The input matrix.

            Returns
            -------
            sympy.Basic
                The sum of squared determinants.
            """
            submatrices = get_submatrices(matrix)
            return sum(submatrix.det() ** 2 for submatrix in submatrices)

        sum_det = sum_of_squared_determinants(jacobian)

        return sum_det

    def get_jacobian(self, segments: list[LineSegment]):
        """
        Compute the algebraic Jacobian matrix of the mechanism.

        Parameters
        ----------
        segments : list of LineSegment
            The line segments of the mechanism.

        Returns
        -------
        sympy.Matrix
            The algebraic Jacobian matrix.
        """
        algebraic_plucker_coords = [joint.equation
                                    for joint in segments if joint.type == 'j']

        # normalization


        jacobian = Matrix.zeros(6, len(algebraic_plucker_coords))
        for i, plucker_line in enumerate(algebraic_plucker_coords):
            jacobian[:, i] = plucker_line.screw

        return jacobian
