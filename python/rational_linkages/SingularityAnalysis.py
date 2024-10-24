from .RationalMechanism import RationalMechanism
from .Linkage import LineSegment

import sympy
from itertools import combinations


"""
Example

from rational_linkages import (RationalMechanism, TransfMatrix, MotionInterpolation,
                               SingularityAnalysis, Plotter)

import sympy as sp

# define poses as input for Bennett synthesis
p0 = TransfMatrix()
p1 = TransfMatrix.from_vectors(approach_z=[-0.0362862, 0.400074, 0.915764],
                               normal_x=[0.988751, -0.118680, 0.0910266],
                               origin=[0.033635718, 0.09436004, 0.03428654])
p2 = TransfMatrix.from_vectors(approach_z=[-0.0463679, -0.445622, 0.894020],
                               normal_x=[0.985161, 0.127655, 0.114724],
                               origin=[-0.052857769, -0.04463076, -0.081766])

poses = [p0, p1, p2]

# construct C(t) from poses
c = MotionInterpolation.interpolate(poses)

# factorize C(t) and obtain mechanism
m = RationalMechanism(c.factorize())
s = SingularityAnalysis()
j = s.check_singularity(m)

t = sp.Symbol('t')
js = sp.simplify(j)
# jp = sp.Poly(js, t)
sp.plot(js, (t, 0.2, 0.7))


# p = Plotter(interactive=True, steps=500, arrows_length=0.05, joint_range_lim=0.1)
# p.plot(m, show_tool=True)
# p.plot(p0, label=r'$\mathbf{p}_0$ (origin)')
# p.plot(p1, label=r'$\mathbf{p}_1$')
# p.plot(p2, label=r'$\mathbf{p}_2$')
# p.show()
"""


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
