import numpy as np
import sympy as sp

from .RationalMechanism import RationalMechanism
from .Linkage import LineSegment

from itertools import product, combinations
from random import shuffle


class CombinatorialSearch:
    """
    Combinatorial search algorithm of collision-free linkages by :footcite:t:`Li2020`.
    """
    def __init__(self,
                 mechanism: RationalMechanism,
                 linkage_lenght: float,
                 step_length: float,
                 ):
        self.mechanism = mechanism
        self.linkage_lenght = linkage_lenght
        self.step_length = step_length
        self.sequences = self._get_combinations_sequences()

    def search_links(self, iteration: int):
        """
        Search for the solution of the combinatorial search algorithm, links only.

        Searches for the smallest polyline that is collision free (only links).

        :param iteration: iteration index
        """
        shift_val = iteration * self.linkage_lenght / self.step_length

        for i, sequence in enumerate(self.sequences):
            print("Iteration: {}, shift_value: {}, sequence {} of {}: {}"
                  .format(iteration, shift_val, i, len(self.sequences), sequence))
            points_params = shift_val * np.asarray(sequence)
            # update the design of the mechanism - set initial design
            self.mechanism.factorizations[0].set_joint_connection_points_by_parameters(
                points_params[:len(self.mechanism.factorizations[0].dq_axes)])
            self.mechanism.factorizations[1].set_joint_connection_points_by_parameters(
                points_params[len(self.mechanism.factorizations[1].dq_axes):][::-1])

            colls = self.mechanism.collision_check(only_links=True,
                                                   terminate_on_first=True)

            if colls is None:
                return points_params

        print("No collision-free solution found for iteration: {}".format(iter))
        return None

    def search_mechnism(self, iteration: int):
        """
        Search for the solution of the combinatorial search algorithm, including joints.

        Searches for the mechanism that is collision free (including joint segments).

        :param iteration: iteration index
        """
        pass

    def _get_combinations_sequences(self):
        """
        Get all combinations of the joint angles and shuffle them.

        :return: list of all combinations of joint angles
        :rtype: list
        """
        # TODO reduced by avoiding 0
        elements = [0, 1, -1]
        #elements = [1, -1]
        combs = list(product(elements, repeat=self.mechanism.num_joints))

        # remove the combination of all zeros, which was already tested
        #combs.remove((0,)*self.mechanism.num_joints)

        shuffle(combs)
        combs = [(-1, -1, 0, 0, 0, 1),
                 (1, 1, 0, 0, -1, -1),
                 (1, 0, 1, 0, -1, 0),
                 (1, 0, -1, 1, -1, 0)]
        return combs


class SingularityAnalysis:
    """
    Singularity analysis algorithm of collision-free linkages by :footcite:t:`Li2020`.
    """
    def __init__(self):
        pass

    def check_singluarity(self, mechanism: RationalMechanism):
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



        jacobian = sp.Matrix.zeros(6, len(algebraic_plucker_coords))
        for i, plucker_line in enumerate(algebraic_plucker_coords):
            jacobian[:, i] = plucker_line.screw

        return jacobian


