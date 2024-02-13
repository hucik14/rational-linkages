import numpy as np

from .RationalMechanism import RationalMechanism
from itertools import product
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

    def search(self, iter: int):
        """
        Search for the solution of the combinatorial search algorithm.

        :param iter: iteration index
        """
        shift_val = iter * self.linkage_lenght / self.step_length

        for i, sequence in enumerate(self.sequences):
            print("Iteration: {}, shift_value: {}, sequence {} of {}: {}".format(iter, shift_val, i, len(self.sequences), sequence))
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
        return combs



