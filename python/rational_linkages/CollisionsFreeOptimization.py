from .RationalMechanism import RationalMechanism
from .NormalizedLine import NormalizedLine

import numpy as np
from itertools import product


class CollisionFreeOptimization:
    """

    """
    def __init__(self, mechanism: RationalMechanism):
        """
        Initialize the combinatorial search algorithm.

        :param RationalMechanism mechanism: The mechanism to optimize.
        """
        self.mechanism = mechanism

    def smallest_polyline(self) -> tuple:
        """
        Get points on mechanism axes that form the smallest polyline.

        This method calculates the smallest polyline that can be formed by points on
        the mechanism axes. It uses scipy's minimize function to find the points on
        the axes that minimize the total distance of the polyline.

        :return: points on the mechanism axes that form the smallest polyline,
            parameters of the points, result of the optimization
        :rtype: tuple
        """
        from scipy.optimize import minimize

        # get the axes represented as normalized lines
        dq_lines = (self.mechanism.factorizations[0].dq_axes
                    + self.mechanism.factorizations[1].dq_axes[::-1])
        lines = [NormalizedLine.from_dual_quaternion(dq_line) for dq_line in dq_lines]

        def objective_function(x):
            p = [line.point_on_line(x[i]) for i, line in enumerate(lines)]

            total_distance = sum(
                np.linalg.norm(p[i] - p[i + 1])
                for i in range(self.mechanism.num_joints - 1))
            # Add distance between last and first point
            total_distance += np.linalg.norm(p[-1] - p[0])
            return total_distance

        x_init = np.zeros(self.mechanism.num_joints)
        result = minimize(objective_function, x_init)

        # double the parameters for the two joint connection points
        points_params = result.x

        points = [line.point_on_line(points_params[i]) for i, line in enumerate(lines)]

        return points, points_params, result

    def optimize(self, method: str, max_iters: int):
        """
        Optimize the mechanism for collision-free operation.

        :param method: optimization method
        :param max_iters: maximum number of iterations
        """
        # initial estimation - the smallest polyline
        points, points_params, result = self.smallest_polyline()

        # update the design of the mechanism - set initial design
        self.mechanism.factorizations[0].set_joint_connection_points_by_parameters(
            points_params[:len(self.mechanism.factorizations[0].dq_axes)])
        self.mechanism.factorizations[1].set_joint_connection_points_by_parameters(
            points_params[len(self.mechanism.factorizations[1].dq_axes):][::-1])

        if method == 'combinatorial_search':
            cs = CombinatorialSearch(self.mechanism,
                                     linkage_length=result.fun,
                                     step_length=10,
                                     max_iters=max_iters)
            coll_free_points_params = cs.combinatorial_search()


class CombinatorialSearch:
    """
    Combinatorial search algorithm of collision-free linkages.

    Algorithm by :footcite:t:`Li2020`.
    """
    def __init__(self,
                 mechanism: RationalMechanism,
                 linkage_length: float,
                 step_length: float = 10.0,  # TODO step length estimation
                 max_iters: int = 10):
        """
        Initialize the combinatorial search algorithm.

        :param RationalMechanism mechanism: The mechanism to optimize.
        :param float linkage_length: length of the linkage
        :param float step_length: length of the step, i.e. the shift distance value, see
            :ref:`combinatorial_search` for more detail
        :param int max_iters: maximum number of iterations
        """
        self.mechanism = mechanism
        self.linkage_length = linkage_length
        self.step_length = step_length
        self.max_iters = max_iters + 1

    def combinatorial_search(self):
        """
        Perform collision-free combinatorial search method.

        :return: list of collision-free points parameters
        :rtype: list
        """

        # check design for collisions
        #init_collisions = self.collision_check(only_links=True, terminate_on_first=True)
        init_collisions = 'test'
        self.step_length = 25

        if init_collisions is not None:
            for i in range(10, self.max_iters):
                coll_free_points_params = self.search_links(i)

                if coll_free_points_params is not None:
                    print("Collision-free for links found, starting joint search...")
                    for i in range(1, self.max_iters):
                        coll_free_points_params = self.search_mechanism(i)

                        if coll_free_points_params is not None:
                            return coll_free_points_params
        else:
            print("No collision-free solution found.")
            return None

    def search_links(self, iteration: int):
        """
        Search for the solution of the combinatorial search algorithm, links only.

        Searches for the smallest polyline that is collision free (only links).

        :param iteration: iteration index
        """
        shift_val = iteration * self.linkage_length / self.step_length

        combs = self._get_combinations_sequences([0, 1, -1])
        combs = [(-1, -1, 0, 0, 0, 1),
                 (1, 1, 0, 0, -1, -1),
                 (1, 0, 1, 0, -1, 0),
                 (1, 0, -1, 1, -1, 0)]

        for i, sequence in enumerate(combs):
            print("Iteration: {}, shift_value: {}, sequence {} of {}: {}"
                  .format(iteration, shift_val, i, len(combs), sequence))
            points_params = shift_val * np.asarray(sequence)

            # update the design of the mechanism
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

    def search_mechanism(self, iteration: int, shift_val: float = 0.001):
        """
        Search for the solution of the combinatorial search algorithm, including joints.

        Searches for the mechanism that is collision free (including joint segments).
        """
        combs = self._get_combinations_sequences([1, -1])
        combs = [(-1, -1, 1, 1, -1, 1),
                 (-1, 1, -1, -1, -1, 1)]

        for i, sequence in enumerate(combs):
            print("Iteration: {}, shift_value: {}, sequence {} of {}: {}"
                  .format(iteration, shift_val, i, len(combs), sequence))
            points_params = shift_val * np.asarray(sequence)
            points_params = [[param, param * -1] for param in points_params]

            # update the design of the mechanism
            self.mechanism.factorizations[0].set_joint_connection_points_by_parameters(
                points_params[:len(self.mechanism.factorizations[0].dq_axes)])
            self.mechanism.factorizations[1].set_joint_connection_points_by_parameters(
                points_params[len(self.mechanism.factorizations[1].dq_axes):][::-1])

            colls = self.mechanism.collision_check(only_links=False,
                                                   terminate_on_first=True)

            if colls is None:
                return points_params

        print("No collision-free solution found for iteration: {}".format(iter))
        return None

    def _get_combinations_sequences(self, elements: list):
        """
        Get all combinations of the joint angles and shuffle them.

        :param list elements: list of elements to combine

        :return: list of all combinations of joint angles
        :rtype: list
        """
        combs = list(product(elements, repeat=self.mechanism.num_joints))

        # remove the combination of all zeros, which was already tested
        if 0 in elements:
            combs.remove((0,) * self.mechanism.num_joints)

        return combs
