from .RationalMechanism import RationalMechanism
from .NormalizedLine import NormalizedLine

import numpy as np
from itertools import product


class CollisionFreeOptimization:
    """
    Class for the optimization of the mechanism for full-cycle collision-free design.
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
        if len(self.mechanism.factorizations) == 1:
            dq_lines = self.mechanism.factorizations[0].dq_axes
        else:
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
        points_params = [np.array([param, param]) for param in points_params]

        points = [line.point_on_line(float(points_params[i][0]))
                  for i, line in enumerate(lines)]

        return points, points_params, result

    def optimize(self,
                 method: str,
                 step_length: float,
                 min_joint_segment_length: float,
                 max_iters: int,
                 **kwargs):
        """
        Optimize the mechanism for collision-free operation.

        :param method: optimization method
        :param step_length: length of the step, i.e. the shift distance value, see
            :ref:`combinatorial_search` for more detail
        :param min_joint_segment_length: minimum length of the joint segment
        :param max_iters: maximum number of iterations
        :param kwargs: additional keyword arguments
        """
        # initial estimation - the smallest polyline
        points, points_params, result = self.smallest_polyline()

        # update the design of the mechanism - set initial design
        self.mechanism.factorizations[0].set_joint_connection_points_by_parameters(
            points_params[:len(self.mechanism.factorizations[0].dq_axes)])
        self.mechanism.factorizations[1].set_joint_connection_points_by_parameters(
            points_params[len(self.mechanism.factorizations[1].dq_axes):][::-1])

        if method == 'combinatorial_search':
            print("Starting combinatorial search algorithm...")
            cs = CombinatorialSearch(self.mechanism,
                                     linkage_length=result.fun,
                                     step_length=step_length,
                                     min_joint_segment_length=min_joint_segment_length,
                                     max_iters=max_iters)
            coll_free_points_params = cs.combinatorial_search(**kwargs)

        return coll_free_points_params


class CombinatorialSearch:
    """
    Combinatorial search algorithm of collision-free linkages.

    Algorithm by :footcite:t:`Li2020`.
    """
    def __init__(self,
                 mechanism: RationalMechanism,
                 linkage_length: float,
                 step_length: float = 10.0,  # TODO step length estimation
                 min_joint_segment_length: float = 0.001,
                 max_iters: int = 10):
        """
        Initialize the combinatorial search algorithm.

        :param RationalMechanism mechanism: The mechanism to optimize.
        :param float linkage_length: length of the linkage
        :param float step_length: length of the step, i.e. the shift distance value, see
            :ref:`combinatorial_search` for more detail
        :param float min_joint_segment_length: minimum length of the joint segment
        :param int max_iters: maximum number of iterations
        """
        self.mechanism = mechanism
        self.linkage_length = linkage_length
        self.step_length = step_length
        self.min_joint_segment_length = min_joint_segment_length
        self.max_iters = max_iters + 1

    def combinatorial_search(self, **kwargs):
        """
        Perform collision-free combinatorial search method.

        :return: list of collision-free points parameters
        :rtype: list
        """

        iter_start = kwargs.get('start_iteration', 1)
        iter_end = kwargs.get('end_iteration', self.max_iters)
        comb_links = kwargs.get('combinations_links', None)
        comb_joints = kwargs.get('combinations_joints', None)

        if comb_links is None:
            # check design for collisions
            init_collisions = self.mechanism.collision_check(only_links=False,
                                                             terminate_on_first=True)
        else:
            # skip initial collision check if combinations are provided
            init_collisions = []

        if init_collisions is not None:
            for i in range(iter_start, iter_end):
                coll_free_links_params = self.search_links(i, combinations=comb_links)

                if coll_free_links_params is not None:
                    print("")
                    print("Collision-free solution for links found, "
                          "starting joint search...")
                    coll_free_params = self.search_mechanism(coll_free_links_params,
                                                             combinations=comb_joints)

                    if coll_free_params is not None:
                        print("Search was successful, collision-free solution found.")
                        return coll_free_params
        else:
            print("Search was unsuccessful, collisions found.")
            return None

    def search_links(self, iteration: int, combinations: list = None):
        """
        Search for the solution of the combinatorial search algorithm, links only.

        Searches for the smallest polyline that is collision free (only links).

        :param iteration: iteration index
        :param list combinations: list of combinations to search links

        :return: list of collision-free points parameters
        :rtype: list
        """
        shift_val = iteration * self.linkage_length / self.step_length

        if combinations is None:
            combs = self._get_combinations_sequences(joints=False)
        else:
            combs = combinations

        # TODO: parallelize the search
        for i, sequence in enumerate(combs):
            print("--- iteration: {}, shift_value: {}, sequence {} of {}: {}"
                  .format(iteration, shift_val, i + 1, len(combs), sequence))
            points_params = shift_val * np.asarray(sequence)
            points_params = [[param] for param in points_params]

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

    def search_mechanism(self, coll_free_links_params: list, combinations: list = None):
        """
        Search for the solution of the combinatorial search algorithm, including joints.

        Searches for the mechanism that is collision free (including joint segments).

        :param list coll_free_links_params: list of collision-free points parameters
        :param list combinations: list of combinations to search mechanism design

        :return: list of collision-free points parameters
        :rtype: list
        """
        shift_val = 0.5 * self.min_joint_segment_length

        if combinations is None:
            combs = self._get_combinations_sequences(joints=True)
        else:
            combs = combinations

        coll_free_links_params = [item * 2 for item in coll_free_links_params]

        for i, sequence in enumerate(combs):
            print("--- joint search. Shift_value: {}, sequence {} of {}: {}"
                  .format(shift_val, i + 1, len(combs), sequence))
            shift_seq = shift_val * np.asarray(sequence)

            points_params = [[params[0] + shift_seq[ii * 2],
                              params[1] + shift_seq[ii * 2 + 1]]
                             for ii, params in enumerate(coll_free_links_params)]

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

    def _get_combinations_sequences(self, joints: bool = False):
        """
        Get all combinations of the joint angles and shuffle them.

        :param bool joints: True if joints segments are searched for, False otherwise

        :return: list of all combinations of joint angles
        :rtype: list
        """
        if not joints:
            elements = [0, 1, -1]
            combs = list(product(elements, repeat=self.mechanism.num_joints))

            # remove the combination of all zeros, which was already tested
            if 0 in elements:
                combs.remove((0,) * self.mechanism.num_joints)

            return combs

        else:
            elements = [-1, 1]
            combs = list(product(elements, repeat=self.mechanism.num_joints * 2))

            # filter out the combs that have the same value for the 2 connection pts
            combs = [x for x in combs
                     if all(x[i] != x[i + 1] for i in range(0, len(x) - 1, 2))]

            # place the most promising combs first
            def generate_tuple(n):
                return tuple(-1 if i % 2 == 0 else 1 for i in range(n))

            tup = (generate_tuple(self.mechanism.num_joints)
                   + generate_tuple(self.mechanism.num_joints)[::-1])
            tup2 = (generate_tuple(self.mechanism.num_joints)[::-1]
                    + generate_tuple(self.mechanism.num_joints))

            combs.remove(tup)
            combs.insert(0, tup)
            combs.remove(tup2)
            combs.insert(0, tup2)

            return combs
