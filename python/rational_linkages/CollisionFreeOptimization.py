from itertools import product

import numpy as np

from .NormalizedLine import NormalizedLine
from .RationalMechanism import RationalMechanism


class CollisionFreeOptimization:
    """Optimization helpers for finding collision-free full-cycle designs.

    The class contains utilities to estimate an initial configuration for the
    mechanism (smallest polyline) and to run higher-level optimization
    routines (for example combinatorial search).
    """
    def __init__(self, mechanism: RationalMechanism):
        """Create a new optimizer for a mechanism.

        Parameters
        ----------
        mechanism
            The :class:`RationalMechanism` instance to optimize.
        """
        self.mechanism = mechanism

    def smallest_polyline(self) -> tuple:
        """Find points on mechanism axes that form the smallest polyline.

        The routine represents each mechanism axis as a normalized line and
        chooses one point on each axis such that the closed polyline connecting
        these points has minimal total Euclidean length. The optimization is
        performed with ``scipy.optimize.minimize``.

        Returns
        -------
        tuple
            ``(points, points_params, result)`` where ``points`` is a list of
            3D points on the axes, ``points_params`` are the parameter values
            used to generate these points (duplicated per joint connection
            point), and ``result`` is the optimizer result object.
        """
        try:
            from scipy.optimize import minimize  # lazy import
        except ImportError:
            raise RuntimeError("Scipy import failed. Check its installation.")

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
        """Run a high-level optimization to obtain collision-free design params.

        Parameters
        ----------
        method
            Name of the optimization method to run (e.g. ``'combinatorial_search'``).
        step_length
            Step length used by the chosen optimization method (see the
            combinatorial search documentation for details).
        min_joint_segment_length
            Minimum allowed length for joint segments during the search.
        max_iters
            Maximum number of iterations to consider.
        **kwargs
            Additional keyword arguments forwarded to the chosen method.

        Returns
        -------
        list
            Parameters for a collision-free design found by the optimizer.
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
    """Combinatorial search for collision-free linkages.

    The algorithm follows the approach in the referenced work
    (:footcite:`Li2020`) and performs an enumerative search over discrete
    shift values for links and joint connection points to find collision-free
    mechanism designs.
    """
    def __init__(self,
                 mechanism: RationalMechanism,
                 linkage_length: float,
                 step_length: float = 10.0,
                 min_joint_segment_length: float = 0.001,
                 max_iters: int = 10):
        """Create a combinatorial search instance.

        Parameters
        ----------
        mechanism
            The :class:`RationalMechanism` to optimize.
        linkage_length
            Total linkage length used to compute shift values.
        step_length, optional
            Step-length divisor used to scale shift increments (default 10.0).
        min_joint_segment_length, optional
            Minimum allowable length for joint segments (default 0.001).
        max_iters, optional
            Maximum number of iterations (default 10).
        """
        self.mechanism = mechanism
        self.linkage_length = linkage_length
        self.step_length = step_length
        self.min_joint_segment_length = min_joint_segment_length
        self.max_iters = max_iters + 1

    def combinatorial_search(self, **kwargs):
        """Run the top-level combinatorial search.

        The search iterates over increasing shift magnitudes and first attempts
        to find collision-free link-only configurations before searching the
        full mechanism including joint segments.

        Parameters
        ----------
        **kwargs
            Optional control parameters (for example ``start_iteration`` and
            ``end_iteration`` can be provided). See method usage in the code.

        Returns
        -------
        list or None
            Collision-free point parameters or ``None`` if no solution was
            found.
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
        """Search for a collision-free link-only configuration.

        Parameters
        ----------
        iteration
            Iteration index used to scale the shift magnitude.
        combinations, optional
            Precomputed combinations of discrete shifts to test. If
            ``None``, combinations are generated internally.

        Returns
        -------
        list or None
            Collision-free points parameters if a solution is found,
            otherwise ``None``.
        """
        shift_val = iteration * self.linkage_length / self.step_length

        if combinations is None:
            combs = self._get_combinations_sequences(joints=False)
        else:
            combs = combinations

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
        """Search for a full collision-free mechanism design including joints.

        Parameters
        ----------
        coll_free_links_params
            Collision-free points parameters for links (as returned by
            :meth:`search_links`). These parameters are doubled and adjusted
            to represent joint connection point pairs.
        combinations, optional
            Precomputed joint-shift combinations to try. If ``None``, these
            are generated internally.

        Returns
        -------
        list or None
            Collision-free point parameters for the full mechanism, or
            ``None`` if no configuration could be found.
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
        """Generate discrete combination sequences used by the search.

        Parameters
        ----------
        joints, optional
            If False, generate combinations for link-only shifts; if True,
            generate combinations for joint-segment shifts.

        Returns
        -------
        list
            A list of tuples representing the discrete sequences to enumerate
            during the search.
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
