import numpy as np
import sympy as sp
from copy import deepcopy
from time import time

from RationalCurve import RationalCurve
from DualQuaternion import DualQuaternion
from NormalizedLine import NormalizedLine
from MotionFactorization import MotionFactorization

PointHomogeneous = 'PointHomogeneous'
TransfMatrix = 'TransfMatrix'


class RationalMechanism(RationalCurve):
    """
    Class representing rational mechanisms in dual quaternion space.
    """

    def __init__(self, factorizations: list[MotionFactorization],
                 end_effector: DualQuaternion = None):
        """
        Initializes a RationalMechanism object
        """
        super().__init__(factorizations[0].set_of_polynomials)
        self.factorizations = factorizations
        self.num_joints = sum([f.number_of_factors for f in factorizations])

        self.is_linkage = True if len(self.factorizations) == 2 else False

        self.end_effector = (
            DualQuaternion(self.evaluate(0, inverted_part=True))
            if end_effector is None else end_effector)

        if self.is_linkage:
            self.segments = self._get_line_segments_of_linkage()

    def get_dh_params(self, unit: str = 'rad', scale: float = 1.0) -> np.ndarray:
        """
        Get the standard Denavit-Hartenberg parameters of the linkage.

        The parameters are in the order: theta, d, a, alpha. It follows the standard
        convention. The first row is are the parameters of the base frame.

        See more in the paper by Huczala et al. [#huczala2022icma]_.

        :param str unit: desired unit of the angle parameters, can be 'deg' or 'rad'
        :param float scale: scale of the length parameters of the linkage

        :return: theta, d, a, alpha array of Denavit-Hartenberg parameters
        :rtype: np.ndarray

        .. [#huczala2022icma] D. Huczala, T. Kot, J. Mlotek, J. Suder and M. Pfurner,
            "An Automated Conversion Between Selected Robot Kinematic Representations,"
            2022 *10th International Conference on Control, Mechatronics and Automation
            (ICCMA)*, Belval, Luxembourg, 2022, pp. 47-52,
            DOI: 10.1109/ICCMA56665.2022.10011595
            (https://doi.org/10.1109/ICCMA56665.2022.10011595).
        """
        frames = self.get_frames()

        # closed-loop mechanism - add 1st joint at the end of the list
        frames.append(frames[1])

        dh = np.zeros((self.num_joints + 1, 4))
        for i in range(self.num_joints + 1):
            th, d, a, al = frames[i].dh_to_other_frame(frames[i+1])

            if unit == 'deg':
                th = np.rad2deg(th)
                al = np.rad2deg(al)
            elif unit != 'rad':
                raise ValueError("unit must be deg or rad")

            dh[i, :] = [th, scale * d, scale * a, al]
        return dh

    def get_frames(self) -> list[TransfMatrix]:
        """
        Get the frames of a linkage that follow standard Denaivt-Hartenberg convention.

        It renurns n+2 frames, where n is the number of joints. The first frame is the
        base frame, and the last frame is an updated frame of the first joint that
        follows the DH convention in respect to the last joint's frame.

        :return: list of TransfMatrix objects
        :rtype: list[TransfMatrix]
        """
        from TransfMatrix import TransfMatrix

        frames = [TransfMatrix()] * (self.num_joints + 2)

        screws = self.get_screw_axes()

        # add the first screw to the end of the list
        screws.append(screws[0])

        # insert origin as the base line
        screws.insert(0, NormalizedLine())

        for i, line in enumerate(screws[1:]):
            # obtain the connection points and the distance to the previous line
            pts, dist, cos_angle = line.common_perpendicular_to_other_line(screws[i])
            vec = pts[0] - pts[1]

            if not np.isclose(dist, 0.0):  # if the lines are skew or parallel
                # normalize vec - future X axis
                vec_x = vec / np.linalg.norm(vec)

                # from line.dir (future Z axis) and x create an SE3 object
                frames[i+1] = TransfMatrix.from_vectors(vec_x, line.direction, origin=pts[0])

            else:  # Z axes are intersecting or coincident
                if np.isclose(np.dot(frames[i].a, line.direction), 1):
                    # Z axes are coincident, therefore the frames are the same
                    frames[i+1] = deepcopy(frames[i])

                elif np.isclose(np.dot(frames[i].a, line.direction), -1):
                    # Z axes are coincident, but differ in orientation
                    rot_x_pi = TransfMatrix.from_rpy([np.pi, 0, 0])
                    frames[i + 1] = TransfMatrix(frames[i].matrix @ rot_x_pi.matrix)

                else:  # Z axis intersect with an angle
                    # future X axis as cross product of previous Z axis and new Z axis
                    vec_x = np.cross(frames[i].a, line.direction)

                    frames[i + 1] = TransfMatrix.from_vectors(vec_x, line.direction, origin=pts[0])

        return frames

    def get_screw_axes(self) -> list[NormalizedLine]:
        """
        Get the normalized lines (screw axes, Plucker coordinates) of the linkage.

        The lines are in home configuration. They consist of two factorizations, and
        the second factorization axes must be reversed.

        :return: list of NormalizedLine objects
        :rtype: list[NormalizedLine]
        """
        screws = []
        for axis in self.factorizations[0].dq_axes:
            screws.append(NormalizedLine(axis.dq2screw()))

        branch2 = []
        for axis in self.factorizations[1].dq_axes:
            branch2.append(NormalizedLine(axis.dq2screw()))

        return screws + branch2[::-1]

    def _get_dh_OLD(self, unit: str = "cos_alpha",
                      scale: float = 1.0, joint_length: float = None) -> tuple:
        """
        Get the Denavit-Hartenberg parameters of the linkage.

        :param str unit: - form of the returned alpha parameter, can be
            "cos_alpha", "deg", or "rad"
        :param float scale: scale of length parameters of the linkage
        :param float joint_length: length of the joint segment

        :return: tuple (d, a, alpha)
        :rtype: tuple
        """
        from NormalizedLine import NormalizedLine
        # Combine factorizations to get the linkage
        linkage = self.factorizations[0] + self.factorizations[1]
        # Create NormalizedLine objects for each axis rotation
        lines = [NormalizedLine(linkage.dq_axes[i].dq2screw())
                 for i in range(len(linkage.dq_axes))]

        d = []
        a = []
        alpha = []
        middle_points = []

        if joint_length is not None:
            joint_segment = joint_length / scale
        else:
            joint_segment = 1.0

        # Calculate Denavit-Hartenberg parameters for each pair of axis rotations
        pts_prev = lines[-1].common_perpendicular_to_other_line(lines[0])[0]
        for i in range(len(lines) - 1):
            pts, a_i, al_i = lines[i].common_perpendicular_to_other_line(lines[i+1])
            d_i = lines[i].get_point_param(pts[0]) - lines[i].get_point_param(pts_prev[1])
            pts_prev = deepcopy(pts)
            d.append(d_i)
            a.append(a_i)
            alpha.append(al_i)

        # Calculate Denavit-Hartenberg parameters for the last pair
        pts, a_i, al_i = lines[-1].common_perpendicular_to_other_line(lines[0])
        d_i = lines[-1].get_point_param(pts_prev[1]) - lines[-1].get_point_param(pts[0])

        d.append(d_i)
        a.append(a_i)
        alpha.append(al_i)

        for i in range(len(lines)):
            params = linkage.linkage[i].points_params

            middle_pts = self._map_joint_segment(d[i],
                                                 joint_segment,
                                                 params,
                                                 scale=scale,)
            #print(np.linalg.norm(middle_pts[0] - middle_pts[1]))
            middle_points.append(middle_pts)

        d = [d_i * scale for d_i in d]
        a = [a_i * scale for a_i in a]

        # Convert alpha to the specified form
        match unit:
            case "cos_alpha":
                pass
            case "deg":
                alpha = [np.degrees(np.arccos(alpha_i)) for alpha_i in alpha]
            case "rad":
                alpha = [np.arccos(alpha_i) for alpha_i in alpha]
            case _:
                raise ValueError("unit must be cos_alpha, deg or rad")

        return d, a, alpha, middle_points

    def _map_joint_segment(self, dh_d, joint_segment: float,
                           points_params: np.ndarray, scale: float = 1.0):
        """
        Map the joint segment to the scale of the linkage.

        :param float joint_segment: length of the joint segment
        :param float d_param: length of the dh parameter d
        :param np.ndarray points_params: list of connection points parameters of the
            joint
        :param float scale: scale of the linkage

        :return: mapped joint segment
        :rtype: np.ndarray
        """

        def map_interval(input_interval, max_length):
            middle_point = (input_interval[0] + input_interval[1]) / 2

            if input_interval[0] < input_interval[1]:
                mapped_interval = [middle_point - max_length/2, middle_point + max_length/2]
            else:
                mapped_interval = [middle_point + max_length/2, middle_point - max_length/2]
            return mapped_interval

        points_params = np.asarray(points_params)

        points_params_len = np.linalg.norm(points_params[0] - points_params[1])

        #if points_params_len > joint_segment:
        new_points_params = map_interval(points_params, (joint_segment + 2/scale))
        #else:
        #    raise ValueError("Joint segment is longer than the designed linkage.")


        # Calculate midpoints
        midpoint1 = new_points_params[0] + (new_points_params[1] - new_points_params[0]) / 4
        midpoint2 = new_points_params[0] + 3 * (new_points_params[1] - new_points_params[0]) / 4

        # Subtract the d_param from the joint segment
        midpoint1 = midpoint1
        midpoint2 = midpoint2

        return np.array([midpoint1, midpoint2]) * scale
    
    def collision_check(self, parallel: bool = False):
        """
        Perform full-cycle collision check on the line-model linkage.

        By default, the collision check is performed in non-parallel mode. This is
        faster for 4-bar linkages and 6-bar lingakes with a "simpler" motion curve,
        but slower for 6-bar linkages with "complex" motions.

        :param bool parallel: if True, perform collision check in parallel using
            multiprocessing

        :return: list of collision check results
        :rtype: list[str]
        """
        start_time = time()
        print("Collision check started...")

        # update the line segments (physical realization of the linkage)
        self.segments = self._get_line_segments_of_linkage()

        iters = []
        for ii in range(len(self.segments)):
            for jj in range(ii + 2, len(self.segments)):
                iters.append((ii, jj))

        print(f"Number of tasks to solve: {len(iters)}")

        if parallel:
            collision_results = self._collision_check_parallel(iters)
        else:
            collision_results = self._collision_check_nonparallel(iters)

        results = [r for r in collision_results if r is not None]
        if len(results) == 0:
            results = ["Linkage is without collisions!"]

        end_time = time()
        print(f"Collision check finished in {end_time - start_time} seconds.")

        return results

    def _collision_check_parallel(self, iters: list[tuple[int, int]]):
        """
        Perform collision check in parallel using multiprocessing.

        Slower for 4-bar linkages and 6-bar lingakes with a "simpler" motion curve,
        faster for 6-bar linkages with "complex" motions.

        :param list iters: list of tuples of indices of the line segments to be checked

        :return: list of collision check results
        :rtype: list[str]
        """
        print("--- running in parallel ---")
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._check_given_pair, iters)

        return list(results)

    def _collision_check_nonparallel(self, iters: list[tuple[int, int]]):
        """
        Perform collision check in non-parallel mode.

        Default option. Faster for 4-bar linkages and 6-bar lingakes with a "simpler"
        motion curve, slower for 6-bar linkages with "complex" motions.

        :param list iters: list of tuples of indices of the line segments to be checked

        :return: list of collision check results
        :rtype: list[str]
        """
        results = []
        for val in iters:
            results.append(self._check_given_pair(val))
        return results

    def _check_given_pair(self, iters: tuple[int, int]):
        """
        Perform collision check for a given pair of line segments and evaluate it.

        :param tuple iters: tuple of indices of the line segments to be checked

        :return: collision check result
        :rtype: str
        """
        i = iters[0]
        j = iters[1]
        # check if two lines are colliding
        collisions, coll_pts = self.colliding_lines(self.segments[i].equation,
                                                    self.segments[j].equation)

        if collisions is not None:
            # check if the collision is between the physical line segments
            physical_collision = [
                self.segments[i].is_point_in_segment(coll_pts[k], t_val) and
                self.segments[j].is_point_in_segment(coll_pts[k], t_val)
                for k, t_val in enumerate(collisions)
            ]
        else:
            physical_collision = [False]

        if True in physical_collision:
            result = f"{physical_collision} at parameters: {collisions} for linkage pair: {self.segments[i].type}_{self.segments[i].factorization_idx}{self.segments[i].idx} X {self.segments[j].type}_{self.segments[j].factorization_idx}{self.segments[j].idx}"
        else:
            #result = f"no collision between {self.segments[i].type}_{self.segments[i].factorization_idx}{self.segments[i].idx} X {self.segments[j].type}_{self.segments[j].factorization_idx}{self.segments[j].idx}"
            result = None

        return result

    def colliding_lines(self, l0: NormalizedLine, l1: NormalizedLine
                        ) -> tuple[list[float], list[PointHomogeneous]]:
        """
        Return the lines that are colliding in the linkage.

        :param NormalizedLine l0: equation of the first line
        :param NormalizedLine l1: equation of the second line

        :return: tuple (list of t values, list of intersection points)
        :rtype: tuple[list[float], list[PointHomogeneous]]
        """
        t = sp.Symbol("t")

        # lines are colliding if expr == 0
        expr = sp.simplify(np.dot(l0.direction, l1.moment) + np.dot(l0.moment, l1.direction))

        # neighbouring lines are colliding all the time (expr == 0)
        if expr == 0:
            return None, None

        e = sp.Expr(expr).subs(t, (t + 1) / 2)

        expr_poly = sp.Poly(e.args[0], t)

        expr_coeffs = expr_poly.all_coeffs()

        # convert to numpy polynomial
        expr_n = np.array(expr_coeffs, dtype="float64")
        np_poly = np.polynomial.polynomial.Polynomial(expr_n[::-1])
        np_poly_inversed = np.polynomial.polynomial.Polynomial(expr_n)

        # solve for t
        colliding_lines_sol = np_poly.roots()
        colliding_lines_sol_inversed = np_poly_inversed.roots()
        # extract real solutions
        sol_real = colliding_lines_sol.real[np.isclose(colliding_lines_sol.imag,
                                                       0, atol=1e-5)]
        sol_real_inversed = colliding_lines_sol_inversed.real[
            np.isclose(colliding_lines_sol_inversed.imag, 0, atol=1e-5)]

        sol_real = [((sol + 1)/2) for sol in sol_real]
        sol_real_inversed = [((sol + 1)/2) for sol in sol_real_inversed]

        solutions = deepcopy(sol_real)

        for sol in sol_real_inversed:
            if not np.isclose(sol, 0):
                sol = 1 / sol
                solutions = np.append(solutions, sol)
            else:
                #sol = np.inf
                sol = 10 ** 20
                solutions = np.append(solutions, sol)

        intersection_points = self.get_intersection_points(l0, l1, solutions)

        return solutions, intersection_points

    def get_intersection_points(self, l0: NormalizedLine, l1: NormalizedLine,
                                t_params: list[float]):
        """
        Return the intersection points of two lines.

        :param NormalizedLine l0: equation of the first line
        :param NormalizedLine l1: equation of the second line
        :param list[float] t_params: list of parameter values - points of intersection

        :return: list of intersection points
        :rtype: list[PointHomogeneous]
        """
        from PointHomogeneous import PointHomogeneous
        intersection_points = [PointHomogeneous()] * len(t_params)

        for i, t_val in enumerate(t_params):
            # evaluate the lines at the given parameter
            l0e = l0.evaluate(t_val)
            l1e = l1.evaluate(t_val)

            # common perpendicular to the two lines - there is none since they
            # intersect, therefore from the list of two points only 1 is needed
            inters_points, d, c = l0e.common_perpendicular_to_other_line(l1e)
            intersection_points[i] = PointHomogeneous.from_3d_point(inters_points[0])

        return intersection_points

    def _get_line_segments_of_linkage(self) -> list:
        """
        Return the line segments of the linkage.

        Line segments are the physical realization of the linkage. This method obtains
        their motion equations using default connection points of the factorizations
        (default meaning the static points in the home configuration).

        :return: list of LineSegment objects
        :rtype: list[LineSegment]
        """
        from Linkage import LineSegment

        t = sp.Symbol("t")

        segments = [[], []]

        # base (static) link has index 0 in the list of the 1st factorization
        eq, p0, p1 = self.factorizations[0].base_link(self.factorizations[1].linkage[0].points[0])
        segments[0].append(LineSegment(eq, p0, p1, linkage_type="b", f_idx=0, idx=0))

        # static joints
        segments[0].append(LineSegment(*self.factorizations[0].joint(0),
                                       linkage_type="j", f_idx=0, idx=0))
        segments[1].append(LineSegment(*self.factorizations[1].joint(0),
                                       linkage_type="j", f_idx=1, idx=0))

        # moving links and joints
        for i in range(2):
            for j in range(1, self.factorizations[i].number_of_factors):
                link, p0, p1 = self.factorizations[i].link(j)
                link = self.factorizations[i].act(link, end_idx=j-1, param=t)
                p0 = self.factorizations[i].act(p0, end_idx=j-1, param=t)
                p1 = self.factorizations[i].act(p1, end_idx=j-1, param=t)
                segments[i].append(LineSegment(link, p0, p1, linkage_type="l", f_idx=i, idx=j))

                joint, p0, p1 = self.factorizations[i].joint(j)
                joint = self.factorizations[i].act(joint, end_idx=j, param=t)
                p0 = self.factorizations[i].act(p0, end_idx=j, param=t)
                p1 = self.factorizations[i].act(p1, end_idx=j, param=t)
                segments[i].append(LineSegment(joint, p0, p1, linkage_type="j", f_idx=i, idx=j))

        # tool (moving - acted) link has index -1 in the list of the 2nd factorization
        tool_link, p0, p1 = self.factorizations[0].tool_link(self.factorizations[1].linkage[-1].points[1])
        tool_link = self.factorizations[0].act(tool_link, param=t)
        p0 = self.factorizations[0].act(p0, param=t)
        p1 = self.factorizations[1].act(p1, param=t)
        tool_idx = self.factorizations[1].number_of_factors
        segments[1].append(LineSegment(tool_link, p0, p1, linkage_type="t", f_idx=1, idx=tool_idx))

        return segments[0] + segments[1][::-1]


