import pickle
from copy import deepcopy
from time import time
from typing import Union
from warnings import warn

import numpy as np
import sympy as sp

from .DualQuaternion import DualQuaternion
from .MotionFactorization import MotionFactorization
from .NormalizedLine import NormalizedLine
from .RationalCurve import RationalCurve
from .TransfMatrix import TransfMatrix
from .PointHomogeneous import PointHomogeneous


class RationalMechanism(RationalCurve):
    """
    Class representing rational mechanisms in dual quaternion space.

    :ivar list factorizations: list of MotionFactorization objects
    :ivar int num_joints: number of joints in the mechanism
    :ivar bool is_linkage: True if the mechanism is a linkage, False if it is 1 branch
        of a linkage
    :ivar DualQuaternion tool_frame: end effector of the mechanism
    :ivar AffineMetric metric: object representing the metric of the mechanism
    :ivar LineSegment segments: list of LineSegment objects representing the physical
        realization of the linkage


    :examples:

    .. testcode:: [rationalmechanism_example1]

        # Create a rational mechanism from given example

        from rational_linkages import RationalMechanism, Plotter, TransfMatrix
        from rational_linkages.models import bennett_ark24


        # load the model of the Bennett's linkage
        m = bennett_ark24()

        # create an interactive plotter object, with 500 descrete steps
        # for the input rational curves, and arrows scaled to 0.05 length
        myplt = Plotter(interactive=True, steps=500, arrows_length=0.05)

        # plot the model with tool frame
        myplt.plot(m, show_tool=True)

        ##### additional plotting options #####
        # create a pose of the identity
        base = TransfMatrix()
        myplt.plot(base)

        # create another pose
        p0 = TransfMatrix.from_rpy_xyz([-90, 0, 0], [0.15, 0, 0], unit='deg')
        myplt.plot(p0)
        ######################################

        # show the plot
        myplt.show()

    .. testcleanup:: [rationalmechanism_example1]

        del RationalMechanism, Plotter, bennett_ark24
        del m, myplt, p0
    """

    def __init__(self, factorizations: list[MotionFactorization],
                 tool: Union[DualQuaternion, str] = None):
        """
        Initializes a RationalMechanism object
        """
        super().__init__(factorizations[0].set_of_polynomials)
        self.factorizations = factorizations
        self.num_joints = sum([f.number_of_factors for f in factorizations])

        self.tool_frame = self._determine_tool(tool)

        self.is_linkage = len(self.factorizations) == 2

        self._segments = None

        self._metric = None

    @property
    def segments(self):
        """
        Return the line segments of the linkage.

        Line segments are the physical realization of the linkage.

        :return: list of LineSegment objects
        :rtype: list[LineSegment]
        """
        if self._segments is None and self.is_linkage:
            self._segments = self._get_line_segments_of_linkage()
        else:
            ValueError("Segments are available only for linkages.")

        return self._segments

    @property
    def metric(self):
        """
        Define a metric in R12 for the mechanism.

        This metric is used for collision detection.
        """
        if self._metric is None:
            from .AffineMetric import AffineMetric  # inner import
            mechanism_points = self.points_at_parameter(0,
                                                        inverted_part=True,
                                                        only_links=False)
            self._metric = AffineMetric(self.curve(), mechanism_points)

        return self._metric

    @classmethod
    def from_saved_file(cls, filename: str):
        """
        Load a linkage object from a file.

        :param str filename: name of the file to load the linkage object from

        :return: linkage object
        :rtype: RationalMechanism

        :raises FileNotFoundError: if the file was not possible to load
        """
        # check if the filename has the .pkl extension
        if filename[-4:] != '.pkl':
            filename = filename + '.pkl'

        try:
            with open(filename, 'rb') as file:
                mechanism = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} was not found or possible "
                                    f"to load.")

        return mechanism

    def save(self, filename: str = None):
        """
        Save the linkage object to a file.

        :param str filename: name of the file to save the linkage object to
        """
        if filename is None:
            filename = 'saved_mechanism.pkl'
        elif filename[-4:] == '.pkl':
            pass
        else:
            filename = filename + '.pkl'

        # update the line segments (physical realization of the linkage) before saving
        self.update_segments()

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def _determine_tool(self, tool: Union[DualQuaternion, None, str]) -> DualQuaternion:
        """
        Determine the tool frame of the mechanism.

        :return: tool frame of the mechanism
        :rtype: DualQuaternion
        """
        if tool is None:
            return DualQuaternion(self.evaluate(0, inverted_part=True))
        elif isinstance(tool, DualQuaternion):
            return tool
        elif tool == 'mid_of_last_link':
            # calculate the midpoint of the last link
            nearly_zero = np.finfo(float).eps
            p0 = self.factorizations[0].direct_kinematics(nearly_zero, inverted_part=True)[-1]
            p1 = self.factorizations[1].direct_kinematics(nearly_zero, inverted_part=True)[-1]

            # define the x axis vector - along the last link
            vec_x = (p1 - p0) / np.linalg.norm(p1 - p0)

            # get some random vector from the last joint points
            vec_y_pts = self.factorizations[0].direct_kinematics(nearly_zero, inverted_part=True)[-2:]
            vec_y = vec_y_pts[1] - vec_y_pts[0]

            # define the z axis vector - perpendicular to the x and y vectors
            vec_z = np.cross(vec_x, vec_y)
            vec_z = vec_z / np.linalg.norm(vec_z)

            mid = (p1 + p0) / 2

            t = TransfMatrix.from_vectors(normal_x=vec_x, approach_z=vec_z, origin=mid)
            return DualQuaternion(t.matrix2dq())
        else:
            raise ValueError("tool must be either DualQuaternion, "
                             "None default motion zero configuration, "
                             "or 'mid_of_last_link'")

    def get_design(self,
                   unit: str = 'rad',
                   scale: float = 1.0,
                   joint_length: float = 20.0,
                   washer_length: float = 1.0,
                   update_design: bool = False,
                   pretty_print: bool = True,
                   onshape_print: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Get the design parameters of the linkage for the CAD model.

        The parameters are in the order: d, a, alpha, connection0,
        connection1, for every link.

        :param str unit: desired unit of the angle parameters, can be 'deg' or 'rad'
        :param float scale: scale of the length parameters of the linkage
        :param float joint_length: length of the joint segment in mm; default is 20 mm
            which corresponds to the CAD model that connects two 20 mm joint parts and
            has 1 mm thick washer between. Total length of the joint is 41 mm. It is
            used to calculate a midpoint distance between the two links that connect.
        :param float washer_length: length of the washer in mm; default is 1 mm
        :param bool update_design: if True, update the design of the mechanism (including joint segments)
        :param bool pretty_print: if True, print the parameters in a readable form,
            otherwise return a numpy array
        :param bool onshape_print: if True, print the parameters in a form that can be
            directly copied to Onshape

        :return: design parameters of the linkage
        :rtype: tuple (np.ndarray, np.ndarray, list)
        """
        screws = deepcopy(self.get_screw_axes())
        screws.append(screws[0])
        frames = self.get_frames()

        connection_params = self.get_segment_connections()
        mid_pts_dist = (joint_length + washer_length) / scale
        connection_params = self.map_connection_params(connection_params, mid_pts_dist)

        if update_design:
            # update the connection points of the joints
            branch0 = connection_params[:len(self.factorizations[0].dq_axes), :]
            branch1 = connection_params[len(self.factorizations[1].dq_axes):, :][::-1]

            branch0 = [[pt[0], pt[1]] for pt in branch0]

            # reorder point pairs of the second branch (since it goes reversed)
            branch1 = [[pt[1], pt[0]] for pt in branch1]

            self.factorizations[0].set_joint_connection_points_by_parameters(branch0)
            self.factorizations[1].set_joint_connection_points_by_parameters(branch1)

            self.update_segments()

        # add the first connection point to the end of the list
        connection_params = np.vstack((connection_params, connection_params[0, :]))

        design_params = np.zeros((self.num_joints, 2))

        for i in range(self.num_joints):
            design_params[i, 0] = (connection_params[i, 1]
                                   - screws[i].get_point_param(frames[i].t))
            design_params[i, 1] = (connection_params[i+1, 0]
                                   - screws[i+1].get_point_param(frames[i+1].t))

        design_params = design_params * scale

        design_points = []
        for i in range(self.num_joints):
            design_points.append(
                [screws[i].direction + design_params[i, 0] * screws[i].direction,
                 screws[i].direction + design_params[i, 1] * screws[i].direction])

        # ignore the first row (base frame)
        dh = self.get_dh_params(unit=unit, scale=scale)

        if onshape_print:
            for i in range(self.num_joints):
                print(f"link{i}: "
                      f"[{dh[i, 1]:.6f}, {dh[i, 2]:.6f}, {dh[i, 3]:.6f}], "
                      f"{design_params[i, 0]:.6f}, {design_params[i, 1]:.6f}")
            pretty_print = False

        if pretty_print:
            for i in range(self.num_joints):
                print("---")
                print(f"Link {i}: d = {dh[i, 1]:.6f}, "
                      f"a = {dh[i, 2]:.6f}, "
                      f"alpha = {dh[i, 3]:.6f}")
                print(f"cp_0 = {design_params[i, 0]:.6f}, "
                      f"cp_1 = {design_params[i, 1]:.6f}")

        return dh, design_params, design_points

    def get_segment_connections(self) -> np.ndarray:
        """
        Get the connection parameters of the linkage.

        :return: connection points of the linkage
        :rtype: np.ndarray
        """
        connection_params = np.zeros((self.num_joints, 2))
        for i in range(len(self.factorizations[0].linkage)):
            connection_params[i, :] = self.factorizations[0].linkage[i].points_params

        if len(self.factorizations) > 1:  # TODO refactor with random choice of the branch
            for i in range(len(self.factorizations[1].linkage)):
                # iterate from back to front
                connection_params[-1-i, :] = self.factorizations[1].linkage[i].points_params[::-1]

        return connection_params

    def get_dh_params(self,
                      unit: str = 'rad',
                      scale: float = 1.0,
                      include_base: bool = False) -> np.ndarray:
        """
        Get the standard Denavit-Hartenberg parameters of the linkage.

        The parameters are in the order: theta, d, a, alpha. It follows the standard
        convention. The first row is are the parameters of the base frame.

        See more in the paper by :footcite:t:`Huczala2022iccma`.

        :param str unit: desired unit of the angle parameters, can be 'deg', 'rad', or
            'tanhalf' for tangent half-angle representation
        :param float scale: scale of the length parameters of the linkage
        :param bool include_base: if True, identity frame will be placed at the
            beginning of the list of frames

        :return: theta, d, a, alpha array of Denavit-Hartenberg parameters
        :rtype: np.ndarray

        .. footbibliography::

        """
        frames = deepcopy(self.get_frames(include_base=include_base))
        j = self.num_joints + 1 if include_base else self.num_joints

        dh = np.zeros((j, 4))
        for i in range(j):
            th, d, a, al = frames[i].dh_to_other_frame(frames[i+1])

            if unit == 'deg':
                th = np.rad2deg(th)
                al = np.rad2deg(al)
            elif unit == 'tanhalf':
                th = np.tan(th / 2)
                al = np.tan(al / 2)
            elif unit != 'rad':
                raise ValueError("unit must be deg or rad")

            dh[i, :] = [th, scale * d, scale * a, al]
        return dh

    def get_frames(self, include_base: bool = False) -> list[TransfMatrix]:
        """
        Get the frames of a linkage that follow standard Denavit-Hartenberg convention.

        It returns n+2 frames, where n is the number of joints. The first frame is the
        base frame, and the last frame is an updated frame of the first joint that
        follows the DH convention in respect to the last joint's frame.

        :param bool include_base: if True, identity frame will be placed at the
            beginning of the list of frames

        :return: list of TransfMatrix objects
        :rtype: list[TransfMatrix]
        """
        from .TransfMatrix import TransfMatrix  # inner import

        screws = deepcopy(self.get_screw_axes())

        # add the first screw to the end of the list
        screws.append(screws[0])

        if include_base:
            frames = [TransfMatrix()] * (self.num_joints + 2)

            # insert origin as the base line
            screws.insert(0, NormalizedLine())
        else:
            frames = [TransfMatrix()] * (self.num_joints + 1)

        for i, line in enumerate(screws[1:]):
            # obtain the connection points and the distance to the previous line
            pts, dist, cos_angle = line.common_perpendicular_to_other_line(screws[i])
            vec = pts[0] - pts[1]

            if not np.isclose(dist, 0.0):  # if the lines are skew or parallel
                # normalize vec - future X axis
                vec_x = vec / np.linalg.norm(vec)

                # if parallel
                if np.isclose(cos_angle, 1.0) or np.isclose(cos_angle, -1.0):
                    # choose origin as footpoint of the line
                    frames[i + 1] = TransfMatrix.from_vectors(vec_x,
                                                              line.direction,
                                                              origin=line.point_on_line())

                else:  # if skew
                    # from line.dir (future Z axis) and x create an SE3 object
                    frames[i+1] = TransfMatrix.from_vectors(vec_x,
                                                            line.direction,
                                                            origin=pts[0])

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

                    frames[i + 1] = TransfMatrix.from_vectors(vec_x,
                                                              line.direction,
                                                              origin=pts[0])

        if not include_base:
            # update the last frame to close the linkage loop
            frames[0] = frames[-1]

        return frames

    def get_global_frames(self) -> list[TransfMatrix]:
        """
        Get the frames of the linkage in the global coordinate system.

        :return: list of TransfMatrix objects
        :rtype: list[TransfMatrix]
        """
        local_frames = self.get_frames(include_base=True)[1:]
        global_frames = [TransfMatrix()]

        for i in range(1, len(local_frames)):
            global_frames.append(global_frames[i-1] * local_frames[i])

        return global_frames

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

    def map_connection_params(self,
                              connection_params: np.ndarray,
                              midpoints_distance: float) -> np.ndarray:
        """
        Map the connection parameters to the given joint length.

        :param np.ndarray connection_params: list of connection points parameters of the
            linkage
        :param float midpoints_distance: distance between the midpoints of the two links
            that connect at a joint

        :return: mapped connection points parameters
        :rtype: np.ndarray
        """
        c_params = np.asarray(connection_params)

        for i in range(len(c_params)):
            c_params_len = np.linalg.norm(c_params[i, 0] - c_params[i, 1])

            if not np.allclose(c_params_len, midpoints_distance):
                c_params[i, :] = self._map_joint_segment(c_params[i, :],
                                                         midpoints_distance)

        return c_params

    @staticmethod
    def _map_joint_segment(points_params: np.ndarray, midpoints_distance: float):
        """
        Map the joint segment to the scale of the linkage.

        :param np.ndarray points_params: list of connection points parameters of the
            joint segment

        :param float midpoints_distance: distance between the midpoints of the two links
            that connect at a joint

        :return: mapped joint segment
        :rtype: np.ndarray
        """

        def map_interval(input_interval, max_length):
            mid_point = (input_interval[0] + input_interval[1]) / 2

            if input_interval[0] < input_interval[1]:
                mapped_interval = [mid_point - max_length/2, mid_point + max_length/2]
            else:
                mapped_interval = [mid_point + max_length/2, mid_point - max_length/2]
            return mapped_interval

        points_params = np.asarray(points_params)
        new_params = map_interval(points_params, midpoints_distance)

        # Calculate midpoints
        #midpoint1 = new_params[0] + (new_params[1] - new_params[0]) / 4
        #midpoint2 = new_params[0] + 3 * (new_params[1] - new_params[0]) / 4

        return new_params
    
    def collision_check(self,
                        parallel: bool = False,
                        pretty_print: bool = True,
                        only_links: bool = False,
                        terminate_on_first: bool = False):
        """
        Perform full-cycle collision check on the line-model linkage.

        By default, the collision check is performed in non-parallel mode. This is
        faster for 4-bar linkages and 6-bar lingakes with a "simpler" motion curve,
        but slower for 6-bar linkages with "complex" motions.

        :param bool parallel: if True, perform collision check in parallel using
            multiprocessing
        :param bool pretty_print: if True, print the results in a readable form
        :param bool only_links: if True, only link-link collisions are checked,
            expecting that distances between joint connection points are minimal
        :param bool terminate_on_first: if True, terminate the collision check when
            the first collision is found

        :return: list of collision check colliding parameter values
        :rtype: list[float]
        """
        start_time = time()
        print("Collision check started...")

        # update the line segments (physical realization of the linkage)
        self.update_segments()

        iters = []
        # iterate over all line segments
        for ii in range(len(self.segments)):
            # for each line segment, iterate over all other line segments that are not
            # its immediate neighbors
            for jj in range(ii + 2, len(self.segments)):
                # in only links should be checked (joint segments have minimal length)
                if only_links:
                    if (self.segments[ii].type == 'j'
                            or self.segments[jj].type == 'j'
                            or jj - ii == 2):  # skip neighbouring links
                        pass
                    else:
                        iters.append((ii, jj))
                else:
                    iters.append((ii, jj))

        # remove the last neighbouring pair of links
        if only_links:
            # find the pair with the highest difference
            max_pair = max(iters, key=lambda x: x[1] - x[0])
            # remove the pair from the list
            iters.remove(max_pair)

        print(f"--- number of tasks to solve: {len(iters)} ---")

        if parallel:
            collision_results = self._collision_check_parallel(iters)
        else:
            collision_results = self._collision_check_nonparallel(iters,
                                                                  terminate_on_first)

        results = [r for r in collision_results if r is not None]
        flattened_results = [item for sublist in results for item in sublist]
        if len(results) == 0:
            flattened_results = None

        end_time = time()
        print(f"--- collision check finished in {end_time - start_time} seconds.")

        if pretty_print:
            if flattened_results is None:
                print("No collisions found.")
            else:
                print(f"The linkage is colliding {len(flattened_results)} times at the "
                      f"following parameter values:")
                for res in flattened_results:
                    print(res)

        return flattened_results

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

    def _collision_check_nonparallel(self, iters: list[tuple[int, int]],
                                     terminate_on_first: bool = False):
        """
        Perform collision check in non-parallel mode.

        Default option. Faster for 4-bar linkages and 6-bar lingakes with a "simpler"
        motion curve, slower for 6-bar linkages with "complex" motions.

        :param list iters: list of tuples of indices of the line segments to be checked
        :param bool terminate_on_first: if True, terminate the collision check when
            the first collision is found

        :return: list of collision check results
        :rtype: list[str]
        """
        results = []
        for val in iters:
            collsion = self._check_given_pair(val)
            results.append(collsion)
            if terminate_on_first and collsion is not None:
                break
        return results

    def _check_given_pair(self, iters: tuple[int, int]) -> list[float]:
        """
        Perform collision check for a given pair of line segments and evaluate it.

        :param tuple iters: tuple of indices of the line segments to be checked

        :return: collision check result
        :rtype: list[float]
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

        result = []
        if True in physical_collision:
            for i, res in enumerate(physical_collision):
                if res:
                    result.append(collisions[i])
            #result = f"{physical_collision} at parameters: {collisions} for linkage pair: {self.segments[i].type}_{self.segments[i].factorization_idx}{self.segments[i].idx} X {self.segments[j].type}_{self.segments[j].factorization_idx}{self.segments[j].idx}"
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

        # lines are colliding when expr = 0
        expr = np.dot(l0.direction, l1.moment) + np.dot(l0.moment, l1.direction)

        # reparametrize the expresion by t -> (t + 1) / 2 to interval [-1, 1]
        e = sp.Expr(expr).subs(t, (t + 1) / 2).evalf()

        expr_poly = sp.Poly(e.args[0], t)
        expr_coeffs = expr_poly.all_coeffs()

        # convert to numpy polynomial
        expr_n = np.array(expr_coeffs, dtype="float64")
        np_poly = np.polynomial.polynomial.Polynomial(expr_n[::-1])

        # inversing coeffs enables to solve intervals (-oo, 0) and (0, oo), that are
        # actually mapped to [-1, 1]
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
            if np.isclose(sol, 0):
                # eps is very small number (avoid division by zero)
                sol = 1 / np.finfo(float).eps
            else:
                sol = 1 / sol

            solutions = np.append(solutions, sol)

        intersection_points = self.get_intersection_points(l0, l1, solutions)

        return solutions, intersection_points

    @staticmethod
    def get_intersection_points(l0: NormalizedLine, l1: NormalizedLine,
                                t_params: list[float]):
        """
        Return the intersection points of two lines.

        :param NormalizedLine l0: equation of the first line
        :param NormalizedLine l1: equation of the second line
        :param list[float] t_params: list of parameter values - points of intersection

        :return: list of intersection points
        :rtype: list[PointHomogeneous]
        """
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
        from .Linkage import LineSegment  # inner import

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

    def get_motion_curve(self):
        """
        Return the rational motion curve of the linkage as RationalCurve object.

        :return: motion curve of the linkage
        :rtype: RationalCurve
        """
        return self.curve()

    def singularity_check(self):
        """
        Perform singularity check of the mechanism.
        """
        from .SingularityAnalysis import SingularityAnalysis  # inner import

        sa = SingularityAnalysis()
        return sa.check_singularity(self)

    def smallest_polyline(self, update_design: bool = False) -> tuple[list, list, float]:
        """
        Obtain the smallest polyline links for the mechanism.

        :param bool update_design: if True, update also the design of the mechanism

        :return: list of points of the smallest polyline, list of points parameters,
            result of the optimization
        :rtype: list, list, float
        """
        from .CollisionFreeOptimization import CollisionFreeOptimization  # inner import

        # get smallest polyline
        pts, points_params, res = CollisionFreeOptimization(self).smallest_polyline()

        # update the design of the mechanism
        if update_design:
            if len(self.factorizations) == 1:
                self.factorizations[0].set_joint_connection_points_by_parameters(points_params)
            else:
                self.factorizations[0].set_joint_connection_points_by_parameters(
                    points_params[:len(self.factorizations[0].dq_axes)])
                self.factorizations[1].set_joint_connection_points_by_parameters(
                    points_params[len(self.factorizations[1].dq_axes):][::-1])

        return pts, points_params, res

    def collision_free_optimization(self,
                                    method: str = None,
                                    step_length=25,
                                    min_joint_segment_length: float = 0.001,
                                    max_iters: int = 10,
                                    **kwargs):
        """
        Perform collision-free optimization of the mechanism.

        :param str method: method of optimization, can be 'combinatorial_search' by
            :footcite:t:`Li2020`
        :param float step_length: length of the step, i.e. the shift distance value, see
            :ref:`combinatorial_search` for more detail
        :param float min_joint_segment_length: minimum length of the joint segment
        :param int max_iters: maximum number of iterations
        :param kwargs: additional keyword arguments

        :return: list of collision-free points parameters
        :rtype: list
        """
        from .CollisionFreeOptimization import CollisionFreeOptimization
        optimizer = CollisionFreeOptimization(self)

        if method is None:
            method = 'combinatorial_search'

        match method:
            case 'combinatorial_search':
                results = optimizer.optimize(method=method,
                                             step_length=step_length,
                                             min_joint_segment_length=min_joint_segment_length,
                                             max_iters=max_iters,
                                             **kwargs)
            case _:
                raise ValueError("Invalid method.")

        return results

    def points_at_parameter(self,
                            t_param: float,
                            inverted_part: bool = False,
                            only_links: bool = False) -> list[PointHomogeneous]:
        """
        Get the points of the mechanism at the given parameter.

        :param float t_param: parameter value
        :param bool inverted_part: if True, return the evaluated points for the inverted
            part of the mechanism
        :param bool only_links: if True, instead of two points per joint segment,
            return only the first one

        :return: list of connection points of the mechanism
        :rtype: list[PointHomogeneous]
        """
        branch0 = self.factorizations[0].direct_kinematics(t_param,
                                                           inverted_part=inverted_part)
        branch1 = self.factorizations[1].direct_kinematics(t_param,
                                                           inverted_part=inverted_part)

        points = branch0 + branch1[::-1]

        if only_links:
            points = [points[i] for i in range(0, len(points), 2)]

        return [PointHomogeneous.from_3d_point(p) for p in points]

    def forward_kinematics(self,
                           joint_angle: float,
                           unit: str = 'rad') -> DualQuaternion:
        """
        Calculate forward (direct) kinematics of the mechanism. Radians are default.

        :param float joint_angle: angle of the joint
        :param str unit: unit of the joint angle, can be 'rad' or 'deg'

        :return: tool frame of the mechanism
        :rtype: DualQuaternion
        """
        if unit == 'deg':
            joint_angle = np.deg2rad(joint_angle)
        elif unit != 'rad':
            raise ValueError("unit must be deg or rad")

        t = self.factorizations[0].joint_angle_to_t_param(joint_angle)

        return DualQuaternion(self.evaluate(t)) * self.tool_frame

    def direct_kinematics(self,
                          joint_angle: float,
                          unit: str = 'rad') -> DualQuaternion:
        """
        Calculate direct (forward) kinematics of the mechanism. Radians are default.

        Calls the forward_kinematics method.

        :param float joint_angle: angle of the joint
        :param str unit: unit of the joint angle, can be 'rad' or 'deg'

        :return: tool frame of the mechanism
        :rtype: DualQuaternion
        """
        return self.forward_kinematics(joint_angle, unit)

    def inverse_kinematics(self,
                           pose: Union[DualQuaternion, TransfMatrix],
                           unit: str = 'rad',
                           method: str = 'gauss-newton',
                           robust: bool = False
                           ) -> float:
        """
        Calculate inverse kinematics for given pose. Returns the joint angle in radians.

        :param Union[DualQuaternion, TransfMatrix] pose: pose of the mechanism
        :param str unit: unit of the joint angle, can be 'rad' or 'deg'
        :param str method: numerically for 'gauss-newton' or 'algebraic'; 'algebraic'
            requires the input pose to be "achievable" by the mechanism, i.e. the pose
            must be on Study quadric and the mechanism must be able to reach it
        :param bool robust: if True, use the Gauss-Newton method with
            many initial guesses and more iteration steps

        :return: joint angle in radians or degrees
        :rtype: float
        """
        if isinstance(pose, TransfMatrix):
            pose = DualQuaternion(pose.matrix2dq())
        elif not isinstance(pose, DualQuaternion):
            raise ValueError("pose must be either DualQuaternion or TransfMatrix")

        if unit != 'rad' and unit != 'deg':
            raise ValueError("unit must be deg or rad")

        if method == 'algebraic':
            # TODO: implement algebraic method
            raise NotImplementedError("Algebraic method is not implemented yet.")
        elif method == 'gauss-newton':
            t = self._ik_gauss_newton(pose, robust_search=robust)
        else:
            raise ValueError("method must be either 'algebraic' or 'gauss-newton")

        joint_angle = self.factorizations[0].t_param_to_joint_angle(t)

        if unit == 'deg':
            joint_angle = np.rad2deg(joint_angle)

        return joint_angle

    def _ik_gauss_newton(self,
                         goal_pose: DualQuaternion,
                         robust_search: bool = False) -> float:
        """
        Calculate inverse kinematics using Gauss-Newton method.

        :param DualQuaternion goal_pose: pose of the mechanism
        :param bool robust_search: if True, use many initial guesses

        :return: parameter value
        :rtype: float

        :warns: if the method does not converge
        """
        def run_gauss_newton(pose, robust):
            t = sp.Symbol("t")

            curves = [self.curve(), self.curve().inverse_curve()]
            success = False
            inversed_part = False
            t_min = [None, float('inf')]
            t_init_set = [0., -0.999999999, 0.999999999, -0.5, 0.5]
            t_res = None
            max_iterations = 10
            tol = 1e-10

            # map pose to the motion curve of identity frame
            pose = pose * self.tool_frame.inv()

            if robust:
                t_init_set = np.linspace(-1.0, 1.0, 30)
                max_iterations = 50

            for inv, curve in enumerate(curves):
                if inv == 1:
                    inversed_part = True

                norm_curve = [element / curve.set_of_polynomials[0]
                              for element in curve.set_of_polynomials]

                c_diff = [element.diff(t) for element in norm_curve]

                for t_val in t_init_set:
                    step_size = 1.0
                    previous_error = float('inf')

                    for i in range(max_iterations):

                        if not robust:
                            if t_val == sp.nan or t_val > 1.0 or t_val < -1.0:
                                break

                        target_pose = pose.array()
                        current_pose = curve.evaluate(t_val)
                        c_diff_eval = np.array([element.subs(t, t_val).evalf()
                                                for element in c_diff])

                        # error to desired pose
                        if (target_pose[0] == 0. or current_pose[0] == 0.):
                            twist_to_desired = target_pose - current_pose
                        else:
                            twist_to_desired = (target_pose / target_pose[0]
                                                - current_pose / current_pose[0])

                        square_dist_to_desired = np.sum(twist_to_desired ** 2)

                        t_val += (step_size * (c_diff_eval @ twist_to_desired)
                                  / np.sum(c_diff_eval ** 2))

                        if square_dist_to_desired > previous_error:
                            step_size *= 0.5
                        else:
                            step_size = 1.0

                        if square_dist_to_desired < tol:
                            success = True
                            t_res = t_val
                            break

                    if square_dist_to_desired < t_min[1]:
                        t_min = [t_val, square_dist_to_desired]

                    if success:
                        break
                if success:
                    break

            if not success:
                t_res = t_min[0]

            if inversed_part:
                if t_res == 0.0:
                    t_res = np.finfo(np.float64).tiny
                t_res = 1 / t_res

            return t_res, success

        t_res, success = run_gauss_newton(pose=goal_pose, robust=robust_search)

        if not success:
            print("Fast search did not converge. Retrying with different initial "
                  "guesses...")
            t_res, success = run_gauss_newton(pose=goal_pose, robust=True)
            print("...done.")
            print("Converged successfully.") if success else (
                warn("Not converged, providing the closest result."))

        return t_res

    @staticmethod
    def traj_p2p_joint_space(joint_angle_start: float,
                             joint_angle_end: float,
                             velocity_start: float = 0.0,
                             velocity_end: float = 0.0,
                             unit: str = 'rad',
                             time_sec: float = 1.0,
                             num_points: int = 100,
                             method: str = 'quintic',
                             generate_csv: bool = False) -> tuple:
        """
        Generate point to point straight line joint space trajectory.

        This method originates from book Modern Robotics :footcite:p:`Lynch2017`
        by Kevin M. Lynch
        and Frank C. Park, and related software package published under MIT
        licence and available at: https://github.com/NxRLab/ModernRobotics

        .. footbibliography::

        :param float joint_angle_start: start parameter value
        :param float joint_angle_end: end parameter value
        :param float velocity_start: start velocity
        :param float velocity_end: end velocity
        :param str unit: unit of the joint angle, can be 'rad' or 'deg'
        :param float time_sec: time of the trajectory [seconds]
        :param int num_points: number of discrete points in the trajectory
        :param str method: method of trajectory generation, can be 'quintic' or 'cubic'
        :param bool generate_csv: if True, generate a CSV file with the trajectory

        :return: tuple of joint position (angle), velocity, and acceleration
        :rtype: tuple

        :raises: ValueError: if unit is not 'rad' or 'deg'
        :raises: ValueError: if method is not 'quintic' or 'cubic'

        :example:

        .. testcode:: [rationalmechanism_example2]

            from rational_linkages import RationalCurve, RationalMechanism
            import numpy as np
            import matplotlib.pyplot as plt

            coeffs = np.array([[0, 0, 0],
                               [4440, 39870, 22134],
                               [16428, 9927, -42966],
                               [-37296,-73843,-115878],
                               [0, 0, 0],
                               [-1332, -14586, -7812],
                               [-2664, -1473, 6510],
                               [-1332, -1881, -3906]])
            c = RationalCurve.from_coeffs(coeffs)
            m = RationalMechanism(c.factorize())

            time = 3  # seconds
            n_steps = 100
            t0 = 0
            t1 = np.pi/4
            method = 'quintic'
            #method = 'cubic'

            pos, vel, acc = m.traj_p2p_joint_space(joint_angle_start=t0,
                                                   joint_angle_end=t1,
                                                   time_sec=time,
                                                   method=method,
                                                   num_points=n_steps)

            # plot the trajectory
            plt.plot(pos)
            plt.plot(vel)
            plt.plot(acc)
            plt.xlabel('Time [sec]')
            plt.legend(['Position [rad]', 'Velocity [rad/s]', 'Acceleration [rad/s^2]'])
            plt.grid()
            plt.show()

        .. testcleanup:: [rationalmechanism_example2]

            del RationalCurve, RationalMechanism, np
            del plt, coeffs, c, m, time, n_steps, t0, t1, method, pos, vel, acc

        """
        if unit == 'deg':
            joint_angle_start = np.deg2rad(joint_angle_start)
            joint_angle_end = np.deg2rad(joint_angle_end)
        elif unit != 'rad':
            raise ValueError("unit must be deg or rad")

        def cubic_time_scaling(tot_time, step):
            return 3 * (step / tot_time) ** 2 - 2 * (step / tot_time) ** 3

        def quintic_time_scaling(tot_time, step):
            return (10 * (step / tot_time) ** 3
                    - 15 * (step / tot_time) ** 4
                    + 6 * (step / tot_time) ** 5)

        def quintic_time_scaling_with_velocity(t, tot_time, th_0, th_f, v_0, v_f):
            t3, t4, t5 = tot_time ** 3, tot_time ** 4, tot_time ** 5
            a_0, a_1, a_2 = th_0, v_0, 0
            a_3 = (20 * (th_f - th_0) - (8 * v_f + 12 * v_0) * tot_time) / (2 * t3)
            a_4 = (30 * (th_0 - th_f) + (14 * v_f + 16 * v_0) * tot_time) / (2 * t4)
            a_5 = (12 * (th_f - th_0) - (6 * v_f + 6 * v_0) * tot_time) / (2 * t5)
            return (a_0 + a_1 * t + a_2 * t ** 2 + a_3 * t ** 3
                    + a_4 * t ** 4 + a_5 * t ** 5)

        time_gap = time_sec / num_points
        traj = np.zeros(num_points)
        for i in range(num_points):
            if method == 'cubic' or method == 'quintic':
                if method == 'cubic':
                    scaling = cubic_time_scaling(time_sec, time_gap * i)
                elif method == 'quintic':
                    scaling = quintic_time_scaling(time_sec, time_gap * i)

                traj[i] = (scaling * np.array(joint_angle_end)
                           + (1 - scaling) * np.array(joint_angle_start))
            elif method == 'quintic_with_velocity':
                traj[i] = quintic_time_scaling_with_velocity(time_gap * i,
                                                             time_sec,
                                                             joint_angle_start,
                                                             joint_angle_end,
                                                             velocity_start,
                                                             velocity_end)
            else:
                raise ValueError("method must be either 'cubic', 'quintic', "
                                 "or 'quintic_with_velocity'")

        if generate_csv:
            RationalMechanism._generate_csv(traj, time_gap)

        vel = np.diff(traj, axis=0) * num_points / time_sec
        acc = np.diff(vel, axis=0) * num_points / time_sec

        return traj, vel, acc

    def traj_smooth_tool(self,
                         joint_angle_start: float,
                         joint_angle_end: float,
                         time_sec: float,
                         point_of_interest: PointHomogeneous = None,
                         unit: str = 'rad',
                         num_points: int = 100,
                         generate_csv: bool = False) -> tuple:
        """
        Generate smooth trajectory for the tool of the mechanism.

        :param float joint_angle_start: start parameter value
        :param float joint_angle_end: end parameter value
        :param float time_sec: time of the trajectory [seconds]
        :param PointHomogeneous point_of_interest: point that will be moved smoothly
        :param str unit: unit of the joint angle, can be 'rad' or 'deg'
        :param int num_points: number of discrete points in the trajectory
        :param bool generate_csv: if True, generate a CSV file with the trajectory

        :return: tuple of joint position (angle), velocity, and acceleration
        :rtype: tuple
        """
        if unit == 'deg':
            joint_angle_start = np.deg2rad(joint_angle_start)
            joint_angle_end = np.deg2rad(joint_angle_end)
        elif unit != 'rad':
            raise ValueError("unit must be deg or rad")

        t_start = self.factorizations[0].joint_angle_to_t_param(joint_angle_start)
        t_end = self.factorizations[0].joint_angle_to_t_param(joint_angle_end)

        if point_of_interest is None:
            ee_point = PointHomogeneous.from_3d_point(
                self.tool_frame.dq2point_via_matrix())
        else:
            ee_point = point_of_interest

        # check if the t vals are in the correct order
        flip = False
        if t_start > t_end:
            flip = True
            t_start, t_end = t_end, t_start

        t_vals = self.split_in_equal_segments(interval=[t_start, t_end],
                                              point_to_act_on=ee_point,
                                              num_segments=num_points)

        joint_angles = [self.factorizations[0].t_param_to_joint_angle(x)
                        for x in t_vals]

        # flip the joint angles back if needed
        if flip:
            joint_angles = joint_angles[::-1]

        if generate_csv:
            time_gap = time_sec / num_points
            RationalMechanism._generate_csv(joint_angles, time_gap)

        vel = np.diff(joint_angles, axis=0) * num_points / time_sec
        acc = np.diff(vel, axis=0) * num_points / time_sec

        return joint_angles, vel, acc

    @staticmethod
    def _generate_csv(traj, time_gap):
        """
        Generate a CSV file with the trajectory.

        :param traj: trajectory
        :param time_gap: time gap
        """
        time_space = np.arange(0, len(traj) * time_gap, time_gap)
        pos = traj
        vel = np.diff(traj, axis=0) / time_gap
        #vel = np.append(np.array([0.0]), vel)  # add .0 to equalize the array length
        vel = np.append(vel, vel[-1])
        # TODO: check if this is correct
        acc = np.diff(vel, axis=0) / time_gap
        #acc = np.append(acc, np.array([0.0]))  # add .0 to equalize the array length
        acc = np.append(acc, acc[-1])

        # Stack the arrays horizontally to create a 2D array with 4 columns
        data = np.column_stack((time_space, pos, vel, acc))

        # Save the stacked array to a CSV file
        np.savetxt('trajectory.csv', data, delimiter=',', fmt='%1.6f')

    def update_metric(self):
        """
        Update the metric of the mechanism.

        Set to none so that the metric is recalculated when needed.
        """
        self._metric = None

    def update_segments(self):
        """
        Update the line segments of the linkage.
        """
        self._segments = self._get_line_segments_of_linkage()

    def get_relative_motions(self):
        """
        Get the relative motions of the mechanism.
        """
        sequence = DualQuaternion()
        branch0 = [sequence := sequence * factor for factor in
                   self.factorizations[0].factors_with_parameter]

        sequence = DualQuaternion()
        branch1 = [sequence := sequence * factor for factor in
                   self.factorizations[1].factors_with_parameter]

        relative_motions = branch0 + branch1[::-1]
        return relative_motions

