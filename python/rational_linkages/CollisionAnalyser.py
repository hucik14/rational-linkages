import numpy

from sympy import symbols, Poly

from .Linkage import LineSegment
from .DualQuaternion import DualQuaternion
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous, PointOrbit
from .RationalBezier import RationalSoo
from .RationalCurve import RationalCurve
from .RationalMechanism import RationalMechanism


class CollisionAnalyser:
    def __init__(self, mechanism: RationalMechanism):
        self.mechanism = mechanism
        self.mechanism_points = mechanism.points_at_parameter(0,
                                                              inverted_part=True,
                                                              only_links=False)
        self.metric = mechanism.metric

        self.segment_orbits = {}
        self.segments = {}
        for segment in mechanism.segments:
            self.segments[segment.id] = segment

        self.motions = self.get_motions()
        self.bezier_splits = self.get_bezier_splits(20)

    def get_bezier_splits(self, min_splits: int = 0) -> list:
        """
        Split the relative motions of the mechanism into bezier curves.
        """
        return [motion.split_in_beziers(min_splits) for motion in self.motions]

    def get_motions(self):
        """
        Get the relative motions of the mechanism represented as rational curves.
        """
        sequence = DualQuaternion()
        branch0 = [sequence := sequence * factor for factor in
                   self.mechanism.factorizations[0].factors_with_parameter]

        sequence = DualQuaternion()
        branch1 = [sequence := sequence * factor for factor in
                   self.mechanism.factorizations[1].factors_with_parameter]

        relative_motions = branch0 + branch1[::-1]

        t = symbols('t')

        motions = []
        for motion in relative_motions:
            motions.append(RationalCurve([Poly(c, t, greedy=False)
                                          for c in motion],
                                         metric=self.metric))
        return motions

    def get_points_orbits(self):
        """
        Get the orbits of the mechanism points.
        """
        return [PointOrbit(*point.get_point_orbit(metric=self.metric))
                for point in self.mechanism_points]

    def get_segment_orbit(self, segment_id: str):
        """
        Get the orbit of a segment.
        """
        segment = self.segments[segment_id]

        if segment.type == 'l':
            if segment.factorization_idx == 0:
                split_idx = segment.idx - 1
                p0_idx = 2 * segment.idx - 1
                p1_idx = 2 * segment.idx
            else:
                split_idx = -1 * segment.idx
                p0_idx = -2 * segment.idx - 1
                p1_idx = -2 * segment.idx
        else:  # type == 'j'
            if segment.factorization_idx == 0:
                split_idx = segment.idx - 1
                p0_idx = 2 * segment.idx
                p1_idx = 2 * segment.idx + 1
            else:
                split_idx = -1 * segment.idx
                p0_idx = -2 * segment.idx - 1
                p1_idx = -2 * segment.idx - 2

        p0 = self.mechanism_points[p0_idx]
        p1 = self.mechanism_points[p1_idx]

        if segment.type != 'b':
            rel_bezier_splits = self.bezier_splits[split_idx]

            orbits0 = [PointOrbit(*p0.get_point_orbit(acting_center=split.ball.center,
                                                      acting_radius=split.ball.radius_squared,
                                                      metric=self.metric),
                                  t_interval=split.t_param_of_motion_curve)
                       for split in rel_bezier_splits]
            orbits1 = [PointOrbit(*p1.get_point_orbit(acting_center=split.ball.center,
                                                      acting_radius=split.ball.radius_squared,
                                                      metric=self.metric),
                                  t_interval=split.t_param_of_motion_curve)
                       for split in rel_bezier_splits]
        else:
            diff = p0.coordinates - p1.coordinates
            radius_sq = numpy.dot(diff, diff) / 10
            orbits0 = [PointOrbit(point_center=p0, radius_squared=radius_sq, t_interval=(None, [-1,1]))]
            orbits1 = [PointOrbit(point_center=p1, radius_squared=radius_sq, t_interval=(None, [-1,1]))]

        all_orbits = []
        for i in range(len(orbits0)):
            orbits_for_t = [orbits0[i].t_interval, orbits0[i]]
            dist = numpy.linalg.norm(orbits0[i].center.normalized_in_3d() - orbits1[i].center.normalized_in_3d())
            radius_sum = orbits0[i].radius + orbits1[i].radius
            if dist > radius_sum:
                add_balls = dist / radius_sum
                num_steps = int(add_balls) * 2 + 1

                # linear interpolation from smaller ball to bigger ball
                radii = 0
                radius_diff = orbits1[i].radius - orbits0[i].radius
                center_diff = orbits1[i].center - orbits0[i].center
                for j in range(1, num_steps):
                    new_radius = orbits0[i].radius + j * radius_diff / num_steps
                    radii += new_radius
                    new_center = orbits0[i].center + 2 * radii * center_diff / (dist * 2)
                    orbits_for_t.append(PointOrbit(new_center, new_radius ** 2, orbits0[i].t_interval))
            orbits_for_t.append(orbits1[i])
            all_orbits.append(orbits_for_t)

        return all_orbits

    def check_two_segments(self, segment0: str, segment1: str, t_interval=None):
        """
        Check if two segments collide.
        """
        if not segment0 in self.segment_orbits:
            self.segment_orbits[segment0] = self.get_segment_orbit(segment0)

        if not segment1 in self.segment_orbits:
            self.segment_orbits[segment1] = self.get_segment_orbit(segment1)

        seg_orb0 = self.segment_orbits[segment0]
        seg_orb1 = self.segment_orbits[segment1]

        if t_interval is None:  # check for all t
            link_balls_0 = []
            for ball in seg_orb0:
                link_balls_0 += ball[1:]

            link_balls_1 = []
            for ball in seg_orb1:
                link_balls_1 += ball[1:]

            import time
            start_time = time.time()

            num_checked_balls = 0
            num_of_collisions = 0
            it_collides = False
            for ball0 in link_balls_0:
                for ball1 in link_balls_1:
                    num_checked_balls += 1
                    if self.check_two_miniballs(ball0, ball1):
                        num_of_collisions += 1
                        it_collides = True

            print(f'Number of checked balls: {num_checked_balls}')
            print(f'time for checking balls: {time.time() - start_time}')

        elif isinstance(t_interval[1], float):
            for i, interval in enumerate(seg_orb0):
                start, end = interval[0][1][0], interval[0][1][1]
                if start <= t_interval[1] <= end and (t_interval[0] == interval[0][0] or interval[0][0] is None):  # None for base
                    link_balls_0 = seg_orb0[i][1:]
                else:
                    ValueError('Given interval is not valid')

            for i, interval in enumerate(seg_orb1):
                start, end = interval[0][1][0], interval[0][1][1]
                if start <= t_interval[1] <= end and (t_interval[0] == interval[0][0] or interval[0][0] is None):
                    link_balls_1 = seg_orb1[i][1:]
                else:
                    ValueError('Given interval is not valid')

            num_of_collisions = 0
            it_collides = False
            for ball0 in link_balls_0:
                for ball1 in link_balls_1:
                    if self.check_two_miniballs(ball0, ball1):
                        num_of_collisions += 1
                        it_collides = True

        print(f'Number of colliding balls: {num_of_collisions}')

        return it_collides

    @staticmethod
    def check_two_miniballs(ball0, ball1):
        """
        Check if two miniballs collide.
        """
        diff = ball0.center.coordinates - ball1.center.coordinates
        center_dist_squared = numpy.dot(diff, diff)
        return center_dist_squared < ball0.radius_squared + ball1.radius_squared

    def get_split_and_point_indices(self, segment):
        """
        Compute split index and point indices for a segment.
        """
        if segment.type == 'l':
            if segment.factorization_idx == 0:
                split_idx = segment.idx - 1
                p0_idx = 2 * segment.idx - 1
                p1_idx = 2 * segment.idx
            else:
                split_idx = -1 * segment.idx
                p0_idx = -2 * segment.idx - 1
                p1_idx = -2 * segment.idx
        else:  # type == 'j'
            if segment.factorization_idx == 0:
                split_idx = segment.idx - 1
                p0_idx = 2 * segment.idx
                p1_idx = 2 * segment.idx + 1
            else:
                split_idx = -1 * segment.idx
                p0_idx = -2 * segment.idx - 1
                p1_idx = -2 * segment.idx - 2
        return split_idx, p0_idx, p1_idx

    def optimize_curved_link(self,
                             segment_id: str,
                             min_splits: int = 20,
                             curve_degree: int = 3):
        """
        Optimize the curved link to avoid collisions.
        """
        if segment_id.startswith('j'):
            raise ValueError('Joints cannot be optimized as curved lines, only links.')

        # get segment creation index
        segment_id_num = None
        for s_id, segment in enumerate(self.segments.values()):
            if segment.id == segment_id:
                segment_id_num = s_id
                break

        indices = list(range(len(self.mechanism.segments)))

        # remove index of segment to optimize and the two neighboring segments
        indices.remove(segment_id_num)
        if segment_id_num != 0:
            indices.remove(segment_id_num - 1)
        else:
            indices.remove(indices[-1])  # remove last if first segment is optimized
        if segment_id_num != len(self.mechanism.segments) - 1:
            indices.remove(segment_id_num + 1)
        else:
            indices.remove(indices[0])  # remove first if last segment is optimized

        # remove odd indices which correspond to joints; keep also zero
        indices_reduced = [idx for i, idx in enumerate(indices)
                           if idx % 2 == 0 or idx == 0]

        bounding_balls = self.obtain_global_bounding_balls(segment_id_num,
                                                           indices_reduced,
                                                           min_splits)

        dh, design_params, design_points = self.mechanism.get_design(
            return_point_homogeneous=True,
            update_design=True,
            pretty_print=False)

        joint_id = segment_id_num // 2
        pt0 = design_points[joint_id - 1][1]
        pt1 = design_points[joint_id][0]

        link_cps = RationalSoo.control_points_between_two_points(pt0,
                                                                 pt1,
                                                                 degree=curve_degree)
        init_control_points = link_cps[1:-1]  # remove the first and last control points

        new_cps = self.optimize_control_points(init_control_points,
                                               bounding_balls)
        new_cps.insert(0, pt0)
        new_cps.append(pt1)

        return RationalSoo(new_cps)

    @staticmethod
    def optimize_control_points(init_points: list[PointHomogeneous],
                                bounding_orbits: list[list[PointOrbit]]):
        """
        Optimize the link control points to avoid collisions with the bounding orbits.
        """
        try:
            from scipy.optimize import minimize  # lazy import
        except ImportError:
            raise RuntimeError("Scipy import failed. Check its installation.")

        def flatten_cps(cps):
            return numpy.array([cp.normalized_in_3d() for cp in cps]).flatten()

        def unflatten_cps(cps_flat):
            return [PointHomogeneous([1, cps_flat[i], cps_flat[i + 1], cps_flat[i + 2]])
                    for i in range(0, len(cps_flat), 3)]

        flattened_orbits = []
        for i in range(len(bounding_orbits)):
            for j in range(len(bounding_orbits[i])):
                flattened_orbits.extend(bounding_orbits[i][j][1:])

        orbit_centers = [orbit.center.normalized_in_3d() for orbit in flattened_orbits]
        orbit_radii = [orbit.radius for orbit in flattened_orbits]

        init_cps = flatten_cps(init_points)
        lambda_reg = 0.1

        def loss(params):
            cps = unflatten_cps(params)
            margin = 0.01
            penalty = 0.0
            for cp in cps:
                for i, orbit in enumerate(flattened_orbits):
                    dist = numpy.linalg.norm(cp.normalized_in_3d() - orbit_centers[i])
                    if dist < orbit_radii[i] + margin:
                        penalty += (orbit_radii[i] + margin - dist) ** 2
            # Regularization: keep cps close to initial guess
            penalty += lambda_reg * numpy.sum((params - init_cps) ** 2)
            return penalty

        res = minimize(loss, init_cps)

        if not res.success:
            raise RuntimeError(f'Optimization failed: {res.message}')
        else:
            new_control_points = unflatten_cps(res.x)

        return new_control_points

    def obtain_global_bounding_balls(self,
                                     segment_id_number: int,
                                     reduced_indices: list[int],
                                     min_splits: int = 20):
        """
        Obtain global covering balls for a segment to optimize it as a curved link.

        :param segment_id_number: The index of the segment to optimize.
        :param reduced_indices: Indices of segments to consider for bounding balls.
        :param min_splits: Minimum number of splits for the bezier curves.
        """

        t = symbols('t')
        motions = []
        for i, idx in enumerate(reduced_indices):
            rel_motion = self.mechanism.relative_motion(segment_id_number, idx)
            motions.append(RationalCurve([Poly(c, t, greedy=False)
                                          for c in rel_motion],
                                         metric=self.metric))

        bezier_splits = [motion.split_in_beziers(min_splits) for motion in motions]

        all_orbits = []
        for i, segment_idx in enumerate(reduced_indices):

            split_idx = i
            p0_idx = segment_idx - 1
            p1_idx = segment_idx

            rel_bezier_splits = bezier_splits[split_idx]

            p0 = self.mechanism_points[p0_idx]
            p1 = self.mechanism_points[p1_idx]

            orbits0 = [PointOrbit(*p0.get_point_orbit(acting_center=split.ball.center,
                                                      acting_radius=split.ball.radius_squared,
                                                      metric=self.metric),
                                  t_interval=split.t_param_of_motion_curve)
                       for split in rel_bezier_splits]
            orbits1 = [PointOrbit(*p1.get_point_orbit(acting_center=split.ball.center,
                                                      acting_radius=split.ball.radius_squared,
                                                      metric=self.metric),
                                  t_interval=split.t_param_of_motion_curve)
                       for split in rel_bezier_splits]

            all_orbits_of_a_link = []
            for i in range(len(orbits0)):
                orbits_for_t = [orbits0[i].t_interval, orbits0[i]]
                dist = numpy.linalg.norm(orbits0[i].center.normalized_in_3d() - orbits1[
                    i].center.normalized_in_3d())
                radius_sum = orbits0[i].radius + orbits1[i].radius
                if dist > radius_sum:
                    add_balls = dist / radius_sum
                    num_steps = int(add_balls) * 2 + 1

                    # linear interpolation from smaller ball to bigger ball
                    radii = 0
                    radius_diff = orbits1[i].radius - orbits0[i].radius
                    center_diff = orbits1[i].center - orbits0[i].center
                    for j in range(1, num_steps):
                        new_radius = orbits0[i].radius + j * radius_diff / num_steps
                        radii += new_radius
                        new_center = orbits0[i].center + 2 * radii * center_diff / (
                                    dist * 2)
                        orbits_for_t.append(PointOrbit(new_center, new_radius ** 2,
                                                       orbits0[i].t_interval))
                orbits_for_t.append(orbits1[i])
                all_orbits_of_a_link.append(orbits_for_t)
            all_orbits.append(all_orbits_of_a_link)

        return all_orbits

    @staticmethod
    def quantify_collision(segment0: LineSegment,
                           segment1: LineSegment,
                           t_val):
        """
        Quantify the collision between two line segments.
        """
        p00 = segment0.point0.evaluate(t_val)
        p01 = segment0.point1.evaluate(t_val)

        p10 = segment1.point0.evaluate(t_val)
        p11 = segment1.point1.evaluate(t_val)

        l0 = NormalizedLine.from_two_points(p00, p01)
        l1 = NormalizedLine.from_two_points(p10, p11)

        pts, dist, cos_alpha = l0.common_perpendicular_to_other_line(l1)

        if numpy.isclose(dist, 0.0):  # lines are intersecting
            quantif = CollisionAnalyser.quatif_intersection_location(
                PointHomogeneous.from_3d_point(pts[0]),
                p00,
                p01)
        else:  # lines are not intersecting, there is a distance
            quantif_l0 = CollisionAnalyser.quatif_intersection_location(
                PointHomogeneous.from_3d_point(pts[0]),
                p00,
                p01)
            quantif_l1 = CollisionAnalyser.quatif_intersection_location(
                PointHomogeneous.from_3d_point(pts[1]),
                p10,
                p11)
            quantif_dist = CollisionAnalyser.map_to_exponential_decay(dist, k=10.0)

            quantif = quantif_dist * (quantif_l0 + quantif_l1)

        return quantif

    @staticmethod
    def quatif_intersection_location(interection_pt, segment_pt0, segment_pt1):
        a = numpy.linalg.norm(
            segment_pt0.normalized_in_3d() - interection_pt.normalized_in_3d())
        b = numpy.linalg.norm(
            segment_pt1.normalized_in_3d() - interection_pt.normalized_in_3d())

        segment_lenght = numpy.linalg.norm(
            segment_pt0.normalized_in_3d() - segment_pt1.normalized_in_3d())
        if a + b > segment_lenght:
            val = a + b - segment_lenght
            quantif_val = CollisionAnalyser.map_to_exponential_decay(val, k=2.0)
        else:
            val = a if a < b else b
            quantif_val = CollisionAnalyser.map_to_range_inside(val, segment_lenght, 2)

        return quantif_val

    @staticmethod
    def map_to_exponential_decay(x, k=1.0):
        """
        Maps a value x using an exponential decay function to the range [0, 1].

        Maps a value x from the range [0, inf) to the interval [0, 1] using an
        exponential decay function.

        :param float x: The input value in the range [0, inf).
        :param float k: The decay rate (must be positive). Default is 1.0.

        :return: The mapped value in the range [0, 1].
        :rtype: float
        """
        if x < 0:
            raise ValueError("x must be non-negative")
        if k <= 0:
            raise ValueError("k must be positive")

        # Exponential decay formula
        y = numpy.exp(-k * x)
        return y

    @staticmethod
    def map_to_range_inside(x, x_max, weight=1.0):
        """
        Maps a value x from the range [0, x_max] to a value y in the range [1, 2].

        Function preserves the ratio of the input value relative to the maximum value.

        :param float x: The input value in the range [0, x_max].
        :param float x_max: The maximum value of the input range.
        :param float weight: The weight to scale the output value. Default is 1.0.

        :return: The mapped value in the range [1, 2].
        :rtype: float
        """
        if x < 0 or x > x_max:
            raise ValueError("x must be in the range [0, x_max]")
        if x_max <= 0:
            raise ValueError("x_max must be greater than 0")

        # linear mapping formula
        y = 1 + (x / x_max) * weight
        return y
