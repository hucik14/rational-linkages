from sympy.plotting.intervalmath import interval

from .RationalMechanism import RationalMechanism
from .RationalCurve import RationalCurve
from .MiniBall import MiniBall
from .DualQuaternion import DualQuaternion
from .PointHomogeneous import PointOrbit

import numpy
import sympy


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
        self.bezier_splits = self.get_bezier_splits(50)

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

        t = sympy.symbols('t')

        motions = []
        for motion in relative_motions:
            motions.append(RationalCurve([sympy.Poly(c, t) for c in motion],
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
        import time

        segment = self.segments[segment_id]

        if segment.type == 'l' or segment.type == 't' or segment.type == 'b':
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
    def get_object_type(obj):
        """
        Get the type of an object.
        """
        if isinstance(obj, MiniBall):
            return 'is_miniball'

    @staticmethod
    def check_two_miniballs(ball0, ball1):
        """
        Check if two miniballs collide.
        """
        diff = ball0.center.coordinates - ball1.center.coordinates
        center_dist_squared = numpy.dot(diff, diff)
        return center_dist_squared < ball0.radius_squared + ball1.radius_squared
