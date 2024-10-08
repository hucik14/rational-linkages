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

        self.segments = {}
        for segment in mechanism.segments:
            self.segments[segment.id] = segment

        self.motions = self.get_motions()
        self.bezier_splits = self.get_bezier_splits(100)

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
        rel_bezier_splits = self.bezier_splits[split_idx]

        orbits0 = [PointOrbit(*p0.get_point_orbit(acting_center=split.ball.center,
                                                  acting_radius=split.ball.radius_squared,
                                                  metric=self.metric))
                   for split in rel_bezier_splits]
        orbits1 = [PointOrbit(*p1.get_point_orbit(acting_center=split.ball.center,
                                                  acting_radius=split.ball.radius_squared,
                                                  metric=self.metric))
                   for split in rel_bezier_splits]

        inner_orbits = []
        # for i in range(len(orbits0)):
        for i in range(1):
            dist = numpy.linalg.norm(orbits0[i].center.normalized_in_3d() - orbits1[i].center.normalized_in_3d())
            radius_sum = orbits0[i].radius + orbits1[i].radius
            radius_ratio = orbits0[i].radius / orbits1[i].radius
            if dist > radius_sum:
                add_balls = dist / radius_sum - 1
                num_steps = int(add_balls) + 3

                # linear interpolation from smaller ball to bigger ball
                radii = 0
                for j in range(1, num_steps):
                    new_radius = orbits0[i].radius + j * (orbits1[i].radius - orbits0[i].radius) / num_steps
                    radii += new_radius
                    new_center = orbits0[i].center + 2 * radii * (orbits1[i].center - orbits0[i].center) / (dist + radius_sum)
                    inner_orbits.append(PointOrbit(new_center, new_radius))

        return orbits0, orbits1, inner_orbits

    def check_two_objects(self, obj0, obj1):
        """
        Check if two objects collide.
        """
        obj0_type = self.get_object_type(obj0)
        obj1_type = self.get_object_type(obj1)

        if obj0_type == 'is_miniball' and obj1_type == 'is_miniball':
            return self.check_two_miniballs(obj0, obj1)

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
        center_distance = numpy.linalg.norm(ball0.center.coordinates - ball1.center.coordinates)
        return center_distance < ball0.radius + ball1.radius
