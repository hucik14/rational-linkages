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
    """Analyze collisions for a rational mechanism.

    The analyser computes motion representations, point orbits and bounding
    balls for segments of a mechanism and provides utilities to check and
    quantify collisions between segments and miniballs.

    Parameters
    ----------
    mechanism
        The mechanism to analyze.
    """
    def __init__(self, mechanism: RationalMechanism):
        """Create a CollisionAnalyser for a mechanism.

        Parameters
        ----------
        mechanism
            The mechanism for which collision analysis is performed.
        """
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
        """Split the relative motions into Bezier segments.

        Each relative motion curve of the mechanism is split into a list of
        Bezier segments using the underlying curve's ``split_in_beziers``
        method.

        Parameters
        ----------
        min_splits
            Minimum number of subdivisions requested for each motion (default
            is 0).

        Returns
        -------
        list
            A list (one per motion) of lists of Bezier split objects.
        """
        return [motion.split_in_beziers(min_splits) for motion in self.motions]

    def get_motions(self):
        """Assemble relative motions as rational curves.

        The mechanism stores factorizations as sequences of dual quaternion
        factors. This method composes those factors to obtain the relative
        motions and converts each motion into a :class:`RationalCurve`.

        Returns
        -------
        list
            List of :class:`RationalCurve` objects representing the relative
            motions of the mechanism.
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
        """Return the orbits for all mechanism points.

        Each mechanism point provides a parameterized orbit; this method
        wraps those results into :class:`PointOrbit` instances using the
        analyser's metric.

        Returns
        -------
        list[PointOrbit]
            A list of :class:`PointOrbit` objects, one per mechanism point.
        """
        return [PointOrbit(*point.get_point_orbit(metric=self.metric))
                for point in self.mechanism_points]

    def get_segment_orbit(self, segment_id: str):
        """Compute the orbit covering for a named segment.

        The method determines the two endpoint points for the requested
        segment and returns a list describing the covering of those endpoints'
        orbits with miniballs. For straight links the relative-motion
        Bezier-splits are used; for base-like segments ('b') a heuristic
        miniball is constructed.

        Parameters
        ----------
        segment_id
            Identifier of the segment whose orbit is required.

        Returns
        -------
        list
            A list of per-split entries. Each entry is itself a list where the
            first element is the time-interval metadata and the remaining
            elements are :class:`PointOrbit` miniballs that cover the segment
            endpoints over that interval.
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
            dist = numpy.linalg.norm(orbits0[i].center.normalized_euclidean() - orbits1[i].center.normalized_euclidean())
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
        """Check whether two segments (by id) collide.

        This method computes or reuses cached miniball coverings for the two
        segments and then checks pairwise miniball intersections. If
        ``t_interval`` is provided it restricts the check to the corresponding
        Bezier split that contains the numeric time value.

        Parameters
        ----------
        segment0, segment1
            Segment identifiers to check for collision.
        t_interval, optional
            If provided, restrict checking to a particular parameter interval
            representation. The function expects an interval-format matching
            the miniball t_interval metadata; if ``None`` the whole motion is
            checked.

        Returns
        -------
        bool
            True if any pair of miniballs collide (indicating a collision for
            the segments), False otherwise.
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
        """Determine whether two miniballs intersect.

        Parameters
        ----------
        ball0, ball1
            Objects providing ``center`` (with a ``coordinates`` attribute)
            and ``radius_squared`` attributes. The function computes the
            squared Euclidean distance between centers and compares to the sum
            of squared radii.

        Returns
        -------
        bool
            True if the squared center distance is strictly less than the sum
            of the two ``radius_squared`` values, False otherwise.
        """
        diff = ball0.center.coordinates - ball1.center.coordinates
        center_dist_squared = numpy.dot(diff, diff)
        return center_dist_squared < ball0.radius_squared + ball1.radius_squared

    def get_split_and_point_indices(self, segment):
        """Compute split index and the corresponding point indices.

        The mapping depends on the segment type and its factorization index.

        Parameters
        ----------
        segment
            The segment object whose indices should be derived.

        Returns
        -------
        tuple
            ``(split_idx, p0_idx, p1_idx)`` where ``split_idx`` is the index of
            the relative motion split and ``p0_idx``/``p1_idx`` are indices
            into the ``mechanism_points`` list for the segment endpoints.
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
        """Optimize a link by replacing it with a curved Bezier (RationalSoo).

        The routine builds a set of bounding miniballs from neighboring
        segments, creates an initial control polygon connecting the two
        endpoint joints and runs a constrained optimization to move internal
        control points away from collisions while keeping them near their
        initial positions.

        Parameters
        ----------
        segment_id
            Identifier of the link to replace with a curved version. Joints
            (identifiers starting with 'j') are not supported and will raise
            an exception.
        min_splits, optional
            Minimum number of Bezier splits to use when constructing
            bounding balls (default 20).
        curve_degree, optional
            Degree of the replacement rational Bezier (default 3).

        Returns
        -------
        RationalSoo
            A rational Bezier representing the optimized curved link.

        Raises
        ------
        ValueError
            If a joint identifier is provided instead of a link id.
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
        """Optimize internal control points to avoid collisions with orbits.

        A numerical optimization is performed (using ``scipy.optimize``)
        to adjust the coordinates of the internal control points so that the
        resulting curve stays outside of the supplied miniball coverings.
        A small regularization term keeps the solution close to the initial
        guess.

        Parameters
        ----------
        init_points
            Initial internal control points (excluding the fixed endpoints).
        bounding_orbits
            A nested list of :class:`PointOrbit` miniballs describing forbidden
            regions for the curve.

        Returns
        -------
        list[PointHomogeneous]
            New optimized control points including the same homogeneous form
            as the inputs.

        Raises
        ------
        RuntimeError
            If SciPy is not available or the optimizer fails to converge.
        """
        try:
            from scipy.optimize import minimize  # lazy import
        except ImportError:
            raise RuntimeError("Scipy import failed. Check its installation.")

        def flatten_cps(cps):
            return numpy.array([cp.normalized_euclidean() for cp in cps]).flatten()

        def unflatten_cps(cps_flat):
            return [PointHomogeneous([1, cps_flat[i], cps_flat[i + 1], cps_flat[i + 2]])
                    for i in range(0, len(cps_flat), 3)]

        flattened_orbits = []
        for i in range(len(bounding_orbits)):
            for j in range(len(bounding_orbits[i])):
                flattened_orbits.extend(bounding_orbits[i][j][1:])

        orbit_centers = [orbit.center.normalized_euclidean() for orbit in flattened_orbits]
        orbit_radii = [orbit.radius for orbit in flattened_orbits]

        init_cps = flatten_cps(init_points)
        lambda_reg = 0.1

        def loss(params):
            cps = unflatten_cps(params)
            margin = 0.01
            penalty = 0.0
            for cp in cps:
                for i, orbit in enumerate(flattened_orbits):
                    dist = numpy.linalg.norm(cp.normalized_euclidean() - orbit_centers[i])
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
        """Create miniball coverings for a selection of neighboring segments.

        For each segment index in ``reduced_indices`` a relative motion curve is
        constructed and split into Bezier pieces; for each Bezier split the
        orbit of the two defining points is covered by miniballs. The result
        can be used to define forbidden regions for an optimization of a
        curved replacement link.

        Parameters
        ----------
        segment_id_number
            Index of the segment being optimized (used to compute relative
            motions).
        reduced_indices
            Indices of other segments to include when computing bounding
            balls.
        min_splits, optional
            Minimum number of Bezier splits to request when splitting the
            relative motion curves (default 20).

        Returns
        -------
        list
            Nested list of miniball coverings. The outer list corresponds to
            each considered segment; each inner entry is a list of per-split
            lists with time-interval metadata followed by :class:`PointOrbit`
            miniballs.
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
                dist = numpy.linalg.norm(orbits0[i].center.normalized_euclidean() - orbits1[
                    i].center.normalized_euclidean())
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
        """Provide a scalar measure for proximity or intersection of segments.

        The function evaluates both segments at parameter ``t_val`` and
        computes the common perpendicular between the two supporting lines.
        If the distance is (near) zero an intersection-location based
        quantification is returned; otherwise the distance and the locations
        on both segments are combined into a scalar score in (0, inf).

        Parameters
        ----------
        segment0, segment1
            The two line segments to quantify collision for.
        t_val
            The parameter value at which to evaluate both segment endpoints.

        Returns
        -------
        float
            A non-negative scalar where larger values indicate stronger
            proximity/collision. The exact scaling is implementation-defined.
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
        """Quantify where an intersection point lies relative to a segment.

        The function computes distances from the intersection point to the
        two segment endpoints and maps these distances to a scalar that
        indicates whether the intersection is outside the segment (mapped
        with an exponential decay) or inside the segment (mapped to a range
        near 1..2 using a linear mapping).

        Parameters
        ----------
        interection_pt
            The intersection point to evaluate.
        segment_pt0, segment_pt1
            Endpoints of the segment being tested.

        Returns
        -------
        float
            A scalar quantification of the intersection location. Larger
            values indicate stronger proximity or intersection severity.
        """
        a = numpy.linalg.norm(
            segment_pt0.normalized_euclidean() - interection_pt.normalized_euclidean())
        b = numpy.linalg.norm(
            segment_pt1.normalized_euclidean() - interection_pt.normalized_euclidean())

        segment_lenght = numpy.linalg.norm(
            segment_pt0.normalized_euclidean() - segment_pt1.normalized_euclidean())
        if a + b > segment_lenght:
            val = a + b - segment_lenght
            quantif_val = CollisionAnalyser.map_to_exponential_decay(val, k=2.0)
        else:
            val = a if a < b else b
            quantif_val = CollisionAnalyser.map_to_range_inside(val, segment_lenght, 2)

        return quantif_val

    @staticmethod
    def map_to_exponential_decay(x, k=1.0):
        """Map a non-negative scalar using exponential decay to (0, 1].

        Parameters
        ----------
        x
            Non-negative input value to map.
        k, optional
            Positive decay rate controlling the steepness of the mapping
            (default 1.0).

        Returns
        -------
        float
            Value in (0, 1] given by exp(-k * x).

        Raises
        ------
        ValueError
            If ``x`` is negative or ``k`` is not positive.
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
        """Linearly map a value in [0, x_max] to the interval [1, 1+weight].

        Parameters
        ----------
        x
            Input value in the closed interval [0, x_max].
        x_max
            Positive maximum value for ``x``.
        weight, optional
            Scale of the mapping; the output range becomes [1, 1+weight]
            (default 1.0).

        Returns
        -------
        float
            Linearly scaled value in [1, 1+weight].

        Raises
        ------
        ValueError
            If ``x`` is outside [0, x_max] or ``x_max`` is not positive.
        """
        if x < 0 or x > x_max:
            raise ValueError("x must be in the range [0, x_max]")
        if x_max <= 0:
            raise ValueError("x_max must be greater than 0")

        # linear mapping formula
        y = 1 + (x / x_max) * weight
        return y
