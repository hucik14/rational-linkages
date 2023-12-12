import numpy as np
import sympy as sp
from copy import deepcopy

from RationalCurve import RationalCurve
from DualQuaternion import DualQuaternion
from MotionFactorization import MotionFactorization

PointHomogeneous = 'PointHomogeneous'
NormalizedLine = 'NormalizedLine'


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

        self.segments = self._get_line_segments_of_linkage()

    def get_dh_params(self, alpha_form: str = "cos_alpha") -> tuple:
        """
        Get the Denavit-Hartenberg parameters of the linkage.

        :param alpha_form: str - form of the returned alpha parameter, can be
            "cos_alpha", "deg", or "rad"

        :return: tuple (d, a, alpha)
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

        # Calculate Denavit-Hartenberg parameters for each pair of axis rotations
        pts_prev = lines[-1].common_perpendicular_to_other_line(lines[0])[0]
        for i in range(len(lines) - 1):
            pts, a_i, al_i = lines[i].common_perpendicular_to_other_line(lines[i+1])
            d_i = lines[i].get_point_param(pts_prev[1]) - lines[i].get_point_param(pts[0])
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

        # Convert alpha to the specified form
        match alpha_form:
            case "cos_alpha":
                pass
            case "deg":
                alpha = [np.degrees(np.arccos(alpha_i)) for alpha_i in alpha]
            case "rad":
                alpha = [np.arccos(alpha_i) for alpha_i in alpha]
            case _:
                raise ValueError("alpha_form must be cos_alpha, deg or rad")

        return d, a, alpha
    
    def collision_check(self):
        """
        Perform full-cycle collision check on the line-model linkage between linkage and
        links of the two given factorizations.
        """
        # update the line segments (physical realization of the linkage)
        self.segments = self._get_line_segments_of_linkage()

        for i in range(len(self.segments)):

            # skip neighbouring lines and avoid redundant checks
            for j in range(i + 2, len(self.segments)):
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

                    print(f"{self.segments[i].type}_{self.segments[i].factorization_idx}{self.segments[i].idx} X {self.segments[j].type}_{self.segments[j].factorization_idx}{self.segments[j].idx}: {collisions}, physical: {physical_collision}")

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

        for sol in sol_real_inversed:
            if not np.isclose(sol, 0):
                sol = 1 / sol
                sol_real = np.append(sol_real, sol)
            else:
                sol = np.inf
                sol_real = np.append(sol_real, sol)

        intersection_points = self.get_intersection_points(l0, l1, sol_real)

        return sol_real, intersection_points

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


