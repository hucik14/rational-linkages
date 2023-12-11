import numpy as np
from copy import deepcopy

import pylab as p

from RationalCurve import RationalCurve
from DualQuaternion import DualQuaternion
from matplotlib import pyplot as plt
from TransfMatrix import TransfMatrix
from matplotlib.widgets import Slider
from MotionFactorization import MotionFactorization

PointHomogeneous = 'PointHomogeneous'


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
        # update the line segments
        self.segments = self._get_line_segments_of_linkage()

        for i in range(len(self.segments)):
            for j in range(i + 2, len(self.segments)):
                collisions, points = self.colliding_lines(self.segments[i].equation, self.segments[j].equation)

                if collisions is not None:
                    # check if the collision is between the physical line segments
                    physical_collision = [False] * len(collisions)

                    for k, t_val in enumerate(collisions):
                        # get the intersection point
                        p = points[k]
                        # check if the intersection point is on the physical line segments
                        physical_collision[k] = self.segments[i].is_point_in_segment(p, t_val) and self.segments[j].is_point_in_segment(p, t_val)
                        #print(physical_collision[k])
                        self.segments[i].is_point_in_segment(p, t_val)
                        self.segments[j].is_point_in_segment(p, t_val)

                    print(f"{self.segments[i].type}_{self.segments[i].factorization_idx}{self.segments[i].idx} X {self.segments[j].type}_{self.segments[j].factorization_idx}{self.segments[j].idx}: {collisions}, physical: {physical_collision}")

    def colliding_lines(self, l0, l1) -> tuple[list[float], list[PointHomogeneous]]:
        """
        Return the lines that are colliding in the linkage.
        """
        from sympy import Poly, Symbol, solve, simplify, nroots
        t = Symbol("t")
        import sympy as sp

        # lines are colliding if expr == 0
        expr = simplify(np.dot(l0.direction, l1.moment) + np.dot(l0.moment, l1.direction))

        # neighbouring lines are colliding all the time (expr == 0)
        if expr == 0:
            return None, None

        #sp_sol = nroots(expr, n=7)
        #sp_sol = solve(expr, t)
        #real_solutions = [sol for sol in sp_sol if sol.is_real]
        #print(real_solutions)


        expr_coeffs = Poly(expr, t).all_coeffs()

        # convert to numpy polynomial
        expr_n = np.array(expr_coeffs, dtype="float64")
        # TODO: check the domain
        np_poly = np.polynomial.polynomial.Polynomial(expr_n[::-1], domain=[-1, 1])

        # solve for t
        colliding_lines_sol = np_poly.roots()
        # extract real solutions
        t_real = colliding_lines_sol.real[abs(colliding_lines_sol.imag) < 1e-5]

        if sp.limit(expr, t, sp.oo) == sp.oo:
            t_real = np.append(t_real, np.inf)
        elif sp.limit(expr, t, sp.oo) == -sp.oo:
            t_real = np.append(t_real, -np.inf)

        intersection_points = self.get_intersection_points(l0, l1, t_real)

        return t_real, intersection_points

    def get_intersection_points(self, l0, l1, t_real: list[float]):
        from PointHomogeneous import PointHomogeneous
        intersection_points = [PointHomogeneous()] * len(t_real)

        for i, t_val in enumerate(t_real):
            l0e = l0.evaluate(t_val)
            l1e = l1.evaluate(t_val)
            p, dist, cos_angle = l0e.common_perpendicular_to_other_line(l1e)
            intersection_points[i] = PointHomogeneous.from_3d_point(p[0])

        return intersection_points

    def _get_line_segments_of_linkage(self) -> list:
        from sympy import Symbol
        from Linkage import LineSegment

        t = Symbol("t")

        segments = [[], []]

        # base (static) link has index 0 in the list of the 1st factorization
        eq, p0, p1 = self.factorizations[0].base_link(self.factorizations[1].linkage[0].points[0])
        s = LineSegment(eq, p0, p1, linkage_type="b", f_idx=0, idx=0)
        segments[0].append(s)

        # static joints
        segments[0].append(LineSegment(*self.factorizations[0].joint(0),
                                       linkage_type="j", f_idx=0, idx=0))
        segments[1].append(LineSegment(*self.factorizations[1].joint(0),
                                       linkage_type="j", f_idx=1, idx=0))

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


