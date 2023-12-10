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
        # get parametric equations of links and joints after acting on them (full cycle)
        joints, links = self._get_links_and_joints_acted_equations()

        # create a dictionary of links and joints
        lines = {f"link_{i}{j}": links[i][j] for i in range(len(links))
                 for j in range(len(links[i]))}
        lines.update({f"joint_{i}{j}": joints[i][j] for i in range(len(joints))
                      for j in range(len(joints[i]))})

        for key_i, line0 in lines.items():
            for key_j, line1 in lines.items():
                if key_i < key_j:  # Avoid redundant checks
                    # check if the lines are colliding
                    collisions, points = self.colliding_lines(line0, line1)

                    if collisions is not None:
                        # check if the collision is between the physical line segments
                        physical_collision = [False] * len(collisions)


                        #for i, t in enumerate(collisions):
                        #    physical_collision[i] = self.check_line_segments_collisions(
                        #        key_i, key_j, points[i], t)

                        print(f"{key_i} and {key_j} collide: {collisions} and the "
                              f"collision is physical: {physical_collision}")

    def colliding_lines(self, l0, l1) -> tuple[list[float], list[PointHomogeneous]]:
        """
        Return the lines that are colliding in the linkage.
        """
        from sympy import Poly, Symbol, solve, simplify
        t = Symbol("t")

        # lines are colliding if expr == 0
        expr = np.dot(l0.direction, l1.moment) + np.dot(l0.moment, l1.direction)

        # neighbouring lines are colliding all the time (expr == 0)
        if simplify(expr) == 0:
            return None, None

        p = Poly(expr, t)
        r = solve(expr, t)

        expr_coeffs = Poly(expr, t).all_coeffs()

        # convert to numpy polynomial
        expr_n = np.array(expr_coeffs, dtype="float64")
        # TODO: check the domain
        np_poly = np.polynomial.polynomial.Polynomial(expr_n[::-1], domain=[-100000000000000000000000000, 100000000000000000000000000])

        # solve for t
        colliding_lines_sol = np_poly.roots()
        # extract real solutions
        t_real = colliding_lines_sol.real[abs(colliding_lines_sol.imag) < 1e-5]

        intersetion_points = self.get_intersection_points(l0, l1, t_real)

        return t_real, intersetion_points

    def get_intersection_points(self, l0, l1, t_real: list[float]):
        from sympy import Symbol, Expr
        from PointHomogeneous import PointHomogeneous
        t = Symbol("t")

        intersection_points = [PointHomogeneous()] * len(t_real)

        for i, val in enumerate(t_real):
            # p0 = Expr(np.dot(l0.direction, l1.moment)).subs(t, t_real[0])
            point = np.cross(l0.moment, l1.direction)
            point = [Expr(point[i]).subs(t, val) for i in range(len(point))]
            point = [point[j].args[0] for j in range(len(point))]
            point = np.asarray(point, dtype="float64")

            intersection_points[i] = PointHomogeneous.from_3d_point(point)

        return intersection_points


    def _get_links_and_joints_acted_equations(self) -> tuple[list, list]:
        from sympy import Symbol

        t = Symbol("t")

        joints = [[], []]
        links = [[], []]

        # base (static) link has index 0 in the list of the 1st factorization
        links[0].append(self.factorizations[0].base_link(
            self.factorizations[1].linkage[0].points[0]))

        # static joints
        joints[0].append(self.factorizations[0].joint(0))
        joints[1].append(self.factorizations[1].joint(0))

        for i in range(2):
            for j in range(self.factorizations[i].number_of_factors - 1):
                link = self.factorizations[i].link(j)
                link = self.factorizations[i].act(link, end_idx=j, param=t)
                links[i].append(link)
                joint = self.factorizations[i].joint(j)
                joint = self.factorizations[i].act(joint, end_idx=j, param=t)
                joints[i].append(joint)

        # tool (moving - acted) link has index -1 in the list of the 2nd factorization
        tool_link = self.factorizations[0].tool_link(self.factorizations[1].linkage[1].points[1])
        tool_link = self.factorizations[0].act(tool_link, end_idx=1, param=t)
        links[1].append(tool_link)

        return joints, links

    def check_line_segments_collisions(self, l0: str, l1: str, t: float) -> bool:
        """
        Check if the line segments are colliding at the given time.
        """

