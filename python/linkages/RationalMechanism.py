import numpy as np
from copy import deepcopy

from RationalCurve import RationalCurve
from DualQuaternion import DualQuaternion
from matplotlib import pyplot as plt
from TransfMatrix import TransfMatrix
from matplotlib.widgets import Slider
from MotionFactorization import MotionFactorization


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
        from NormalizedLine import NormalizedLine
        from sympy import Symbol

        t = Symbol("t")

        l0 = NormalizedLine.from_two_points(self.factorizations[0].linkage[0].points[1],
                                            self.factorizations[0].linkage[1].points[0])
        l1 = NormalizedLine.from_two_points(self.factorizations[0].linkage[1].points[1],
                                            self.factorizations[1].linkage[1].points[1])

        l0_acted = self.factorizations[0].act(l0, start_idx=0, end_idx=0, param=t)
        l1_acted = self.factorizations[0].act(l1, start_idx=0, end_idx=1, param=t)

        l0 = NormalizedLine.from_two_points(self.factorizations[0].linkage[0].points[1],
                                            self.factorizations[0].linkage[0].points[0])
        self.colliding_lines(l0, l0_acted)

        lines = []
        # static lines
        l00 = self.factorizations[0].base_link(self.factorizations[1].linkage[0].points[0])
        j00 = self.factorizations[0].joint(0)
        j10 = self.factorizations[1].joint(0)

        # j00 actuation
        l01 = self.factorizations[0].link(0)
        l01 = self.factorizations[0].act(l01, end_idx=0, param=t)
        j01 = self.factorizations[0].joint(1)
        j01 = self.factorizations[0].act(j01, end_idx=0, param=t)

        # j10 actuation
        l11 = self.factorizations[1].link(0)
        l11 = self.factorizations[1].act(l01, end_idx=0, param=t)
        j11 = self.factorizations[1].joint(1)
        j11 = self.factorizations[1].act(j01, end_idx=0, param=t)

        # j01 actuaion is the same as j11 actuation
        l_t = self.factorizations[0].tool_link(self.factorizations[1].linkage[1].points[1])
        l_t = self.factorizations[0].act(l_t, start_idx=0, param=t)

        lines.append(l00)
        lines.append(j00)
        lines.append(j10)
        lines.append(l01)
        lines.append(j01)
        lines.append(l11)
        lines.append(j11)
        lines.append(l_t)

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line0 = lines[i]
                line1 = lines[j]
                collision = self.colliding_lines(line0, line1)
                print(f"Lines {i} and {j} collide: {collision}")

    def colliding_lines(self, l0, l1):
        """
        Return the lines that are colliding in the linkage.
        """
        from sympy import Poly, Symbol
        t = Symbol("t")

        # lines are colliding if expr == 0
        expr = np.dot(l0.direction, l1.moment) + np.dot(l0.moment, l1.direction)

        # neibouring lines are colliding all the time (expr == 0)
        if expr == 0:
            print(" --- neighbouring lines are colliding ---")
            return []

        expr_coeffs = Poly(expr, t).all_coeffs()

        # convert to numpy polynomial
        expr_n = np.array(expr_coeffs, dtype="float64")
        np_poly = np.polynomial.polynomial.Polynomial(expr_n[::-1])

        # solve for t
        colliding_lines_sol = np_poly.roots()
        # extract real solutions
        t_real = colliding_lines_sol.real[abs(colliding_lines_sol.imag) < 1e-5]

        print(expr_coeffs)
        print(t_real)

        return t_real



