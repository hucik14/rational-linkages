from typing import Union
from copy import deepcopy

import sympy as sp

from .DualQuaternion import DualQuaternion
from .RationalCurve import RationalCurve
from .RationalDualQuaternion import RationalDualQuaternion
from .TransfMatrix import TransfMatrix
from .PointHomogeneous import PointHomogeneous
from .Quaternion import Quaternion


class MotionInterpolation:
    """
    Method for interpolation of poses by rational motion curve in SE(3).

    There are two methods for interpolation of poses by rational motion curve, please
    see the following examples for more details.

    :see also: :ref:`interpolation_background`, :ref:`interpolation_examples`

    :examples:

    .. testcode::

        # 4-pose interpolation

        from rational_linkages import (DualQuaternion, Plotter, FactorizationProvider,
                                       MotionInterpolation, RationalMechanism)


        if __name__ == "__main__":
            # 4 poses
            p0 = DualQuaternion()  # identity
            p1 = DualQuaternion.as_rational([0, 0, 0, 1, 1, 0, 1, 0])
            p2 = DualQuaternion.as_rational([1, 2, 0, 0, -2, 1, 0, 0])
            p3 = DualQuaternion.as_rational([3, 0, 1, 0, 1, 0, -3, 0])

            # obtain the interpolated motion curve
            c = MotionInterpolation.interpolate([p0, p1, p2, p3])

            # factorize the motion curve
            f = FactorizationProvider().factorize_motion_curve(c)

            # create a mechanism from the factorization
            m = RationalMechanism(f)

            # create an interactive plotter object, with 1000 descrete steps
            # for the input rational curves, and arrows scaled to 0.5 length
            myplt = Plotter(interactive=True, steps=1000, arrows_length=0.5)
            myplt.plot(m, show_tool=True)

            # plot the poses
            for pose in [p0, p1, p2, p3]:
                myplt.plot(pose)

            # show the plot
            myplt.show()

    .. testcode::

        # 3-pose interpolation

        from rational_linkages import DualQuaternion, Plotter, MotionInterpolation


        if __name__ == "__main__":
            p0 = DualQuaternion([0, 17, -33, -89, 0, -6, 5, -3])
            p1 = DualQuaternion([0, 84, -21, -287, 0, -30, 3, -9])
            p2 = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3])

            c = MotionInterpolation.interpolate([p0, p1, p2])

            plt = Plotter(interactive=False, steps=500, arrows_length=0.05)
            plt.plot(c, interval='closed')

            for i, pose in enumerate([p0, p1, p2]):
                plt.plot(pose, label='p{}'.format(i+1))
    """
    def __init__(self):
        """
        Creates a new instance of the rational motion interpolation class.
        """
        pass

    @staticmethod
    def interpolate(poses: list[Union[DualQuaternion, TransfMatrix]]) -> RationalCurve:
        """
        Interpolates the given 3 poses by a rational motion curve in SE(3).

        :param list[Union[DualQuaternion, TransfMatrix]] poses: The poses to
            interpolate.

        :return: The rational motion curve.
        :rtype: RationalCurve
        """
        # check number of poses
        if not (2 <= len(poses) <= 4):
            raise ValueError('The number of poses must be 2, 3 or 4.')

        rational_poses = []

        # convert poses to rational dual quaternions
        for pose in poses:
            if isinstance(pose, TransfMatrix):
                rational_poses.append(DualQuaternion.as_rational(pose.matrix2dq()))
            elif isinstance(pose, DualQuaternion) and not pose.is_rational:
                rational_poses.append(DualQuaternion.as_rational(pose.array()))
            elif isinstance(pose, DualQuaternion) and pose.is_rational:
                rational_poses.append(pose)
            else:
                raise TypeError('The given poses must be either TransfMatrix '
                                 'or DualQuaternion.')

        # normalize the poses on Study quadric
        rational_poses = [pose.back_projection() for pose in rational_poses]

        # interpolate the rational poses
        if len(rational_poses) == 4:
            curve_eqs = MotionInterpolation.interpolate_cubic(rational_poses)
            return RationalCurve(curve_eqs)
        elif len(rational_poses) == 3:
            curve_eqs = MotionInterpolation.interpolate_quadratic(rational_poses)
            return RationalCurve(curve_eqs)
        elif len(rational_poses) == 2:
            curve_eqs = MotionInterpolation.interpolate_quadratic_2poses(rational_poses)
            return RationalCurve(curve_eqs)

    @staticmethod
    def interpolate_quadratic(poses: list[DualQuaternion]) -> list[sp.Poly]:
        """
        Interpolates the given 3 rational poses by a quadratic curve in SE(3).

        :param list[DualQuaternion] poses: The rational poses to interpolate.

        :return: The rational motion curve.
        :rtype: list[sp.Poly]
        """
        alpha = sp.Symbol('alpha')
        omega = sp.Symbol('omega')
        t = sp.Symbol('t')

        p0 = poses[0].array()
        p1 = poses[1].array()
        p2 = poses[2].array()

        c = alpha * p0 + (p1 - alpha * p0 - omega * p2) * t + omega * p2 * t**2
        symbolic_curve = RationalDualQuaternion(c)

        # apply Stydy condition, i.e. obtain epsilon norm of the curve
        study_norm = symbolic_curve.norm()

        # simplify the norm
        study_norm = sp.simplify(study_norm[4] / (t * (t - 1)))

        # obtain the equations for alpha and omega
        eq0 = study_norm.subs(t, 0)
        eq1 = study_norm.subs(t, 1)

        # solve the equations symbolically
        sols = sp.solve([eq0, eq1], [alpha, omega], dict=True)

        # get non zero solution
        nonzero_sol = None
        for sol in sols:
            if sol[alpha] and sol[omega]:
                if (not (sol[alpha] == 0 and sol[omega] == 0)
                        and sol[alpha].is_number
                        and sol[omega].is_number):
                    nonzero_sol = sol

        if nonzero_sol is None:
            raise ValueError('Interpolation failed for the given poses.')

        al = nonzero_sol[alpha]
        om = nonzero_sol[omega]
        # obtain the interpolated curve
        c_interp = al * p0 + (p1 - al * p0 - om * p2) * t + om * p2 * t**2

        # list of polynomials
        poly = [sp.Poly(el, t) for el in c_interp]

        return poly

    @staticmethod
    def interpolate_quadratic_2poses(poses: list[DualQuaternion]) -> list[sp.Poly]:
        """
        Interpolates the given 2 rational poses by a quadratic curve in SE(3).

        Adds the 3rd pose that is either identity or a random pose that returns
        solution.

        :param list[DualQuaternion] poses: The rational poses to interpolate.

        :return: Polynomials of rational motion curve.
        :rtype: list[sp.Poly]
        """
        # Try to interpolate with identity pose
        try:
            return MotionInterpolation.interpolate_quadratic(
                poses + [DualQuaternion.as_rational([1, 0, 0, 0, 0, 0, 0, 0])])
        except Exception:
            print('Failed to interpolate with identity pose. Trying to interpolate '
                  'with random poses...')

            # Calculate the mid point between the two poses
            p0 = PointHomogeneous(poses[0].dq)
            p1 = PointHomogeneous(poses[1].dq)
            mid_p = p0.linear_interpolation(p1, 0.5)
            mid_pose = DualQuaternion(mid_p.array())

            # get mean value of mid_pose coordinates
            mean = sum(mid_pose.array()) / len(mid_pose.array())
            # TODO: implement better random pose generation using mid_pose

            shortest_curve_length = float('inf')
            shortest_set = None
            best_pose = None

            for i in range(1, 30):
                additional_pose = DualQuaternion.as_rational(
                    DualQuaternion.random_on_study_quadric(
                        mean * 0.2 * i).array()).back_projection()

                new_poses = deepcopy(poses)
                new_poses.append(additional_pose)

                try:
                    polynomial_set = MotionInterpolation.interpolate_quadratic(new_poses)
                except Exception:
                    polynomial_set = None

                # If interpolation was successful, check if it's the best so far
                if polynomial_set is not None:
                    new_curve_length = RationalCurve(polynomial_set).get_path_length(
                        num_of_points=500)
                    if new_curve_length < shortest_curve_length:
                        shortest_set = polynomial_set
                        best_pose = additional_pose
                        shortest_curve_length = new_curve_length

            if shortest_set is not None:
                print('Chosen pose:')
                print(best_pose)
                print(shortest_curve_length)

                return shortest_set

            else:  # if no solution was found
                raise ValueError('Interpolation failed for the given poses.')

    @staticmethod
    def interpolate_cubic(poses: list[DualQuaternion]) -> list[sp.Poly]:
        """
        Interpolates the given 4 rational poses by a cubic curve in SE(3).

        The 4 poses span a projective 3-space, which is intersected with Study quadric.
        This intersection gives another quadric containing all 4 poses, and it also
        contains cubic curves if it contains lines. The algorithm later searches
        for one of the cubic curves that interpolates the 4 poses.

        :see also: :ref:`interpolation_background`

        :param list[DualQuaternion] poses: The rational poses to interpolate.

        :return: The rational motion curve.
        :rtype: list[sp.Poly]

        :raises ValueError: If the interpolation has no solution, 'k' does not exist.
        """
        # obtain additional dual quaternions k1, k2
        try:
            k = MotionInterpolation._obtain_k_dq(poses)
        except Exception:
            raise ValueError('Interpolation has no solution.')

        # solve for t[i] - the parameter of the rational motion curve for i-th pose
        t_sols = MotionInterpolation._solve_for_t(poses, k)

        # Lagrange's interpolation part
        # lambdas for interpolation - scalar multiples of the poses
        lams = sp.symbols("lams1:5")

        parametric_points = [sp.Matrix(poses[0].array()),
                             sp.Matrix(lams[0] * poses[1].array()),
                             sp.Matrix(lams[1] * poses[2].array()),
                             sp.Matrix(lams[2] * poses[3].array())]

        # obtain the Lagrange interpolation for poses p0, p1, p2, p3
        interp = MotionInterpolation._lagrange_poly_interpolation(parametric_points)

        t = sp.symbols("t:4")
        x = sp.symbols("x")

        # to avoid calculation with infinity, substitute t[i] with 1/t[i]
        temp = [element.subs(t[0], 0) for element in interp]
        temp2 = [element.subs(x, 1 / x) for element in temp]
        temp3 = [sp.together(element * x ** 3) for element in temp2]
        temp4 = [sp.together(element.subs({t[1]: 1 / t_sols[0],
                                           t[2]: 1 / t_sols[1],
                                           t[3]: 1 / t_sols[2]}))
                 for element in temp3]

        # obtain additional parametric pose p4
        lam = sp.symbols("lam")
        poses.append(DualQuaternion([lam, 0, 0, 0, 0, 0, 0, 0]) - k[0])

        eqs_lambda = [element.subs(x, lam) - lams[-1] * poses[-1].array()[i]
                      for i, element in enumerate(temp4)]

        sols_lambda = sp.solve(eqs_lambda, lams, domain='RR')

        # obtain the family of solutions
        poly = [element.subs(sols_lambda) for element in temp4]

        # choose one solution by setting lambda, in this case lambda = 0
        poly = [element.subs(lam, 0) for element in poly]

        t = sp.Symbol("t")
        poly = [element.subs(x, t) for element in poly]

        return [sp.Poly(element, t) for element in poly]

    @staticmethod
    def _obtain_k_dq(poses: list[DualQuaternion]) -> list[DualQuaternion]:
        """
        Obtain additional dual quaternions k1, k2 for interpolation of 4 poses.

        :param list[DualQuaternion] poses: The rational poses to interpolate.

        :return: Two additional dual quaternions for interpolation.
        :rtype: list[DualQuaternion]
        """
        x = sp.symbols("x:3")

        k = DualQuaternion(poses[0].array() + x[0] * poses[1].array()
                           + x[1] * poses[2].array() + x[2] * poses[3].array())

        eqs = [k[0], k[4], k.norm().array()[4]]

        sol = sp.solve(eqs, x, domain=sp.S.Reals)

        k_as_expr = [sp.Expr(el) for el in k]

        k1 = [el.subs({x[0]: sol[0][0], x[1]: sol[0][1], x[2]: sol[0][2]})
              for el in k_as_expr]
        k2 = [el.subs({x[0]: sol[1][0], x[1]: sol[1][1], x[2]: sol[1][2]})
              for el in k_as_expr]

        k1_dq = DualQuaternion([el.args[0] for el in k1])
        k2_dq = DualQuaternion([el.args[0] for el in k2])

        return [k1_dq, k2_dq]

    @staticmethod
    def _solve_for_t(poses: list[DualQuaternion], k: list[DualQuaternion]):
        """
        Solve for t[i] - the parameter of the rational motion curve for i-th pose.

        :param list[DualQuaternion] poses: The rational poses to interpolate.
        :param list[DualQuaternion] k: The additional dual quaternions for interpolation.

        :return: The solutions for t[i].
        :rtype: list
        """
        t = sp.symbols("t:3")

        study_cond_mat = sp.Matrix([[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]])

        t_dq = [DualQuaternion([t[i], 0, 0, 0, 0, 0, 0, 0]) for i in range(3)]

        eqs = [sp.Matrix((t_dq[0] - k[0]).array()).transpose() @ study_cond_mat
               @ sp.Matrix(poses[1].array()),
               sp.Matrix((t_dq[1] - k[0]).array()).transpose() @ study_cond_mat
               @ sp.Matrix(poses[2].array()),
               sp.Matrix((t_dq[2] - k[0]).array()).transpose() @ study_cond_mat
               @ sp.Matrix(poses[3].array())]

        sols_t = sp.solve(eqs, t)

        # covert to list and retrun
        return [val for i, (key, val) in enumerate(sols_t.items())]

    @staticmethod
    def _lagrange_polynomial(degree, index, x, t):
        """
        Calculate the Lagrange polynomial for interpolation.

        :param int degree: The degree of the Lagrange polynomial.
        :param int index: The index of the Lagrange polynomial.
        :param symbol x: The interpolation point (indeterminate).
        :param list[symbol] t: The interpolation nodes.

        :return: The Lagrange polynomial.
        :rtype: sp.Expr
        """
        lagrange_poly = 1
        for i in range(degree + 1):
            if i != index:
                lagrange_poly *= (x - t[i]) / (t[index] - t[i])
        return lagrange_poly

    @staticmethod
    def _lagrange_poly_interpolation(poses: list[sp.Matrix]):
        """
        Calculate the interpolation polynomial using Lagrange interpolation.

        :param list[sp.Matrix] poses: The poses to interpolate.

        :return: The interpolation polynomial.
        :rtype: sp.Matrix
        """
        # indeterminate x
        x = sp.symbols('x')

        # interpolation nodes
        t = sp.symbols("t:4")

        degree = len(poses) - 1
        result = sp.Matrix([0, 0, 0, 0, 0, 0, 0, 0])

        for i in range(degree + 1):
            result += poses[i] * MotionInterpolation._lagrange_polynomial(degree,
                                                                          i, x, t)
        return result
