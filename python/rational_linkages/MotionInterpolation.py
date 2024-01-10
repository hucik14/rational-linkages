import sympy as sp
from typing import Union

from .DualQuaternion import DualQuaternion
from .TransfMatrix import TransfMatrix
from .RationalCurve import RationalCurve


class MotionInterpolation:
    """
    Method for interpolation of 3 poses and origin by rational motion curve in SE(3).
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
        if len(poses) != 3:
            raise ValueError('The number of poses must be 3.')

        rational_poses = [DualQuaternion()]

        # convert poses to rational dual quaternions
        for pose in poses:
            if isinstance(pose, TransfMatrix):
                rational_poses.append(DualQuaternion.as_rational(pose.matrix2dq()))
            elif isinstance(pose, DualQuaternion) and not pose.is_rational:
                rational_poses.append(DualQuaternion.as_rational(pose.array()))
            elif isinstance(pose, DualQuaternion) and pose.is_rational:
                rational_poses.append(pose)
            else:
                raise ValueError('The given poses must be either TransfMatrix '
                                 'or DualQuaternion.')

        curve_eqs = MotionInterpolation._interpolate_rational_poses(rational_poses)
        return RationalCurve(curve_eqs)

    @staticmethod
    def _interpolate_rational_poses(poses: list[DualQuaternion]) -> list[sp.Poly]:
        """
        Interpolates the given 4 rational poses by a rational motion curve in SE(3).

        :param list[DualQuaternion] poses: The rational poses to interpolate.

        :return: The rational motion curve.
        :rtype: list[sp.Poly]
        """
        # obtain additional dual quaternions k1, k2
        k = MotionInterpolation._obtain_k_dq(poses)

        # solve for t[i] - the parameter of the rational motion curve for i-th pose
        t_sols = MotionInterpolation._solve_for_t(poses, k)

        # lambdas for interpolation
        # TODO: check if this is correct, without lam0
        lams = sp.symbols("lams1:5")

        parametric_points = [sp.Matrix(poses[0].array()),
                             sp.Matrix(lams[0] * poses[1].array()),
                             sp.Matrix(lams[1] * poses[2].array()),
                             sp.Matrix(lams[2] * poses[3].array())]

        interp = MotionInterpolation._lagrange_poly_interpolation(parametric_points)

        t = sp.symbols("t:4")
        x = sp.symbols("x")

        temp = [element.subs(t[0], 0) for element in interp]
        temp2 = [element.subs(x, 1 / x) for element in temp]
        temp3 = [sp.together(element * x ** 3) for element in temp2]
        temp4 = [sp.together(element.subs({t[1]: 1 / t_sols[0], t[2]: 1 / t_sols[1],
                                           t[3]: 1 / t_sols[2]}))
                 for element in temp3]

        # obtain additional parametric pose
        lam = sp.symbols("lam")
        poses.append(DualQuaternion([lam, 0, 0, 0, 0, 0, 0, 0]) - k[0])

        eqs_lambda = [element.subs(x, lam) - lams[-1] * poses[-1].array()[i]
                      for i, element in enumerate(temp4)]

        sols_lambda = sp.solve(eqs_lambda, lams, domain='RR')

        poly = [element.subs(sols_lambda) for element in temp4]
        poly = [element.subs(lam, 0) for element in poly]

        t = sp.Symbol("t")
        poly = [element.subs(x, t) for element in poly]

        return [sp.Poly(element, t) for element in poly]

    @staticmethod
    def _obtain_k_dq(poses: list[DualQuaternion]) -> list[DualQuaternion]:
        """
        Obtain additional dual quaternions K1, K2 for interpolation of 4 poses.

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
