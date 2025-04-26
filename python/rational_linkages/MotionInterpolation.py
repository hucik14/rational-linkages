from typing import Union
from copy import deepcopy
from warnings import warn

import sympy as sp
import numpy as np

from .DualQuaternion import DualQuaternion
from .RationalCurve import RationalCurve
from .RationalDualQuaternion import RationalDualQuaternion
from .TransfMatrix import TransfMatrix
from .PointHomogeneous import PointHomogeneous
from .Quaternion import Quaternion
from .RationalBezier import RationalBezier

from .utils_rust import motion_interp_x3

class MotionInterpolation:
    """
    Method for interpolation of poses by rational motion curve in SE(3).

    There are two methods for interpolation of poses by rational motion curve, please
    see the following examples for more details.

    :see also: :ref:`interpolation_background`, :ref:`interpolation_examples`

    :examples:

    .. testcode:: [motion_interpolation_example1]

        # 4-pose interpolation

        from rational_linkages import (DualQuaternion, Plotter, FactorizationProvider,
                                       MotionInterpolation, RationalMechanism)


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

    .. testcleanup:: [motion_interpolation_example1]

        del DualQuaternion, Plotter, FactorizationProvider, MotionInterpolation
        del RationalMechanism
        del p0, p1, p2, p3, c, f, m, myplt, pose


    .. testcode:: [motion_interpolation_example2]

        # 3-pose interpolation

        from rational_linkages import DualQuaternion, Plotter, MotionInterpolation


        p0 = DualQuaternion([0, 17, -33, -89, 0, -6, 5, -3])
        p1 = DualQuaternion([0, 84, -21, -287, 0, -30, 3, -9])
        p2 = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3])

        c = MotionInterpolation.interpolate([p0, p1, p2])

        plt = Plotter(interactive=False, steps=500, arrows_length=0.05)
        plt.plot(c, interval='closed')

        for i, pose in enumerate([p0, p1, p2]):
            plt.plot(pose, label='p{}'.format(i+1))

        plt.show()

    .. testcleanup:: [motion_interpolation_example2]

        del DualQuaternion, Plotter, MotionInterpolation
        del p0, p1, p2, c, plt, pose

    """
    def __init__(self):
        """
        Creates a new instance of the rational motion interpolation class.
        """
        pass

    @staticmethod
    def interpolate(poses_or_points: list[Union[DualQuaternion, TransfMatrix, PointHomogeneous]],
                    lambda_val: Union[float, int] = 0,
                    motion_family: int = 0) -> RationalCurve:
        """
        Interpolates the given 2, 3, 4 poses or 5 points by a rational motion in SE(3).

        :param list[Union[DualQuaternion, TransfMatrix, PointHomogeneous]]
            poses_or_points: The poses or points to interpolate.
        :param Union[float, int] lambda_val: The lambda parameter for the interpolation.
            Only used for cubic interpolation using 4 poses.
        :param int motion_family: The family of the motion curve. 0 - default, 1 - other
            solution. Only used for cubic interpolation using 4 poses.

        :return: The rational motion curve.
        :rtype: RationalCurve
        """
        # check number of poses
        if not ((2 <= len(poses_or_points) <= 5) or len(poses_or_points) == 7):
            raise ValueError('Only 2-4 poses or 5 or 7 points can be interpolated.')

        p0_array = np.asarray(poses_or_points[0].array(), dtype='float64')

        # check if the first pose is the identity matrix
        if ((isinstance(poses_or_points[0], TransfMatrix)
            and not np.allclose(p0_array, TransfMatrix().matrix))
                or (isinstance(poses_or_points[0], DualQuaternion)
                    and not np.allclose(p0_array, DualQuaternion().dq))):

            if len(poses_or_points) == 4:
                raise ValueError('The first pose must be the identity matrix')
            elif len(poses_or_points) == 3:
                warn('The first pose IS NOT the identity. The interpolation '
                     'results may be unstable. They will yield non-univariate '
                     'polynomial which has to be transformed to visually '
                     'interpolate the curve.',
                     UserWarning)

        rational_poses = []

        # convert poses to rational dual quaternions
        for pose in poses_or_points:
            if isinstance(pose, TransfMatrix):
                rational_poses.append(DualQuaternion.as_rational(pose.matrix2dq()))
            elif isinstance(pose, DualQuaternion) and not pose.is_rational:
                rational_poses.append(DualQuaternion.as_rational(pose.array()))
            elif isinstance(pose, DualQuaternion) and pose.is_rational:
                rational_poses.append(pose)
            elif isinstance(pose, PointHomogeneous):
                rational_poses.append(pose)
            else:
                raise TypeError('The given poses must be either TransfMatrix,'
                                 ' DualQuaternion, or PointHomogeneous.')

        # normalize the DQ poses on Study quadric
        if len(rational_poses) != 5 and len(rational_poses) != 7:
            rational_poses = [pose.back_projection() for pose in rational_poses]

        # interpolate the rational poses
        if len(rational_poses) == 4:
            curve_eqs = MotionInterpolation.interpolate_cubic(rational_poses,
                                                              lambda_val=lambda_val,
                                                              motion_family=motion_family)
            return RationalCurve(curve_eqs)
        elif len(rational_poses) == 3:
            curve_eqs = MotionInterpolation.interpolate_quadratic(rational_poses)
            return RationalCurve(curve_eqs)
        elif len(rational_poses) == 2:
            curve_eqs = MotionInterpolation.interpolate_quadratic_2_poses(rational_poses)
            return RationalCurve(curve_eqs)
        elif len(rational_poses) == 5:
            curve_eqs = MotionInterpolation.interpolate_points_quadratic(rational_poses)
            return RationalCurve(curve_eqs)
        elif len(rational_poses) == 7:
            curve_eqs = MotionInterpolation.interpolate_points_cubic(rational_poses)
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

        c = alpha * p2 + (p1 - alpha * p2 - omega * p0) * t + omega * p0 * t**2
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
        c_interp = al * p2 + (p1 - al * p2 - om * p0) * t + om * p0 * t**2

        # list of polynomials
        poly = [sp.Poly(el, t) for el in c_interp]

        return poly

    @staticmethod
    def interpolate_quadratic_numerically(poses: list[DualQuaternion]) -> np.ndarray:
        """
        Interpolates the given 3 poses by a quadratic curve in SE(3).

        :param list[DualQuaternion] poses: the rational poses to interpolate.

        :return: numerical coefficients of the motion curve
        :rtype: np.ndarray
        """
        p0 = poses[0].array()
        p1 = poses[1].array()
        p2 = poses[2].array()

        # split dual quaternion into real (first 4) and dual (last 4) parts
        p0_r, p0_d = p0[:4], p0[4:]
        p1_r, p1_d = p1[:4], p1[4:]
        p2_r, p2_d = p2[:4], p2[4:]

        # compute the common denominator
        denom = np.dot(p2_r, p0_d) + np.dot(p0_r, p2_d)
        if np.abs(denom) < 1e-12:
            raise ValueError("Interpolation failed: denominator nearly zero. The poses p0 and p2 are dependent.")

        # parameters from the Study condition
        omega = (np.dot(p2_r, p1_d) + np.dot(p1_r, p2_d)) / denom
        alpha = (np.dot(p1_r, p0_d) + np.dot(p0_r, p1_d)) / denom

        c0 = alpha * p2
        c1 = p1 - alpha * p2 - omega * p0
        c2 = omega * p0

        # return array of coefficients
        return np.stack([c2, c1, c0], axis=1)


    @staticmethod
    def interpolate_quadratic_2_poses(poses: list[DualQuaternion]) -> list[sp.Poly]:
        """
        Interpolates the given 2 rational poses by a quadratic curve in SE(3).

        Adds the 3rd pose that is either identity or a random pose that returns
        solution.

        :param list[DualQuaternion] poses: The rational poses to interpolate.

        :return: Polynomials of rational motion curve.
        :rtype: list[sp.Poly]
        """
        try:
            return MotionInterpolation.interpolate_quadratic_2_poses_optimized(poses)
        except Exception:
            print('Failed to interpolate with a random pose optimized for shortest '
                  'path length. Trying to interpolate with other random poses...')
            return MotionInterpolation.interpolate_quadratic_2_poses_random(poses)

    @staticmethod
    def interpolate_quadratic_2_poses_random(poses: list[DualQuaternion]
                                             ) -> list[sp.Poly]:
        """
        Interpolates the given 2 rational poses by a quadratic curve in SE(3).

        Adds the 10 times 3rd pose that is random and returns the one with shortest
        path-length.

        :param list[DualQuaternion] poses: The rational poses to interpolate.

        :return: Polynomials of rational motion curve.
        :rtype: list[sp.Poly]
        """
        # Calculate the mid point between the two poses
        p0 = PointHomogeneous(poses[0].dq)
        p1 = PointHomogeneous(poses[1].dq)
        mid_p = p0.linear_interpolation(p1, 0.5)
        mid_pose = DualQuaternion(mid_p.array())

        # get mean value of mid_pose coordinates
        mean = sum(mid_pose.array()) / len(mid_pose.array())

        shortest_curve_length = float('inf')
        shortest_set = None
        best_pose = None

        for i in range(1, 10):
            additional_pose = DualQuaternion.as_rational(
                DualQuaternion.random_on_study_quadric(
                    mean * 0.3 * i).array()).back_projection()

            new_poses = deepcopy(poses)
            new_poses.append(additional_pose)

            try:
                polynomial_set = MotionInterpolation.interpolate_quadratic(
                    new_poses)
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

            return shortest_set

        else:  # if no solution was found
            raise ValueError('Interpolation failed for the given poses.')

    @staticmethod
    def interpolate_quadratic_2_poses_optimized(poses: list[DualQuaternion],
                                                max_iter: int = 0,
                                                ) -> list[sp.Poly]:
        """
        Interpolates the given 2 rational poses by a quadratic curve in SE(3).

        Adds the 3rd pose that is optimized for the shortest path-length.

        :param list[DualQuaternion] poses: The rational poses to interpolate
        :param int max_iter: The maximum number of iterations for the optimization,
            if 0, the optimization will run until the tolerance is reached.

        :return: Polynomials of rational motion curve.
        :rtype: list[sp.Poly]
        """
        from scipy.optimize import minimize  # inner import

        mid_pose = DualQuaternion.random_on_study_quadric()
        mid_pose_tr = TransfMatrix(mid_pose.dq2matrix())
        x0 = mid_pose_tr.t

        def objective_func(x):
            optim_pose = mid_pose_tr
            optim_pose.t = x

            new_poses = deepcopy(poses)
            new_poses.append(DualQuaternion.as_rational(
                                 optim_pose.matrix2dq()).back_projection())

            length = RationalCurve(
                MotionInterpolation.interpolate_quadratic(new_poses)).get_path_length(
                num_of_points=300
            )

            return length

        if max_iter == 0:
            res = minimize(objective_func, x0, tol=1e-3)
        else:
            res = minimize(objective_func, x0, tol=1e-3, options={'maxiter': max_iter})

        optimal_pose = mid_pose_tr
        optimal_pose.t = res.x
        optimal_pose_projected = DualQuaternion.as_rational(
            optimal_pose.matrix2dq()).back_projection()
        print('Optimal pose:')
        print(optimal_pose_projected)

        poses.append(optimal_pose_projected)

        return MotionInterpolation.interpolate_quadratic(poses)

    @staticmethod
    def interpolate_cubic(poses: list[DualQuaternion],
                          lambda_val: Union[float, int] = 0,
                          motion_family: int = 0) -> list[sp.Poly]:
        """
        Interpolates the given 4 rational poses by a cubic curve in SE(3).

        The 4 poses span a projective 3-space, which is intersected with Study quadric.
        This intersection gives another quadric containing all 4 poses, and it also
        contains cubic curves if it contains lines. The algorithm later searches
        for one of the cubic curves that interpolates the 4 poses.

        :see also: :ref:`interpolation_background`

        :param list[DualQuaternion] poses: The rational poses to interpolate.
        :param Union[float, int] lambda_val: The lambda parameter for the interpolation.
        :param int motion_family: The family of the motion curve. 0 - default, 1 - other
            solution


        :return: The rational motion curve.
        :rtype: list[sp.Poly]

        :raises ValueError: If the interpolation has no solution, 'k' does not exist.
        """
        # obtain additional dual quaternions k1, k2
        poses = deepcopy(poses)
        try:
            k = MotionInterpolation._obtain_k_dq(poses)
        except Exception:
            raise ValueError('Interpolation has no solution.')

        # solve for t[i] - the parameter of the rational motion curve for i-th pose
        t_sols = MotionInterpolation._solve_for_t(poses, k[motion_family])

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
        poses.append(DualQuaternion([lam, 0, 0, 0, 0, 0, 0, 0]) - k[motion_family])

        eqs_lambda = [element.subs(x, lam) - lams[-1] * poses[-1].array()[i]
                      for i, element in enumerate(temp4)]

        sols_lambda = sp.solve(eqs_lambda, lams, domain='RR')

        # obtain the family of solutions
        poly = [element.subs(sols_lambda) for element in temp4]

        # choose one solution by setting lambda, default lam=0
        poly = [element.subs(lam, lambda_val) for element in poly]

        t = sp.Symbol("t")
        poly = [element.subs(x, t) for element in poly]

        return [sp.Poly(element, t) for element in poly]

    @staticmethod
    def interpolate_cubic_numerically(poses: list[DualQuaternion],
                                      k_idx: int = 0,
                                      lambda_val: Union[float, int] = 0) -> np.ndarray:
        """
        Interpolates the given 4 rational poses by a cubic curve in SE(3).

        The 4 poses span a projective 3-space, which is intersected with Study quadric.
        This intersection gives another quadric containing all 4 poses, and it also
        contains cubic curves if it contains lines. The algorithm later searches
        for one of the cubic curves that interpolates the 4 poses.

        :see also: :ref:`interpolation_background`

        :param list[DualQuaternion] poses: The rational poses to interpolate.
        :param Union[float, int] lambda_val: The lambda parameter for the interpolation.

        :return: the numerical coefficients of the motion curve
        :rtype: np.ndarray

        :raises ValueError: If the interpolation has no solution, 'k' does not exist.
        """
        def lagrange_poly(x_indet, lambdas, t_vals, p1arr, p2arr, p3arr):
            lam1, lam2, lam3 = lambdas
            p10, p11, p12, p13, p14, p15, p16, p17 = p1arr
            p20, p21, p22, p23, p24, p25, p26, p27 = p2arr
            p30, p31, p32, p33, p34, p35, p36, p37 = p3arr
            t2, t3, t4 = t_vals
            x = x_indet
            return [
                lam1 * p10 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p10 * t2 ** 3 * t3 ** 2 * x - lam1 * p10 * t2 ** 3 * t3 * t4 ** 2 + lam1 * p10 * t2 ** 3 * t3 * x ** 2 + lam1 * p10 * t2 ** 3 * t4 ** 2 * x - lam1 * p10 * t2 ** 3 * t4 * x ** 2 - lam2 * p20 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p20 * t2 ** 2 * t3 ** 3 * x + lam2 * p20 * t2 * t3 ** 3 * t4 ** 2 - lam2 * p20 * t2 * t3 ** 3 * x ** 2 - lam2 * p20 * t3 ** 3 * t4 ** 2 * x + lam2 * p20 * t3 ** 3 * t4 * x ** 2 + lam3 * p30 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p30 * t2 ** 2 * t4 ** 3 * x - lam3 * p30 * t2 * t3 ** 2 * t4 ** 3 + lam3 * p30 * t2 * t4 ** 3 * x ** 2 + lam3 * p30 * t3 ** 2 * t4 ** 3 * x - lam3 * p30 * t3 * t4 ** 3 * x ** 2 - t2 ** 3 * t3 ** 2 * t4 + t2 ** 3 * t3 ** 2 * x + t2 ** 3 * t3 * t4 ** 2 - t2 ** 3 * t3 * x ** 2 - t2 ** 3 * t4 ** 2 * x + t2 ** 3 * t4 * x ** 2 + t2 ** 2 * t3 ** 3 * t4 - t2 ** 2 * t3 ** 3 * x - t2 ** 2 * t3 * t4 ** 3 + t2 ** 2 * t3 * x ** 3 + t2 ** 2 * t4 ** 3 * x - t2 ** 2 * t4 * x ** 3 - t2 * t3 ** 3 * t4 ** 2 + t2 * t3 ** 3 * x ** 2 + t2 * t3 ** 2 * t4 ** 3 - t2 * t3 ** 2 * x ** 3 - t2 * t4 ** 3 * x ** 2 + t2 * t4 ** 2 * x ** 3 + t3 ** 3 * t4 ** 2 * x - t3 ** 3 * t4 * x ** 2 - t3 ** 2 * t4 ** 3 * x + t3 ** 2 * t4 * x ** 3 + t3 * t4 ** 3 * x ** 2 - t3 * t4 ** 2 * x ** 3,
                lam1 * p11 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p11 * t2 ** 3 * t3 ** 2 * x - lam1 * p11 * t2 ** 3 * t3 * t4 ** 2 + lam1 * p11 * t2 ** 3 * t3 * x ** 2 + lam1 * p11 * t2 ** 3 * t4 ** 2 * x - lam1 * p11 * t2 ** 3 * t4 * x ** 2 - lam2 * p21 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p21 * t2 ** 2 * t3 ** 3 * x + lam2 * p21 * t2 * t3 ** 3 * t4 ** 2 - lam2 * p21 * t2 * t3 ** 3 * x ** 2 - lam2 * p21 * t3 ** 3 * t4 ** 2 * x + lam2 * p21 * t3 ** 3 * t4 * x ** 2 + lam3 * p31 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p31 * t2 ** 2 * t4 ** 3 * x - lam3 * p31 * t2 * t3 ** 2 * t4 ** 3 + lam3 * p31 * t2 * t4 ** 3 * x ** 2 + lam3 * p31 * t3 ** 2 * t4 ** 3 * x - lam3 * p31 * t3 * t4 ** 3 * x ** 2,
                lam1 * p12 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p12 * t2 ** 3 * t3 ** 2 * x - lam1 * p12 * t2 ** 3 * t3 * t4 ** 2 + lam1 * p12 * t2 ** 3 * t3 * x ** 2 + lam1 * p12 * t2 ** 3 * t4 ** 2 * x - lam1 * p12 * t2 ** 3 * t4 * x ** 2 - lam2 * p22 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p22 * t2 ** 2 * t3 ** 3 * x + lam2 * p22 * t2 * t3 ** 3 * t4 ** 2 - lam2 * p22 * t2 * t3 ** 3 * x ** 2 - lam2 * p22 * t3 ** 3 * t4 ** 2 * x + lam2 * p22 * t3 ** 3 * t4 * x ** 2 + lam3 * p32 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p32 * t2 ** 2 * t4 ** 3 * x - lam3 * p32 * t2 * t3 ** 2 * t4 ** 3 + lam3 * p32 * t2 * t4 ** 3 * x ** 2 + lam3 * p32 * t3 ** 2 * t4 ** 3 * x - lam3 * p32 * t3 * t4 ** 3 * x ** 2,
                lam1 * p13 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p13 * t2 ** 3 * t3 ** 2 * x - lam1 * p13 * t2 ** 3 * t3 * t4 ** 2 + lam1 * p13 * t2 ** 3 * t3 * x ** 2 + lam1 * p13 * t2 ** 3 * t4 ** 2 * x - lam1 * p13 * t2 ** 3 * t4 * x ** 2 - lam2 * p23 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p23 * t2 ** 2 * t3 ** 3 * x + lam2 * p23 * t2 * t3 ** 3 * t4 ** 2 - lam2 * p23 * t2 * t3 ** 3 * x ** 2 - lam2 * p23 * t3 ** 3 * t4 ** 2 * x + lam2 * p23 * t3 ** 3 * t4 * x ** 2 + lam3 * p33 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p33 * t2 ** 2 * t4 ** 3 * x - lam3 * p33 * t2 * t3 ** 2 * t4 ** 3 + lam3 * p33 * t2 * t4 ** 3 * x ** 2 + lam3 * p33 * t3 ** 2 * t4 ** 3 * x - lam3 * p33 * t3 * t4 ** 3 * x ** 2,
                lam1 * p14 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p14 * t2 ** 3 * t3 ** 2 * x - lam1 * p14 * t2 ** 3 * t3 * t4 ** 2 + lam1 * p14 * t2 ** 3 * t3 * x ** 2 + lam1 * p14 * t2 ** 3 * t4 ** 2 * x - lam1 * p14 * t2 ** 3 * t4 * x ** 2 - lam2 * p24 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p24 * t2 ** 2 * t3 ** 3 * x + lam2 * p24 * t2 * t3 ** 3 * t4 ** 2 - lam2 * p24 * t2 * t3 ** 3 * x ** 2 - lam2 * p24 * t3 ** 3 * t4 ** 2 * x + lam2 * p24 * t3 ** 3 * t4 * x ** 2 + lam3 * p34 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p34 * t2 ** 2 * t4 ** 3 * x - lam3 * p34 * t2 * t3 ** 2 * t4 ** 3 + lam3 * p34 * t2 * t4 ** 3 * x ** 2 + lam3 * p34 * t3 ** 2 * t4 ** 3 * x - lam3 * p34 * t3 * t4 ** 3 * x ** 2,
                lam1 * p15 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p15 * t2 ** 3 * t3 ** 2 * x - lam1 * p15 * t2 ** 3 * t3 * t4 ** 2 + lam1 * p15 * t2 ** 3 * t3 * x ** 2 + lam1 * p15 * t2 ** 3 * t4 ** 2 * x - lam1 * p15 * t2 ** 3 * t4 * x ** 2 - lam2 * p25 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p25 * t2 ** 2 * t3 ** 3 * x + lam2 * p25 * t2 * t3 ** 3 * t4 ** 2 - lam2 * p25 * t2 * t3 ** 3 * x ** 2 - lam2 * p25 * t3 ** 3 * t4 ** 2 * x + lam2 * p25 * t3 ** 3 * t4 * x ** 2 + lam3 * p35 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p35 * t2 ** 2 * t4 ** 3 * x - lam3 * p35 * t2 * t3 ** 2 * t4 ** 3 + lam3 * p35 * t2 * t4 ** 3 * x ** 2 + lam3 * p35 * t3 ** 2 * t4 ** 3 * x - lam3 * p35 * t3 * t4 ** 3 * x ** 2,
                lam1 * p16 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p16 * t2 ** 3 * t3 ** 2 * x - lam1 * p16 * t2 ** 3 * t3 * t4 ** 2 + lam1 * p16 * t2 ** 3 * t3 * x ** 2 + lam1 * p16 * t2 ** 3 * t4 ** 2 * x - lam1 * p16 * t2 ** 3 * t4 * x ** 2 - lam2 * p26 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p26 * t2 ** 2 * t3 ** 3 * x + lam2 * p26 * t2 * t3 ** 3 * t4 ** 2 - lam2 * p26 * t2 * t3 ** 3 * x ** 2 - lam2 * p26 * t3 ** 3 * t4 ** 2 * x + lam2 * p26 * t3 ** 3 * t4 * x ** 2 + lam3 * p36 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p36 * t2 ** 2 * t4 ** 3 * x - lam3 * p36 * t2 * t3 ** 2 * t4 ** 3 + lam3 * p36 * t2 * t4 ** 3 * x ** 2 + lam3 * p36 * t3 ** 2 * t4 ** 3 * x - lam3 * p36 * t3 * t4 ** 3 * x ** 2,
                lam1 * p17 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p17 * t2 ** 3 * t3 ** 2 * x - lam1 * p17 * t2 ** 3 * t3 * t4 ** 2 + lam1 * p17 * t2 ** 3 * t3 * x ** 2 + lam1 * p17 * t2 ** 3 * t4 ** 2 * x - lam1 * p17 * t2 ** 3 * t4 * x ** 2 - lam2 * p27 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p27 * t2 ** 2 * t3 ** 3 * x + lam2 * p27 * t2 * t3 ** 3 * t4 ** 2 - lam2 * p27 * t2 * t3 ** 3 * x ** 2 - lam2 * p27 * t3 ** 3 * t4 ** 2 * x + lam2 * p27 * t3 ** 3 * t4 * x ** 2 + lam3 * p37 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p37 * t2 ** 2 * t4 ** 3 * x - lam3 * p37 * t2 * t3 ** 2 * t4 ** 3 + lam3 * p37 * t2 * t4 ** 3 * x ** 2 + lam3 * p37 * t3 ** 2 * t4 ** 3 * x - lam3 * p37 * t3 * t4 ** 3 * x ** 2
            ]

        def lagrange(lam, k_dq, t_vals, lambdas, p1arr, p2arr, p3arr):
            k0, k1, k2, k3, k4, k5, k6, k7 = k_dq
            lam1, lam2, lam3, lam4 = lambdas
            p10, p11, p12, p13, p14, p15, p16, p17 = p1arr
            p20, p21, p22, p23, p24, p25, p26, p27 = p2arr
            p30, p31, p32, p33, p34, p35, p36, p37 = p3arr
            t2, t3, t4 = t_vals
            return [
                -lam2 * p20 * lam * t4 ** 2 * t3 ** 3 + lam2 * p20 * lam * t2 ** 2 * t3 ** 3 + lam2 * p20 * lam ** 2 * t4 * t3 ** 3 - lam2 * p20 * lam ** 2 * t2 * t3 ** 3 + lam3 * p30 * lam * t4 ** 3 * t3 ** 2 - lam3 * p30 * lam * t2 ** 2 * t4 ** 3 - lam3 * p30 * lam ** 2 * t4 ** 3 * t3 + lam3 * p30 * lam ** 2 * t2 * t4 ** 3 + lam1 * p10 * lam * t2 ** 3 * t4 ** 2 - lam1 * p10 * lam * t2 ** 3 * t3 ** 2 - lam1 * p10 * lam ** 2 * t2 ** 3 * t4 + lam1 * p10 * lam ** 2 * t2 ** 3 * t3 - lam1 * p10 * t2 ** 3 * t4 ** 2 * t3 + lam1 * p10 * t2 ** 3 * t4 * t3 ** 2 + lam2 * p20 * t2 * t4 ** 2 * t3 ** 3 - lam2 * p20 * t2 ** 2 * t4 * t3 ** 3 - lam3 * p30 * t2 * t4 ** 3 * t3 ** 2 + lam3 * p30 * t2 ** 2 * t4 ** 3 * t3 + t2 * t3 ** 2 * t4 ** 3 - t2 ** 2 * t3 * t4 ** 3 + t2 ** 2 * t3 ** 3 * t4 - t2 * t3 ** 3 * t4 ** 2 - t2 ** 3 * t3 ** 2 * t4 + t2 ** 3 * t3 * t4 ** 2 + lam ** 2 * t4 ** 3 * t3 + lam * t2 ** 2 * t4 ** 3 - lam * t4 ** 3 * t3 ** 2 + lam ** 2 * t2 ** 3 * t4 + lam * t4 ** 2 * t3 ** 3 + lam * t2 ** 3 * t3 ** 2 - lam * t2 ** 2 * t3 ** 3 - lam ** 2 * t4 * t3 ** 3 - lam ** 2 * t2 * t4 ** 3 - lam ** 2 * t2 ** 3 * t3 + lam ** 3 * t2 ** 2 * t3 - lam ** 3 * t2 * t3 ** 2 + lam ** 3 * t2 * t4 ** 2 - lam ** 3 * t2 ** 2 * t4 - lam ** 3 * t4 ** 2 * t3 + lam ** 3 * t4 * t3 ** 2 - lam * t2 ** 3 * t4 ** 2 + lam ** 2 * t2 * t3 ** 3 - lam4 * (-k0 + lam),
                lam ** 2 * lam1 * p11 * t2 ** 3 * t3 - lam ** 2 * lam1 * p11 * t2 ** 3 * t4 - lam ** 2 * lam2 * p21 * t2 * t3 ** 3 + lam ** 2 * lam2 * p21 * t3 ** 3 * t4 + lam ** 2 * lam3 * p31 * t2 * t4 ** 3 - lam ** 2 * lam3 * p31 * t3 * t4 ** 3 - lam * lam1 * p11 * t2 ** 3 * t3 ** 2 + lam * lam1 * p11 * t2 ** 3 * t4 ** 2 + lam * lam2 * p21 * t2 ** 2 * t3 ** 3 - lam * lam2 * p21 * t3 ** 3 * t4 ** 2 - lam * lam3 * p31 * t2 ** 2 * t4 ** 3 + lam * lam3 * p31 * t3 ** 2 * t4 ** 3 + lam1 * p11 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p11 * t2 ** 3 * t3 * t4 ** 2 - lam2 * p21 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p21 * t2 * t3 ** 3 * t4 ** 2 + lam3 * p31 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p31 * t2 * t3 ** 2 * t4 ** 3 + k1 * lam4,
                lam ** 2 * lam1 * p12 * t2 ** 3 * t3 - lam ** 2 * lam1 * p12 * t2 ** 3 * t4 - lam ** 2 * lam2 * p22 * t2 * t3 ** 3 + lam ** 2 * lam2 * p22 * t3 ** 3 * t4 + lam ** 2 * lam3 * p32 * t2 * t4 ** 3 - lam ** 2 * lam3 * p32 * t3 * t4 ** 3 - lam * lam1 * p12 * t2 ** 3 * t3 ** 2 + lam * lam1 * p12 * t2 ** 3 * t4 ** 2 + lam * lam2 * p22 * t2 ** 2 * t3 ** 3 - lam * lam2 * p22 * t3 ** 3 * t4 ** 2 - lam * lam3 * p32 * t2 ** 2 * t4 ** 3 + lam * lam3 * p32 * t3 ** 2 * t4 ** 3 + lam1 * p12 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p12 * t2 ** 3 * t3 * t4 ** 2 - lam2 * p22 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p22 * t2 * t3 ** 3 * t4 ** 2 + lam3 * p32 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p32 * t2 * t3 ** 2 * t4 ** 3 + k2 * lam4,
                lam ** 2 * lam1 * p13 * t2 ** 3 * t3 - lam ** 2 * lam1 * p13 * t2 ** 3 * t4 - lam ** 2 * lam2 * p23 * t2 * t3 ** 3 + lam ** 2 * lam2 * p23 * t3 ** 3 * t4 + lam ** 2 * lam3 * p33 * t2 * t4 ** 3 - lam ** 2 * lam3 * p33 * t3 * t4 ** 3 - lam * lam1 * p13 * t2 ** 3 * t3 ** 2 + lam * lam1 * p13 * t2 ** 3 * t4 ** 2 + lam * lam2 * p23 * t2 ** 2 * t3 ** 3 - lam * lam2 * p23 * t3 ** 3 * t4 ** 2 - lam * lam3 * p33 * t2 ** 2 * t4 ** 3 + lam * lam3 * p33 * t3 ** 2 * t4 ** 3 + lam1 * p13 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p13 * t2 ** 3 * t3 * t4 ** 2 - lam2 * p23 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p23 * t2 * t3 ** 3 * t4 ** 2 + lam3 * p33 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p33 * t2 * t3 ** 2 * t4 ** 3 + k3 * lam4,
                lam ** 2 * lam1 * p14 * t2 ** 3 * t3 - lam ** 2 * lam1 * p14 * t2 ** 3 * t4 - lam ** 2 * lam2 * p24 * t2 * t3 ** 3 + lam ** 2 * lam2 * p24 * t3 ** 3 * t4 + lam ** 2 * lam3 * p34 * t2 * t4 ** 3 - lam ** 2 * lam3 * p34 * t3 * t4 ** 3 - lam * lam1 * p14 * t2 ** 3 * t3 ** 2 + lam * lam1 * p14 * t2 ** 3 * t4 ** 2 + lam * lam2 * p24 * t2 ** 2 * t3 ** 3 - lam * lam2 * p24 * t3 ** 3 * t4 ** 2 - lam * lam3 * p34 * t2 ** 2 * t4 ** 3 + lam * lam3 * p34 * t3 ** 2 * t4 ** 3 + lam1 * p14 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p14 * t2 ** 3 * t3 * t4 ** 2 - lam2 * p24 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p24 * t2 * t3 ** 3 * t4 ** 2 + lam3 * p34 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p34 * t2 * t3 ** 2 * t4 ** 3 + k4 * lam4,
                lam ** 2 * lam1 * p15 * t2 ** 3 * t3 - lam ** 2 * lam1 * p15 * t2 ** 3 * t4 - lam ** 2 * lam2 * p25 * t2 * t3 ** 3 + lam ** 2 * lam2 * p25 * t3 ** 3 * t4 + lam ** 2 * lam3 * p35 * t2 * t4 ** 3 - lam ** 2 * lam3 * p35 * t3 * t4 ** 3 - lam * lam1 * p15 * t2 ** 3 * t3 ** 2 + lam * lam1 * p15 * t2 ** 3 * t4 ** 2 + lam * lam2 * p25 * t2 ** 2 * t3 ** 3 - lam * lam2 * p25 * t3 ** 3 * t4 ** 2 - lam * lam3 * p35 * t2 ** 2 * t4 ** 3 + lam * lam3 * p35 * t3 ** 2 * t4 ** 3 + lam1 * p15 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p15 * t2 ** 3 * t3 * t4 ** 2 - lam2 * p25 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p25 * t2 * t3 ** 3 * t4 ** 2 + lam3 * p35 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p35 * t2 * t3 ** 2 * t4 ** 3 + k5 * lam4,
                lam ** 2 * lam1 * p16 * t2 ** 3 * t3 - lam ** 2 * lam1 * p16 * t2 ** 3 * t4 - lam ** 2 * lam2 * p26 * t2 * t3 ** 3 + lam ** 2 * lam2 * p26 * t3 ** 3 * t4 + lam ** 2 * lam3 * p36 * t2 * t4 ** 3 - lam ** 2 * lam3 * p36 * t3 * t4 ** 3 - lam * lam1 * p16 * t2 ** 3 * t3 ** 2 + lam * lam1 * p16 * t2 ** 3 * t4 ** 2 + lam * lam2 * p26 * t2 ** 2 * t3 ** 3 - lam * lam2 * p26 * t3 ** 3 * t4 ** 2 - lam * lam3 * p36 * t2 ** 2 * t4 ** 3 + lam * lam3 * p36 * t3 ** 2 * t4 ** 3 + lam1 * p16 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p16 * t2 ** 3 * t3 * t4 ** 2 - lam2 * p26 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p26 * t2 * t3 ** 3 * t4 ** 2 + lam3 * p36 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p36 * t2 * t3 ** 2 * t4 ** 3 + k6 * lam4,
                lam ** 2 * lam1 * p17 * t2 ** 3 * t3 - lam ** 2 * lam1 * p17 * t2 ** 3 * t4 - lam ** 2 * lam2 * p27 * t2 * t3 ** 3 + lam ** 2 * lam2 * p27 * t3 ** 3 * t4 + lam ** 2 * lam3 * p37 * t2 * t4 ** 3 - lam ** 2 * lam3 * p37 * t3 * t4 ** 3 - lam * lam1 * p17 * t2 ** 3 * t3 ** 2 + lam * lam1 * p17 * t2 ** 3 * t4 ** 2 + lam * lam2 * p27 * t2 ** 2 * t3 ** 3 - lam * lam2 * p27 * t3 ** 3 * t4 ** 2 - lam * lam3 * p37 * t2 ** 2 * t4 ** 3 + lam * lam3 * p37 * t3 ** 2 * t4 ** 3 + lam1 * p17 * t2 ** 3 * t3 ** 2 * t4 - lam1 * p17 * t2 ** 3 * t3 * t4 ** 2 - lam2 * p27 * t2 ** 2 * t3 ** 3 * t4 + lam2 * p27 * t2 * t3 ** 3 * t4 ** 2 + lam3 * p37 * t2 ** 2 * t3 * t4 ** 3 - lam3 * p37 * t2 * t3 ** 2 * t4 ** 3 + k7 * lam4
            ]

        poses = deepcopy(poses)
        lambda_val = 1e-16 if abs(lambda_val) < 1e-16 else lambda_val

        p0 = poses[0].array()
        p1 = poses[1].array()
        p2 = poses[2].array()
        p3 = poses[3].array()

        try:
            x3_sols = motion_interp_x3(p1, p2, p3)
            k = []
            for x3 in x3_sols:
                x2 = -(p1[0] * p3[4] * x3 - p1[4] * p3[0] * x3 - p1[4]) / (
                            p1[0] * p2[4] - p1[4] * p2[0])
                x1 = (p2[0] * p3[4] * x3 - p2[4] * p3[0] * x3 - p2[4]) / (
                            p1[0] * p2[4] - p1[4] * p2[0])
                k_array = p0 + x1 * p1 + x2 * p2 + x3 * p3
                k.append(DualQuaternion(k_array))
        except Exception:
            raise ValueError('Interpolation failed, change input poses, as they do not '
                             'given poses do not fulfill the interpolation conditions.')

        # # solve for t[i] - the parameter of the rational motion curve for i-th pose
        # t_sols = MotionInterpolation._solve_for_t_numerically(poses, k)
        study_cond_mat = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0]])
        # Calculate identity DQ product with study matrix once
        k0_array = k[k_idx].array()
        study_cond = [study_cond_mat @ poses[i].array() for i in range(1, 4)]
        # Pre-compute results for the divisors
        denominators = np.array([p0 @ study_cond[i]
                                 for i in range(0, 3)])
        # Pre-compute results for the numerators
        numerators = np.array([k0_array @ study_cond[i]
                               for i in range(0, 3)])
        # Perform division for all three parameters at once
        safe_denominators = np.where(denominators == 0, 1e-16, denominators)
        t_sols = numerators / safe_denominators

        # Lagrange's interpolation part
        # lambdas for interpolation - scalar multiples of the poses
        lams = sp.symbols("lams1:5")
        x = sp.symbols("x")


        parametric_points = [np.array(poses[0].array()),
                             np.array(lams[0] * poses[1].array()),
                             np.array(lams[1] * poses[2].array()),
                             np.array(lams[2] * poses[3].array())]
        # obtain the Lagrange interpolation for poses p0, p1, p2, p3
        interp = MotionInterpolation.lagrange_interpolation_numerically(
            parametric_points,
            1 / x,
            [0.0, 1/t_sols[0], 1/t_sols[1], 1/t_sols[2]])
        temp_evaluated = [el * x ** 3 for el in interp]

        # obtain additional parametric pose p4
        poses.append(DualQuaternion([lambda_val, 0, 0, 0, 0, 0, 0, 0]) - k[k_idx])
        eqs_lambda = [element.subs(x, lambda_val) - lams[-1] * poses[-1].array()[i]
                      for i, element in enumerate(temp_evaluated)]

        # eqs_lambda = lagrange(lambda_val, k0_array, t_sols, lams, p1, p2, p3)
        # temp_evaluated = lagrange_poly(x, lams[0:3], t_sols, p1, p2, p3)

        sols_lambda = sp.nsolve(eqs_lambda, lams, [1., 1., -1., -1.], dict=True)

        coeffs = [[expr.coeff(x, i) for i in range(3, -1, -1)] for expr in
                  (sp.expand(element.subs(sols_lambda[0])) for element in temp_evaluated)]

        return np.array(coeffs, dtype='float64')


    @staticmethod
    def _obtain_k_dq(poses: list[DualQuaternion]) -> list[DualQuaternion]:
        """
        Obtain additional dual quaternions k1, k2 for interpolation of 4 poses.

        :param list[DualQuaternion] poses: The rational poses to interpolate.

        :return: Two additional dual quaternions for interpolation.
        :rtype: list[DualQuaternion]
        """
        x = sp.symbols("xx:3")

        k = DualQuaternion(poses[0].array() + x[0] * poses[1].array()
                           + x[1] * poses[2].array() + x[2] * poses[3].array())

        eqs = [k[0], k[4], k.norm().array()[4]]

        sol = sp.solve(eqs, x, domain='RR')

        k_as_expr = [sp.Expr(el) for el in k]

        k1 = [el.subs({x[0]: sol[0][0], x[1]: sol[0][1], x[2]: sol[0][2]})
              for el in k_as_expr]
        k2 = [el.subs({x[0]: sol[1][0], x[1]: sol[1][1], x[2]: sol[1][2]})
              for el in k_as_expr]

        k1_dq = DualQuaternion([el.args[0] for el in k1])
        k2_dq = DualQuaternion([el.args[0] for el in k2])

        return [k1_dq, k2_dq]

    @staticmethod
    def _solve_for_t(poses: list[DualQuaternion],
                     k: DualQuaternion) -> list:
        """
        Solve for t[i] - the parameter of the rational motion curve for i-th pose.

        :param list[DualQuaternion] poses: The rational poses to interpolate.
        :param list[DualQuaternion] k: The additional dual quaternions for interpolation.

        :return: The solutions for t[i].
        :rtype: list
        """
        t = sp.symbols("t:3")

        study_cond_mat = sp.Matrix([[0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0]])

        t_dq = [DualQuaternion([t[i], 0, 0, 0, 0, 0, 0, 0]) for i in range(3)]

        eqs = [sp.Matrix((t_dq[0] - k).array()).transpose() @ study_cond_mat
               @ sp.Matrix(poses[1].array()),
               sp.Matrix((t_dq[1] - k).array()).transpose() @ study_cond_mat
               @ sp.Matrix(poses[2].array()),
               sp.Matrix((t_dq[2] - k).array()).transpose() @ study_cond_mat
               @ sp.Matrix(poses[3].array())]

        sols_t = sp.solve(eqs, t)
        # covert to list and return
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

    @staticmethod
    def lagrange_interpolation_numerically(poses, x, t):
        """
        Perform Lagrange interpolation for 8-dimensional poses.
        """
        def lagrange_polynomial(i, x, t):
            p = 1.0
            for j in range(len(t)):
                if j != i:
                    p *= (x - t[j]) / (t[i] - t[j])
            return p

        poses = np.asarray(poses)
        n = len(t)
        # Ensure that the number of poses matches the number of nodes
        if poses.shape[0] != n:
            raise ValueError(
                "The number of poses must equal the number of interpolation nodes.")

        result = np.zeros(8)
        for i in range(n):
            li = lagrange_polynomial(i, x, t)
            result = result + poses[i] * li
        return result

    @staticmethod
    def interpolate_points_quadratic(points: list[PointHomogeneous],
                                     return_numeric: bool = False) -> (
            Union[list[sp.Poly], np.ndarray]):
        """
        Interpolates the given 5 points by a quadratic curve in SE(3).

        :param list[PointHomogeneous] points: The points to interpolate.
        :param bool return_numeric: If True, the result will be returned as numpy
            polynomials, otherwise as sympy polynomials.

        :return: The rational motion curve.
        :rtype: Union[list[sp.Poly], np.ndarray]
        """
        if not all(isinstance(p, PointHomogeneous) for p in points):
            raise TypeError('The given points must be PointHomogeneous.')

        if len(points) != 5:
            raise ValueError('The number of points must be 5.')

        # check if the points are Sympy Rational
        perform_rational = True if all(p.is_rational for p in points) else False

        points = [p if p[0] == 1 else PointHomogeneous(p.normalize()) for p in points]

        # map to Quaternions, divide by -2 (Study mapping from 3D)
        # and add 0 to the real part
        a0, a1, a2, a3, a4 = [Quaternion([0,
                                          p.array()[1] / -2,
                                          p.array()[2] / -2,
                                          p.array()[3] / -2]) for p in points]

        d41 = a4 - a1
        d21 = a2 - a1
        d43 = a4 - a3
        d23 = a2 - a3
        d10 = a1 - a0
        d30 = a3 - a0

        d32 = a3 - a2
        d14 = a1 - a4

        if np.allclose(float((d43.inv() * d32 * d21.inv() * d14)[0]), -3.0):
            raise ValueError("Not possible to interpolate")

        w0 = Quaternion()
        w2 = (-9 * d41.inv() * d21 - 3 * d43.inv() * d23).inv() * (
                    9 * d41.inv() * d10 - d43.inv() * d30) * w0
        w4 = (-1 * d21.inv() * d41 - 3 * d23.inv() * d43).inv() * (
                    3 * d21.inv() * d10 + d23.inv() * d30) * w0

        w_mid = -1 * (w0 + 2 * w2 + w4) / 2
        a_mid = -1 * (a0 * w0 + 2 * a2 * w2 + a4 * w4) * w_mid.inv() / 2

        # get the control points of Bezier curve from constructed dual quaternions
        cp0 = PointHomogeneous(np.concatenate((w0.array(), (a0 * w0).array())),
                               rational=perform_rational)
        cp1 = PointHomogeneous(np.concatenate((w_mid.array(), (a_mid * w_mid).array())),
                               rational=perform_rational)
        cp2 = PointHomogeneous(np.concatenate((w4.array(), (a4 * w4).array())),
                               rational=perform_rational)

        if return_numeric:
            return RationalBezier.get_numerical_coeffs([cp0, cp1, cp2])
        else:
            return RationalBezier([cp0, cp1, cp2]).set_of_polynomials

    @staticmethod
    def interpolate_points_cubic(points: list[PointHomogeneous],
                                 return_numeric: bool = False) -> (
            Union[list[sp.Poly], np.ndarray]):
        """
        Interpolates the given 7 points by a cubic curve in SE(3).

        :param list[Union[sp.Poly, np.polynomial.Polynomial]] points: The points
            to interpolate.
        :param bool return_numeric: If True, the result will be returned as numpy
            polynomials, otherwise as sympy polynomials.

        :return: The rational motion curve.
        :rtype: list[Union[sp.Poly, np.polynomial.Polynomial]]
        """
        if not all(isinstance(p, PointHomogeneous) for p in points):
            raise TypeError('The given points must be PointHomogeneous.')

        if len(points) != 7:
            raise ValueError('The number of points must be 7.')

        points = [p if p[0] == 1 else PointHomogeneous(p.normalize()) for p in points]

        # Check if the points are Sympy Rational
        perform_rational = True if all(p.is_rational for p in points) else False

        # map to Quaternions, divide by -2 (Study mapping from 3D)
        # and add 0 to the real part
        a0, a1, a2, a3, a4, a5, a6  = [Quaternion([0,
                                                   p.array()[1] / -2,
                                                   p.array()[2] / -2,
                                                   p.array()[3] / -2]) for p in points]

        def q_prod(q0, q1, q2, q3):
            return q0.inv() * q1 - q2.inv() * q3

        f12 = 15
        f14 = 5
        f16 = 3
        f18 = -15
        f22 = 9
        f24 = -9
        f26 = -3
        f28 = 3
        f32 = -5
        f34 = -15
        f36 = 15
        f38 = -3

        c12 = f12 * (a2 - a1)
        c14 = f14 * (a4 - a1)
        c16 = f16 * (a6 - a1)
        c22 = f22 * (a2 - a3)
        c24 = f24 * (a4 - a3)
        c26 = f26 * (a6 - a3)
        c32 = f32 * (a2 - a5)
        c34 = f34 * (a4 - a5)
        c36 = f36 * (a6 - a5)

        c18 = f18 * (a1 - a0)
        c28 = f28 * (a3 - a0)
        c38 = f38 * (a5 - a0)

        e24 = q_prod(c12, c14, c22, c24)
        e26 = q_prod(c12, c16, c22, c26)
        e28 = q_prod(c12, c18, c22, c28)
        e34 = q_prod(c12, c14, c32, c34)
        e36 = q_prod(c12, c16, c32, c36)
        e38 = q_prod(c12, c18, c32, c38)

        r24 = q_prod(e26, e24, e36, e34)
        r28 = q_prod(e26, e28, e36, e38)
        r36 = q_prod(e24, e26, e34, e36)
        r38 = q_prod(e24, e28, e34, e38)

        w0 = Quaternion()
        if perform_rational:
            w0 = Quaternion([sp.Rational(coord) for coord in w0.array()])

        w4 = r24.inv() * r28
        w6 = r36.inv() * r38
        w2 = c12.inv() * (c18 - c14 * w4 - c16 * w6)

        w_c0 = -2 * w0 / 9
        a_c0 = a0
        w_c1 = 5 * w0 / 27 + 2 * w2 / 9 + w4 / 9 + 2 * w6 / 27
        a_c1 = ((5 * a0 * w0 /27 + 2 * a2 * w2 / 9 + a4 * w4 / 9 + 2 * a6 * w6 / 27)
                * w_c1.inv())
        w_c2 = -2 * w0 / 27 - 1 * w2 / 9 - 2 * w4 / 9 - 5 * w6 / 27
        a_c2 = ((-2 * a0 * w0 / 27 -1 * a2 * w2 / 9 -2 * a4 * w4 / 9 -5 * a6 * w6 / 27)
                * w_c2.inv())
        w_c3 = 2 * w6 / 9
        a_c3 = a6

        # get the control points of Bezier curve from constructed dual quaternions
        cp0 = PointHomogeneous(np.concatenate((w_c0.array(), (a_c0 * w_c0).array())),
                               rational=perform_rational)
        cp1 = PointHomogeneous(np.concatenate((w_c1.array(), (a_c1 * w_c1).array())),
                               rational=perform_rational)
        cp2 = PointHomogeneous(np.concatenate((w_c2.array(), (a_c2 * w_c2).array())),
                               rational=perform_rational)
        cp3 = PointHomogeneous(np.concatenate((w_c3.array(), (a_c3 * w_c3).array())),
                               rational=perform_rational)

        if return_numeric:
            return RationalBezier.get_numerical_coeffs([cp0, cp1, cp2, cp3])
        else:
            return RationalBezier([cp0, cp1, cp2, cp3]).set_of_polynomials
