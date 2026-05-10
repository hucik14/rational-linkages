from typing import Union

import numpy

from .AffineMetric import AffineMetric
from .DualQuaternion import DualQuaternion
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve

try:
    from scipy.optimize import minimize  # lazy import, optional dependency
except ImportError:
    minimize = None


class MotionApproximation:
    """Utilities to approximate rational motion curves from samples.

    The class provides routines to approximate a low-degree rational motion
    curve that passes close to a set of given poses or points. Optimization
    is performed using SciPy when available.
    """
    def __init__(self):
        pass

    @staticmethod
    def approximate(init_curve,
                    poses_or_points: list[Union[DualQuaternion, PointHomogeneous]],
                    t_vals: Union[list[float], numpy.ndarray]
                    ) -> tuple[RationalCurve, dict]:
        """Approximate a motion curve that passes through given poses or points.

        Parameters
        ----------
        init_curve
            Initial :class:`.RationalCurve` guess. A good starting guess can be
            produced with the motion interpolation utilities.
        poses_or_points
            Sequence of target poses (:class:`.DualQuaternion`) or target points
            (:class:`.PointHomogeneous`) to approximate.
        t_vals
            Parameter values corresponding to the targets, in the same order.

        Returns
        -------
        tuple
            ``(approximated_curve, optimization_result)`` where
            ``approximated_curve`` is a :class:`.RationalCurve` and
            ``optimization_result`` is the solver's result object.
        """
        if init_curve.degree != 3:
            raise ValueError("So far, only cubic curves are supported")

        t_array = numpy.asarray(t_vals)

        if isinstance(poses_or_points[0], DualQuaternion):
            approx_curve, opt_result \
                = MotionApproximation._cubic_approximation(init_curve,
                                                           poses_or_points,
                                                           t_array)
        elif isinstance(poses_or_points[0], PointHomogeneous):
            approx_curve, opt_result \
                = MotionApproximation._cubic_approximation_for_points(init_curve,
                                                                      poses_or_points,
                                                                      t_array)
        else:
            raise TypeError("poses must be a list of DualQuaternion or PointHomogeneous objects")

        return approx_curve, opt_result

    @staticmethod
    def _construct_curve(flattended_coeffs) -> RationalCurve:
        """Build a monic cubic :class:`.RationalCurve` from flattened coefficients.

        Parameters
        ----------
        flattended_coeffs
            24-element flattened array containing the non-monic part of the
            cubic coefficients (8 x 3 entries).

        Returns
        -------
        RationalCurve
            Constructed rational cubic curve.
        """
        coeffs = numpy.zeros((8, 4))  # Preallocate an array of shape (8, 4)
        coeffs[0, 0] = 1
        coeffs[:, 1:] = flattended_coeffs.reshape(8, 3)

        return RationalCurve.from_coeffs(coeffs)

    @staticmethod
    def _construct_curve_nonmonic(flattended_coeffs) -> RationalCurve:
        """Build a non-monic cubic :class:`.RationalCurve` from flattened coeffs.

        Parameters
        ----------
        flattended_coeffs
            Flattened array with 32 coefficients (8 x 4) for a general cubic.

        Returns
        -------
        RationalCurve
            Constructed rational cubic curve.
        """
        return RationalCurve.from_coeffs(flattended_coeffs.reshape(8, 4))

    @staticmethod
    def _cubic_approximation(init_curve,
                             poses,
                             t_vals) -> tuple[RationalCurve, dict]:
        """Perform cubic motion approximation for pose targets.

        Parameters
        ----------
        init_curve
            Initial :class:`.RationalCurve` guess used to parameterize the
            optimization variables.
        poses
            Sequence of :class:`.DualQuaternion` target poses the curve should
            approximate.
        t_vals
            Array of parameter values for the poses.

        Returns
        -------
        tuple
            ``(result_curve, result)`` where ``result_curve`` is the optimized
            :class:`.RationalCurve` and ``result`` is the optimizer result.
        """
        metric = AffineMetric(init_curve,
                              [PointHomogeneous.from_3d_point(pose.dq2point_via_matrix())
                               for pose in poses])

        num_added_poses = len(poses) - 4

        initial_guess = init_curve.coeffs[:,1:4].flatten()
        initial_guess = numpy.concatenate((initial_guess, t_vals[-num_added_poses:]), axis=None)

        def objective_function(params):
            """
            Objective function to minimize the sum of squared distances between
            the poses and the curve
            """
            curve = MotionApproximation._construct_curve(params[:24])

            for i in range(num_added_poses):
                val = i + 1
                t_vals[-val] = params[24:][i]

            sq_dist = 0.
            for i, pose in enumerate(poses):
                curve_pose = DualQuaternion(curve.evaluate(t_vals[i]))
                sq_dist += metric.squared_distance(pose, curve_pose)

            return sq_dist

        def constraint_func(params):
            curve = MotionApproximation._construct_curve(params[:24])
            sq_err = curve.study_quadric_check()

            if len(sq_err) != 8:  # expand if necessary to avoid index errors
                sq_err = numpy.concatenate((sq_err, numpy.zeros(8 - len(sq_err))), axis=None)

            return sq_err

        def callback(params):
            current_distance = objective_function(params)
            current_constraint = constraint_func(params)
            print(f"Objective function: {current_distance}, Constraints:")
            print(current_constraint)

        constraints = []
        for i in range(6):  # separate constraint functions for Study Quadric equation
            constraints.append({
                'type': 'eq',
                'fun': (lambda params, index=i: constraint_func(params)[index])
            })

        result = minimize(objective_function,
                          initial_guess,
                          constraints=constraints,
                          callback=callback,
                          options={'maxiter': 50,
                                   'ftol': 1e-16,
                                   },
                          )

        print(result)
        result_curve = MotionApproximation._construct_curve(result.x[:24])

        return result_curve, result

    @staticmethod
    def _cubic_approximation_for_points(init_curve,
                                        points,
                                        t_vals) -> tuple[RationalCurve, dict]:
        """Perform cubic approximation for 3D point targets.

        Parameters
        ----------
        init_curve
            Initial :class:`.RationalCurve` guess.
        points
            Sequence of :class:`.PointHomogeneous` target points.
        t_vals
            Array of parameter values for the points.

        Returns
        -------
        tuple
            ``(result_curve, result)`` where ``result_curve`` is the
            optimized :class:`.RationalCurve` and ``result`` is the optimizer result.
        """
        t_vals_init = numpy.array([0, 1/6, 1/3, 1/2, 2/3, 5/6, 1])
        t_vals = numpy.concatenate((t_vals_init, t_vals), axis=None)

        num_added_points = len(points) - 7

        initial_guess = init_curve.coeffs.flatten()
        initial_guess = numpy.concatenate((initial_guess, t_vals[-num_added_points:]), axis=None)

        def objective_function(params):
            """
            Objective function to minimize the sum of squared distances between
            the poses and the curve
            """
            curve = MotionApproximation._construct_curve_nonmonic(params[:32])

            for i in range(num_added_points):
                val = i + 1
                t_vals[-val] = params[32:][i]

            sq_dist = 0.
            for i, pt in enumerate(points):
                # Get the 3D point from the curve
                curve_pt = DualQuaternion(
                    curve.evaluate(t_vals[i])).dq2point_via_matrix()
                target_pt = pt.normalized_euclidean()

                sq_dist += numpy.linalg.norm(curve_pt - target_pt) ** 2

            return sq_dist

        def constraint_func(params):
            curve = MotionApproximation._construct_curve_nonmonic(params[:32])
            sq_err = curve.study_quadric_check()

            if len(sq_err) != 8:  # expand if necessary to avoid index errors
                sq_err = numpy.concatenate((sq_err, numpy.zeros(8 - len(sq_err))), axis=None)

            return sq_err

        def callback(params):
            current_distance = objective_function(params)
            current_constraint = constraint_func(params)
            print(f"Objective function: {current_distance}, Constraints:")
            print(current_constraint)

        constraints = []
        for i in range(6):  # separate constraint functions for Study Quadric equation
            constraints.append({
                'type': 'eq',
                'fun': (lambda params, index=i: constraint_func(params)[index])
            })

        result = minimize(objective_function,
                          initial_guess,
                          constraints=constraints,
                          callback=callback,
                          options={'maxiter': 20,
                                   'ftol': 1e-14,
                                   },
                          )

        print(result)
        result_curve = MotionApproximation._construct_curve_nonmonic(result.x[:32])

        return result_curve, result

    @staticmethod
    def force_study_quadric(init_curve: RationalCurve):
        """Adjust a curve so its coefficients satisfy the Study quadric.

        The function optimizes coefficient values so that the resulting
        rational curve lies on the Study quadric (within numerical tolerance).

        Parameters
        ----------
        init_curve
            The :class:`.RationalCurve` whose coefficients are to be adjusted.

        Returns
        -------
        tuple
            ``(result_curve, result)`` where ``result_curve`` is the adjusted
            :class:`.RationalCurve` and ``result`` is the optimizer result.
        """
        initial_guess = init_curve.coeffs.flatten()

        def objective_func(params):
            curve = MotionApproximation._construct_curve_nonmonic(params[:32])
            sq_err = curve.study_quadric_check()

            # sum of squares of the errors
            return numpy.sum(sq_err**2)

        def callback(params):
            current_distance = objective_func(params)
            print(f"Objective function: {current_distance}")

        result = minimize(objective_func,
                          initial_guess,
                          method='Powell', # Powell, --TNC, --SLSQP
                          callback=callback,
                          tol=1e-14,
                          options={'maxiter': 100,
                                   'ftol': 1e-14,
                                   },
                          )

        print(result)
        result_curve = MotionApproximation._construct_curve_nonmonic(result.x[:32])

        return result_curve, result

