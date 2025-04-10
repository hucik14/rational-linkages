import numpy as np
from scipy.optimize import minimize
from typing import Union

from .DualQuaternion import DualQuaternion
from .AffineMetric import AffineMetric
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve


### NOT YET in the documentation ### TODO: add to docs


class MotionApproximation:
    """
    MotionApproximation class
    """
    def __init__(self):
        pass

    @staticmethod
    def approximate(init_curve,
                    poses: list[DualQuaternion],
                    t_vals: Union[list[float], np.ndarray]
                    ) -> tuple[RationalCurve, dict]:
        """
        Approximate a motion curve that passes through the given poses

        :param RationalCurve init_curve: initial curve (guess), use interpolation
            algorithm from :class:`.MotionInterpolation.MotionInterpolation` to get
            a good initial guess
        :param list[DualQuaternion] poses: poses to be approximated
        :param Union[list[float], np.ndarray] t_vals: parameter t values for the poses
            in the same order

        :return: Approximated curve and optimization result
        :rtype: tuple[RationalCurve, dict]
        """
        if init_curve.degree != 3:
            raise ValueError("So far, only cubic curves are supported")

        t_array = np.asarray(t_vals)
        approx_curve, opt_result = MotionApproximation._cubic_approximation(init_curve,
                                                                            poses,
                                                                            t_array)

        return approx_curve, opt_result

    @staticmethod
    def _construct_curve(flattended_coeffs) -> RationalCurve:
        """
        Construct a RationalCurve from the flattened coefficients

        :param flattended_coeffs: flattened coefficients

        :return: RationalCurve constructed from the coefficients
        :rtype: RationalCurve
        """
        coeffs = np.zeros((8, 4))  # Preallocate an array of shape (8, 4)
        coeffs[0, 0] = 1
        coeffs[:, 1:] = flattended_coeffs.reshape(8, 3)

        return RationalCurve.from_coeffs(coeffs)

    @staticmethod
    def _cubic_approximation(init_curve,
                             poses,
                             t_vals) -> tuple[RationalCurve, dict]:
        """
        Get the curve of the cubic motion approximation

        :return: Approximated curve
        :rtype: tuple[RationalCurve, dict]
        """
        metric = AffineMetric(init_curve,
                              [PointHomogeneous.from_3d_point(pose.dq2point_via_matrix())
                               for pose in poses])

        num_added_poses = len(poses) - 4

        initial_guess = init_curve.coeffs[:,1:4].flatten()
        initial_guess = np.concatenate((initial_guess, t_vals[-num_added_poses:]), axis=None)

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
                sq_err = np.concatenate((sq_err, np.zeros(8 - len(sq_err))), axis=None)

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
                          options={'maxiter': 200,
                                   'ftol': 1e-16,
                                   },
                          )

        print(result)
        result_curve = MotionApproximation._construct_curve(result.x[:24])

        return result_curve, result
