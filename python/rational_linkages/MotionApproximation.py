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

        approx_curve, opt_result = MotionApproximation._cubic_approximation(init_curve,
                                                                            poses,
                                                                            t_vals)

        return approx_curve, opt_result

    @staticmethod
    def _construct_curve(flattended_coeffs) -> RationalCurve:
        """
        Construct a RationalCurve from the flattened coefficients

        :param flattended_coeffs: flattened coefficients

        :return: RationalCurve constructed from the coefficients
        :rtype: RationalCurve
        """
        coeffs = np.array([np.concatenate(([1], flattended_coeffs[:3]), axis=None),
                           np.concatenate(([0], flattended_coeffs[3:6]), axis=None),
                           np.concatenate(([0], flattended_coeffs[6:9]), axis=None),
                           np.concatenate(([0], flattended_coeffs[9:12]), axis=None),
                           np.concatenate(([0], flattended_coeffs[12:15]), axis=None),
                           np.concatenate(([0], flattended_coeffs[15:18]), axis=None),
                           np.concatenate(([0], flattended_coeffs[18:21]), axis=None),
                           np.concatenate(([0], flattended_coeffs[21:]), axis=None)
                           ])
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

        initial_guess = init_curve.coeffs[:,1:4].flatten()

        def objective_function(params):
            """
            Objective function to minimize the sum of squared distances between
            the poses and the curve
            """
            curve = MotionApproximation._construct_curve(params)

            sq_dist = 0.
            for i, pose in enumerate(poses):
                curve_pose = DualQuaternion(curve.evaluate(t_vals[i]))
                sq_dist += metric.squared_distance(pose, curve_pose)

            return sq_dist

        def constraint_func(params):
            curve = MotionApproximation._construct_curve(params)
            sq_err = curve.study_quadric_check()

            if len(sq_err) != 6:  # expand if necessary to avoid index errors
                sq_err = np.concatenate((sq_err, np.zeros(6 - len(sq_err))), axis=None)

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
        result_curve = MotionApproximation._construct_curve(result.x)

        return result_curve, result
