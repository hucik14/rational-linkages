import numpy as np
from scipy.optimize import minimize

from .DualQuaternion import DualQuaternion
from .AffineMetric import AffineMetric
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve


class MotionApproximation:
    """
    MotionApproximation class
    """
    def __init__(self):
        pass

    @staticmethod
    def approximate(init_curve,
                    init_poses,
                    poses: list[DualQuaternion],
                    guessed_t: list[float]):
        """
        Initializes a MotionApproximation object

        :param poses: list[DualQuaternion] - list of poses
        """
        num_poses = len(init_poses) + len(poses)
        num_params = int(8 * 3)

        poses = init_poses + poses

        approx_curve, opt_result = MotionApproximation._cubic_approximation(init_curve,
                                                                            poses,
                                                                            guessed_t,
                                                                            num_poses,
                                                                            num_params,
                                                                            )

        return approx_curve, opt_result

    @staticmethod
    def _construct_curve(flattended_coeffs):
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
                             guessed_t,
                             num_poses,
                             num_params,):
        """
        Get the curve of the cubic motion approximation

        :return: MotionApproximationCurve
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
                curve_pose = DualQuaternion(curve.evaluate(guessed_t[i]))
                sq_dist += metric.squared_distance(pose, curve_pose)

            # # Compute constraint violation penalty
            # constraint_violations = constraint_func(params)
            # penalty_weight = 1e5  # Large weight to prioritize constraints
            # penalty = penalty_weight * constraint_violations ** 2

            return sq_dist #+ penalty

        def constraint_func(params):
            curve = MotionApproximation._construct_curve(params)

            poly_list = [np.polynomial.Polynomial(curve.coeffs[i, :][::-1])
                         for i in range(8)]

            sq_err = (poly_list[0] * poly_list[4] + poly_list[1] * poly_list[5]
                      + poly_list[2] * poly_list[6] + poly_list[3] * poly_list[7])

            if len(sq_err.coef) != 8:
                # expand to 8 coefficients
                sq_err.coef = np.concatenate((sq_err.coef, np.zeros(8 - len(sq_err.coef))), axis=None)

            # return sum(np.array(sq_err.coef) ** 2)
            return sq_err.coef

        def callback(params):
            current_distance = objective_function(params)
            current_constraint = constraint_func(params)
            print(f"OF: {current_distance}, Constraints: {current_constraint}")

        # constraints = {'type': 'eq', 'fun': constraint_func}
        constraints = []
        for i in range(8):  # Create 8 separate constraint functions
            constraints.append({
                'type': 'eq',
                'fun': lambda params, index=i: constraint_func(params)[index]
            })

        result = minimize(objective_function,
                          initial_guess,
                          constraints=constraints,  # Use the list of constraints
                          callback=callback,
                          options={'maxiter': 100,
                                   'ftol': 1e-16,
                                   },
                          )

        print(result)

        result_curve = MotionApproximation._construct_curve(result.x)

        return result_curve, result
