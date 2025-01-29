import numpy as np
from scipy.optimize import minimize

from .DualQuaternion import DualQuaternion
from .AffineMetric import AffineMetric
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve


best_params = None
lowest_constraint_violation = float('inf')


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

            # print('Objective Function Value:', sq_dist)

            return sq_dist

        def constraint(params):
            curve = MotionApproximation._construct_curve(params)

            t_vars = np.linspace(-10, 10, 100)

            pts = [DualQuaternion(curve.evaluate(some_t)) for some_t in t_vars]

            constr_eqs = []
            constr_sum = 0.
            for pt in pts:
                constr_eqs.append(pt[0] * pt[4] + pt[1] * pt[5] + pt[2] * pt[6] + pt[3] * pt[7])
                constr_sum += abs(pt[0] * pt[4] + pt[1] * pt[5] + pt[2] * pt[6] + pt[3] * pt[7])

            # print(f"Constraint Value: {dist_to_study_quadric}")

            return constraints

        def callback(params):
            global best_params, lowest_constraint_violation
            current_distance = objective_function(params)
            print(f"Current OF: {current_distance}, constraints: {current_constraints}")

            if abs(current_constraints) < lowest_constraint_violation:
                lowest_constraint_violation = abs(current_constraints)
                best_params = params.copy()

        constraints = {'type': 'eq', 'fun': constraint}

        result = minimize(objective_function,
                          initial_guess,
                          constraints=constraints,
                          callback=callback,
                          options={'maxiter': 200},
                          )

        print(result)

        result_curve = MotionApproximation._construct_curve(result.x)

        global best_params

        if best_params is None:
            return result_curve, result
        else:
            return MotionApproximation._construct_curve(best_params), result



