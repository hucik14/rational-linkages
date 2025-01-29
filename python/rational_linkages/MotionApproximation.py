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

        approx_curve = MotionApproximation._cubic_approximation(init_curve,
                                                                poses,
                                                                guessed_t,
                                                                num_poses,
                                                                num_params,
                                                                )

        return approx_curve

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

            distance = 0.
            for i, pose in enumerate(poses):
                curve_pose = DualQuaternion(curve.evaluate(guessed_t[i]))
                distance += metric.squared_distance(pose, curve_pose)

            # print('Objective Function Value:', distance)

            return distance

        def constraint(params):
            curve = MotionApproximation._construct_curve(params)

            t_vars = np.linspace(-10, 10, 100)

            pts = [DualQuaternion(curve.evaluate(some_t)) for some_t in t_vars]

            dist_to_study_quadric = 0.
            for pt in pts:
                dist_to_study_quadric += (pt[0] * pt[4] + pt[1] * pt[5] + pt[2] * pt[6] + pt[3] * pt[7])

            # print(f"Constraint Value: {dist_to_study_quadric}")

            return dist_to_study_quadric

        def callback(params):
            current_distance = objective_function(params)
            print(f"Current Objective Function Value: {current_distance}")

        constraints = {'type': 'eq', 'fun': constraint}

        result = minimize(objective_function,
                          initial_guess,
                          constraints=constraints,
                          # method='SLSQP',
                          callback=callback,
                          # options={'ftol': 1.},
                          )

        optimized_params = result.x
        print("Optimized Parameters:", optimized_params)
        print(result)

        result_curve = MotionApproximation._construct_curve(optimized_params)

        return result_curve



