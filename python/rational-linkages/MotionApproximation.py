import numpy as np
from scipy.optimize import minimize
from DualQuaternion import DualQuaternion


class MotionApproximation:
    """
    MotionApproximation class
    """

    def __init__(self, poses: list[DualQuaternion]):
        """
        Initializes a MotionApproximation object

        :param poses: list[DualQuaternion] - list of poses
        """
        self.poses = poses
        self.num_poses = len(poses)
        self.num_params = int(8 * 3 + 2)  # for degree 2 curve

        self.curve = self.get_curve()

    def get_curve(self):
        """
        Get the curve of the motion approximation

        :return: MotionApproximationCurve
        """

        initial_guess = np.zeros(self.num_params)
        initial_guess[2] = 1.0

        def objective_function(params):
            """
            Objective function to minimize the sum of squared distances between
            the poses and the curve
            """
            #t, s0, s1, s2, i0, i1, i2, j0, j1, j2, k0, k1, k2, es0, es1, es2, ei0, ei1, ei2, ej0, ej1, ej2, ek0, ek1, ek2 = params
            return np.sum([val ** 2 for val in params[2::]])

        def constraint(params):
            t0, t1, s0, s1, s2, i0, i1, i2, j0, j1, j2, k0, k1, k2, es0, es1, es2, ei0, ei1, ei2, ej0, ej1, ej2, ek0, ek1, ek2 = params
            t = [t0, t1]

            #c0 = ((s0 + s1 * t + s2 * t ** 2) * (es0 + es1 * t + es2 * t ** 2)
            #      + (i0 + i1 * t + i2 * t ** 2) * (ei0 + ei1 * t + ei2 * t ** 2)
            #      + (j0 + j1 * t + j2 * t ** 2) * (ej0 + ej1 * t + ej2 * t ** 2)
            #      + (k0 + k1 * t + k2 * t ** 2) * (ek0 + ek1 * t + ek2 * t ** 2))

            c = []

            for i in range(self.num_poses):
                c.append(s0 + s1 * t[i] + s2 * t[i] ** 2 - self.poses[i].dq[0])
                c.append(i0 + i1 * t[i] + i2 * t[i] ** 2 - self.poses[i].dq[1])
                c.append(j0 + j1 * t[i] + j2 * t[i] ** 2 - self.poses[i].dq[2])
                c.append(k0 + k1 * t[i] + k2 * t[i] ** 2 - self.poses[i].dq[3])
                c.append(es0 + es1 * t[i] + es2 * t[i] ** 2 - self.poses[i].dq[4])
                c.append(ei0 + ei1 * t[i] + ei2 * t[i] ** 2 - self.poses[i].dq[5])
                c.append(ej0 + ej1 * t[i] + ej2 * t[i] ** 2 - self.poses[i].dq[6])
                c.append(ek0 + ek1 * t[i] + ek2 * t[i] ** 2 - self.poses[i].dq[7])

            return c

        constraints = {'type': 'eq', 'fun': constraint}

        result = minimize(objective_function, initial_guess, constraints=constraints, method='SLSQP')

        optimized_params = result.x
        print("Optimized Parameters:", optimized_params)
        print(result)

        return result



