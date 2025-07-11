# NOT TESTED

import numpy as np

from rational_linkages import (
    DualQuaternion,
    MotionInterpolation,
    Plotter,
    RationalMechanism,
    TransfMatrix,
)

from rational_linkages.MotionApproximation import MotionApproximation


p0 = DualQuaternion([1., 0, 0, 0, 0, 0, 0, 0])
p1 = DualQuaternion([0., 0, 0, 1, 1, 0, 1, 0])
p2 = DualQuaternion([1., 2, 0, 0, -2, 1, 0, 0])
p3 = DualQuaternion([3., 0, 1, 0, 1, 0, -3, 0])

c = MotionInterpolation.interpolate([p0, p1, p2, p3])

# t_sols2 = MotionInterpolation._solve_for_t([p0, p1, p2, p3], MotionInterpolation._obtain_k_dq([p0, p1, p2, p3]))
# print(t_sols2)
t_sols = MotionInterpolation._solve_for_t([p0, p1, p2, p3], MotionInterpolation._obtain_k_dq([p0, p1, p2, p3])[0])
# print(t_sols)

add_t = 1.7
p4 = TransfMatrix(c.evaluate_as_matrix(add_t)) * TransfMatrix.from_rpy_xyz([5, -2, 4], [-0.3, 0, 0.2], unit='deg')
p4 = DualQuaternion(p4.matrix2dq())

add_t2 = 30
p5 = TransfMatrix(c.evaluate_as_matrix(add_t2)) * TransfMatrix.from_rpy_xyz([5, -3, 5], [-0.2, 0.3, 0.1], unit='deg')
p5 = DualQuaternion(p5.matrix2dq())

myinf = 1/np.finfo(float).eps
ts = np.concatenate((myinf, t_sols, [add_t, add_t2]), axis=None)

a, res = MotionApproximation.approximate(c, [p0, p1, p2, p3, p4, p5], ts)

af, res2 = MotionApproximation.force_study_quadric(a)

# p = Plotter(mechanism=m, steps=1000, arrows_length=0.5)
p = Plotter(arrows_length=0.5)

p.plot(a, interval='closed', color='blue')
p.plot(af, interval='closed', color='red')

# if a.is_on_study_quadric():
#     m = RationalMechanism(a.factorize())
#     p.plot(m.curve(), interval='closed', with_poses=True)
#     print("")
#     print("Result on Study Quadric")
# else:
#     p.plot(a, interval='closed')
#     print("")
#     print("Optimization did not converge (result not on study quadric)")


poses_to_plot = [p0, p1, p2, p3, p4, p5]

for i, pose in enumerate(poses_to_plot):
    p.plot(pose, label=f"p{i}")

# p.plot(c, interval='closed', linestyle=':', color='gray')

p.show()
