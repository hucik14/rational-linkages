import numpy as np

from rational_linkages import (
    DualQuaternion,
    MotionInterpolation,
    Plotter,
    RationalMechanism,
    TransfMatrix,
    MotionApproximation,
)


p0 = DualQuaternion([1., 0, 0, 0, 0, 0, 0, 0])
p1 = DualQuaternion([0., 0, 0, 1, 1, 0, 1, 0])
p2 = DualQuaternion([1., 2, 0, 0, -2, 1, 0, 0])
p3 = DualQuaternion([3., 0, 1, 0, 1, 0, -3, 0])

c = MotionInterpolation.interpolate([p0, p1, p2, p3])

t_sols = MotionInterpolation._solve_for_t([p0, p1, p2, p3], MotionInterpolation._obtain_k_dq([p0, p1, p2, p3]))
t_sols2 = MotionInterpolation.solve_for_t_numerically([p0, p1, p2, p3], MotionInterpolation._obtain_k_dq([p0, p1, p2, p3]))
print(t_sols)
print(t_sols2)

add_t = 1.7
p4 = TransfMatrix(c.evaluate_as_matrix(add_t)) * TransfMatrix.from_rpy_xyz([5, -2, 4], [-0.3, 0, 0.2], unit='deg')
p4 = DualQuaternion(p4.matrix2dq())

myinf = 1/np.finfo(float).eps
ts = np.concatenate((myinf, t_sols, [add_t]), axis=None)

a, res = MotionApproximation.approximate(c, [p0, p1, p2, p3, p4], ts)

p = Plotter(interactive=True, steps=1000, arrows_length=0.5)

if a.is_on_study_quadric():
    m = RationalMechanism(a.factorize())
    p.plot(m, show_tool=True)
    print("")
    print("Result on Study Quadric")
else:
    p.plot(a, interval='closed')
    print("")
    print("Optimization did not converge (result not on study quadric)")


poses_to_plot = [p0, p1, p2, p3, p4]

for i, pose in enumerate(poses_to_plot):
    p.plot(pose, label=f"p{i}")

p.plot(c, interval='closed', linestyle=':', color='gray')

p.show()
