# NOT TESTED

from rational_linkages import Plotter, MotionInterpolation, PointHomogeneous, DualQuaternion, RationalMechanism
from rational_linkages.MotionApproximation import MotionApproximation

import numpy as np

# Define 7 points in PR3 space (homogeneous coordinates: [w, x, y, z])
points = [
    PointHomogeneous([1, 0, 0, 0]),    # Point 1
    PointHomogeneous([1, 2, 0, 0]),   # Point 2
    PointHomogeneous([1, 2, 1, 0]),   # Point 3
    PointHomogeneous([1, 0, 1, 0]),   # Point 4
    PointHomogeneous([1, 0, 1, 1]),   # Point 5
    PointHomogeneous([1, 2, 1, 1]),   # Point 6
    PointHomogeneous([1, 2, 0, 1]),    # Point 7
    PointHomogeneous([1, 0, 0, 1]),    # Added points
    # PointHomogeneous([1, 0.5, 0.5, 0.5]),
    # PointHomogeneous([1, 0.2, 0.1, 0.3]),
]

# Perform cubic interpolation
c = MotionInterpolation.interpolate(points[:7])

# Perform motion approximation

t_exp = np.array([1.7])
# p = PointHomogeneous.from_3d_point(DualQuaternion(c.evaluate(t_exp[0])).dq2point_via_matrix())
# points[-1] = p

t_vals_init = np.array([0, 1/6, 1/3, 1/2, 2/3, 5/6, 1])
t_vals = np.concatenate((t_vals_init, t_exp), axis=None)
ag, res = MotionApproximation.optimize_for_t_and_study_quadric_global(c, points, t_vals)

# ac, res = MotionApproximation.approximate(c, points, t_exp)
# a1, res1 = MotionApproximation.force_study_quadric(ac)
#
# t_vals_init = np.array([0, 1/6, 1/3, 1/2, 2/3, 5/6, 1])
# t_vals = np.concatenate((t_vals_init, t_exp), axis=None)
# a2, res2 = MotionApproximation.optimize_for_t_and_study_quadric(a1, points, t_vals)
#
# a3, res3 = MotionApproximation.force_study_quadric(ac)

# m = RationalMechanism(a.factorize())

# Plot the interpolated curve and the original points
plotter = Plotter()
# plotter = Plotter(mechanism=m)
plotter.plot(c, interval='closed', color='blue')
# plotter.plot(a2, interval='closed', color='red')
# plotter.plot(a3, interval='closed', color='green')

plotter.plot(ag, interval='closed', color='green')


for i, pt in enumerate(points):
    # Uncomment the line below to transform points into the mechanism path
    # pt = rebase.inv().act(pt)
    plotter.plot(pt, label=f'p{i}')

plotter.show()
