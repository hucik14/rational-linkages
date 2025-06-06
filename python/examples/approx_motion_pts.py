from rational_linkages import Plotter, MotionInterpolation, PointHomogeneous, MotionApproximation, DualQuaternion, RationalMechanism

import numpy as np

# Define 7 points in PR3 space (homogeneous coordinates: [w, x, y, z])
points = [
    PointHomogeneous([1, 0, 0, 0]),    # Point 1
    PointHomogeneous([1, 1, 0, -2]),   # Point 2
    PointHomogeneous([1, 2, -1, 0]),   # Point 3
    PointHomogeneous([1, -3, 0, 3]),   # Point 4
    PointHomogeneous([1, 2, 1, -1]),   # Point 5
    PointHomogeneous([1, 2, 3, -3]),   # Point 6
    PointHomogeneous([1, 1, 1, 1]),    # Point 7
    PointHomogeneous([1, 1, 0.3, 0.5]),    # Added point
]

# Perform cubic interpolation
c = MotionInterpolation.interpolate(points[:7])

# Perform motion approximation

t_exp = np.array([1.7])
# p = PointHomogeneous.from_3d_point(DualQuaternion(c.evaluate(t_exp[0])).dq2point_via_matrix())
# points[-1] = p

ac, res = MotionApproximation.approximate(c, points, t_exp)
a, res2 = MotionApproximation.force_study_quadric(ac)

# m = RationalMechanism(a.factorize())

# Plot the interpolated curve and the original points
plotter = Plotter()
# plotter = Plotter(mechanism=m)
plotter.plot(c, interval='closed', color='blue')
plotter.plot(ac, interval='closed', color='red')
plotter.plot(a, interval='closed')


for i, pt in enumerate(points):
    # Uncomment the line below to transform points into the mechanism path
    # pt = rebase.inv().act(pt)
    plotter.plot(pt, label=f'p{i}')

plotter.show()
