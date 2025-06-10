# Quadratic interpolation of 5 points

from rational_linkages import (Plotter, MotionInterpolation, PointHomogeneous,
                               DualQuaternion, RationalMechanism)


# Define 5 points in PR3 space (1st coordinate is projective, then x, y, z)
a0 = PointHomogeneous([1, 0, 0, 0])
a1 = PointHomogeneous([1, 1, 0, -2])
a2 = PointHomogeneous([1, 2, -1, 0])
a3 = PointHomogeneous([1, -3, 0, 3])
a4 = PointHomogeneous([1, 2, 1, -1])
points = [a0, a1, a2, a3, a4]

interpolated_curve = MotionInterpolation.interpolate(points)
m = RationalMechanism(interpolated_curve.factorize())

# due to non-monic solution, to transform the given points and plot them in mechanism
# path, get static transform 'rebase'
rebase = DualQuaternion(interpolated_curve.evaluate(1e12))

p = Plotter(mechanism=m, base=rebase, arrows_length=0.5)

p.plot(interpolated_curve, interval='closed')

for i, pt in enumerate(points):
    p.plot(pt, label=f'a{i}')

p.show()