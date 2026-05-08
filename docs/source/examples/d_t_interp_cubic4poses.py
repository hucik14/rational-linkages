"""
# Cubic interpolation of 4 poses
"""

from rational_linkages import DualQuaternion, Plotter, MotionInterpolation, RationalMechanism


# 4 poses
p0 = DualQuaternion()  # identity
p1 = DualQuaternion.as_rational([0, 0, 0, 1, 1, 0, 1, 0])
p2 = DualQuaternion.as_rational([1, 2, 0, 0, -2, 1, 0, 0])
p3 = DualQuaternion.as_rational([3, 0, 1, 0, 1, 0, -3, 0])

# obtain the interpolated motion curve
c = MotionInterpolation.interpolate([p0, p1, p2, p3])

# factorize the motion curve
fs = c.factorize()

# create a mechanism from the factorization
m = RationalMechanism(fs)

# create an interactive plotter object, with 500 descrete steps
# for the input rational curves, and arrows scaled to 0.05 length
myplt = Plotter(mechanism=m, steps=500, arrows_length=0.5)

# plot the poses
for pose in [p0, p1, p2, p3]:
    myplt.plot(pose)

# show the plot
myplt.show()

