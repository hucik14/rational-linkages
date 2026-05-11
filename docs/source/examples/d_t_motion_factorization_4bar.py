from rational_linkages import (DualQuaternion, MotionFactorization,
                               RationalMechanism, Plotter)


f1 = MotionFactorization([DualQuaternion.as_rational([0, 0, 0, 1, 0, 0, 0, 0]),
                          DualQuaternion.as_rational([0, 0, 0, 2, 0, 0, -1, 0])])

# find factorizations
factorizations = f1.factorize()

# create mechanism
m = RationalMechanism(factorizations, tool='mid_of_last_link')

# plot mechanism
plt = Plotter(mechanism=m, arrows_length=0.05)
plt.show()

