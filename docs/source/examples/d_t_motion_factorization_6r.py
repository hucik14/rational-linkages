from rational_linkages import (DualQuaternion, MotionFactorization, RationalMechanism,
                               Plotter)


h1 = DualQuaternion.as_rational([0, 1, 0, 0, 0, 0, 0, 0])
h2 = DualQuaternion.as_rational([0, 0, 3, 0, 0, 0, 0, 1])
h3 = DualQuaternion.as_rational([0, 1, 1, 0, 0, 0, 0, -2])

f1 = MotionFactorization([h1, h2, h3])

# find factorizations
factorizations = f1.factorize()

# create mechanism
m = RationalMechanism(factorizations)

# plot mechanism
plt = Plotter(mechanism=m, arrows_length=0.2)
plt.show()

