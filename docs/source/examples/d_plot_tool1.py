"""
# Tool frame on motion curve
"""

from rational_linkages import (RationalMechanism, DualQuaternion,
                               Plotter, MotionFactorization)


# Define factorizations
f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                          DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])

f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                          DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

# Create mechanism
m = RationalMechanism([f1, f2])

# Create plotter
plt = Plotter(mechanism=m, backend='matplotlib', arrows_length=0.2)
plt.show()

