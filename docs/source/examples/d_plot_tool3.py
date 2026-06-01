"""
# Tool frame specified as DualQuaternion
"""

from rational_linkages import (RationalMechanism, DualQuaternion, TransfMatrix,
                               Plotter, MotionFactorization)


# Define factorizations
f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                          DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])

f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                          DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

# Create tool frame from transformation matrix
tool_matrix = TransfMatrix.from_rpy_xyz([90, 0, 45], [-0.2, 0.5, 0], unit='deg')
tool_dq = DualQuaternion(tool_matrix.matrix2dq())

# Create mechanism
m = RationalMechanism([f1, f2], tool=tool_dq)

# Create plotter
plt = Plotter(mechanism=m, backend='matplotlib', arrows_length=0.2)

# Plot the default motion curve
plt.plot(m.get_motion_curve(), label='motion curve', interval='closed',
       color='red', linewidth='0.7', linestyle=':')
plt.show()

