"""
# Plotting static objects
"""

from rational_linkages import Plotter, DualQuaternion, PointHomogeneous, NormalizedLine, TransfMatrix


# create plotter object, arg steps says how many discrete steps will be used for
# plotting curves
plt = Plotter(backend='matplotlib')

# create two DualQuaternion objects
identity = DualQuaternion()
pose1 = DualQuaternion([0, 0, 1, 0, 0, -0.5, 1, 0])
pose2 = TransfMatrix.from_rpy_xyz([0, -90, 0], [0, 0, 0.5], unit='deg')

# create a point with homogeneous coordinates w = 1, x = 2, y = -3, z = 1.5
point = PointHomogeneous([1, 2, -3, 1.5])

# create a normalized line from direction vector and the previously specified point
line = NormalizedLine.from_direction_and_point([0, 0, 1], point.normalized_euclidean())

# plot the objects
# 1-line command
plt.plot(identity, label='base')
plt.plot(point, label='pt')
plt.plot(line, label='l1')
# or for cycle
for i, obj in enumerate([pose1, pose2]):
    plt.plot(obj, label='p{}'.format(i + 1))

plt.show()

