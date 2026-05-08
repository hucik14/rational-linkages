# Quadratic interpolation of 3 poses

from rational_linkages import DualQuaternion, Plotter, MotionInterpolation


p0 = DualQuaternion([0, 17, -33, -89, 0, -6, 5, -3])
p1 = DualQuaternion([0, 84, -21, -287, 0, -30, 3, -9])
p2 = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3])

c = MotionInterpolation.interpolate([p0, p1, p2])

plt = Plotter(steps=500, arrows_length=0.05)
plt.plot(c, interval='closed')

for i, pose in enumerate([p0, p1, p2]):
    plt.plot(pose, label='p{}'.format(i+1))
plt.show()

