from rational_linkages import Plotter, DualQuaternion, NormalizedLine, PointHomogeneous
from copy import deepcopy

pt1 = PointHomogeneous([1, 0.3, 1, 0])  # w = 1, x = 0.3, y = 1, z = 0
pt2 = PointHomogeneous([1, 0.3, 1, 1])  # w = 1, x = 0.3, y = 1, z = 1
l = NormalizedLine.from_two_points(pt1, pt2)

dq1 = DualQuaternion(l.line2dq_array())

dq2 = deepcopy(
    dq1)  # alter the rotation part of the dual quaternion to get some other transformation
dq2[0] = 2

p = Plotter(arrows_length=0.5, backend='matplotlib')
p.plot(l)
p.plot(pt1, label='pt1', color='red')
p.plot(pt2, label='pt2', color='red')

p.plot(DualQuaternion(), label='origin')
p.plot(dq1, label='half-turn')
p.plot(dq2, label='some-rotation')
p.show()