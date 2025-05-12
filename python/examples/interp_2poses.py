# Quadratic interpolation of 2 poses with an optimized 3rd pose

from rational_linkages import (DualQuaternion, Plotter, MotionInterpolation,
                               TransfMatrix, RationalMechanism)


p2 = DualQuaternion()
p0 = TransfMatrix.from_rpy_xyz([0.5, 0, 0], [1, 0, 0])
p1 = TransfMatrix.from_rpy_xyz([0, 0, 0.5], [0, 1, 0])

p0 = TransfMatrix.from_rpy_xyz([180, 0, 0], [0.1, 0.2, 0.1], unit='deg')
p1 = TransfMatrix.from_rpy_xyz([-90, 0, 90], [0.15, -0.2, 0.2], unit='deg')



p0 = TransfMatrix.from_rpy_xyz([0, 0, 0], [0.1, 0.2, 0.1], unit='deg')
p1 = TransfMatrix.from_rpy_xyz([0, 0, 90], [0.15, -0.2, 0.2], unit='deg')

p0 = DualQuaternion(p0.matrix2dq())
p1 = DualQuaternion(p1.matrix2dq())

interpolated_curve = MotionInterpolation.interpolate([p0, p1])
m = RationalMechanism(interpolated_curve.factorize())

p = Plotter(mechanism=m, steps=500, arrows_length=0.05)
p.plot(p0)
p.plot(p1)

p.plot(interpolated_curve, interval='closed', label='interpolated curve')

p.show()
