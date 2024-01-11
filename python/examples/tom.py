from rational_linkages import DualQuaternion
from rational_linkages import PointHomogeneous
from rational_linkages import NormalizedLine
from TransfMatrix import TransfMatrix
from Plotter import Plotter
from MotionApproximation import MotionApproximation
import numpy as np

#plt = Plotter()

# be careful of the convention of the TransfMatrix class (european vs. american)

# create transformation matrix from roll, pitch, yaw angles
tf = TransfMatrix.from_rpy([0, 0, 45], units='deg')
# add translation
tf.t = [3, 1, 2]

tf2dq = DualQuaternion(tf.matrix2dq())

origin = DualQuaternion()

pt1 = PointHomogeneous([1, 1, 0, 0])
pt2 = PointHomogeneous([1, 0, 1, 1])

line = NormalizedLine.from_two_points(pt1, pt2)
"""
plt.plot(pt1, label="bod 1")
plt.plot(pt2, label="bod 2")
plt.plot(line, label="primka")

plt.plot(tf, label="tf_matice")
plt.plot(origin, label="origin")

a = MotionApproximation([origin, tf2dq])
"""
o = TransfMatrix()
t = TransfMatrix.from_rpy_xyz([0, 0, -90], [2, 0, -3], units='deg')
print(t)
print(o.dh_to_other_frame(t))

