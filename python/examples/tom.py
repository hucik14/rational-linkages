from DualQuaternion import DualQuaternion
from PointHomogeneous import PointHomogeneous
from NormalizedLine import NormalizedLine
from TransfMatrix import TransfMatrix
from Plotter import Plotter


plt = Plotter()

tf = TransfMatrix()
# add translation
tf.t = [3, 1, 2]

origin = DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0])

pt1 = PointHomogeneous([1, 1, 0, 0])
pt2 = PointHomogeneous([1, 0, 1, 1])

line = NormalizedLine.from_two_points(pt1, pt2)

plt.plot(pt1, label="bod 1")
plt.plot(pt2, label="bod 2")
plt.plot(line, label="primka")

plt.plot(tf, label="tf_matice")
plt.plot(origin, label="origin")

