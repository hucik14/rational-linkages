# Create a transformation matrix from Tait-Bryan angles and translation vector,
# and convert it to dual quaternion

from rational_linkages import TransfMatrix, DualQuaternion
from math import pi

# Identity/origin
T0 = TransfMatrix()

# Create a transformation matrix from Tait-Bryan angles and translation vector
T1 = TransfMatrix.from_rpy_xyz([pi / 2, 0, 0], [1, 2, 3])

# Create a transformation matrix from Tait-Bryan angles and translation vector,
# use degrees instead of radians
T2 = TransfMatrix.from_rpy_xyz([0, -90, 0], [4, 5, 6], unit='deg')

# Convert the transformation matrices to dual quaternions
T_list = [T0, T1, T2]

for T in T_list:
    p = DualQuaternion(T.matrix2dq())
    print("--------------------")
    print("Transformation matrix:")
    print(T)
    print("Corresponding dual quaternion:")
    print(p)
    print("--------------------")

# Create TransfMatrix from DualQuaternion
p = DualQuaternion(T2.matrix2dq())
T = TransfMatrix(p.dq2matrix())
print(T)
