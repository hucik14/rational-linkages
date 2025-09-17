# This tutorial explains how to obtain Pluecker coordinates (joint screw axes)
# for a Bennett 4R linkage with rational parameters using sympy.
# This is a necessary prerequisite for recovering a motion curve as a rational curve.


import sympy as sp
from rational_linkages.utils import tr_from_dh_rationally, normalized_line_rationally

# Define rational zero and one
r_zero = sp.Rational(0)
r_one = sp.Rational(1)

# Define link lengths and twist angles
a0 = sp.Rational(220, 1000)  # 220 mm in meters
t0 = r_one  # tan(90/2) = 1; eventually approximate it
# as a rational number as in the case of t1 bellow

# Approximate tan(150/2) = tan(75°) as a rational number
t1 = sp.Rational(3732, 1000)  # tan(75°) ≈ 3.732

# Adjust a1 to maintain the Bennett condition
a1 = a0 * ((2*t1)/(1+t1**2)) * ((2*t0)/(1+t0**2))  # ≈ 110.001 mm

# Define Denavit-Hartenberg (DH) parameters
theta = [r_zero, r_zero, r_zero, r_zero]
d = [r_zero, r_zero, r_zero, r_zero]
a = [a0, a1, a0, a1]
alpha = [t0, t1, t0, t1]

# Create local transformation matrices from DH parameters
local_tm = []
for i in range(4):
    local_tm.append(tr_from_dh_rationally(theta[i], d[i], a[i], alpha[i]))

# Define a 90° rotation around the Z-axis as a transformation matrix
rotate_z_pi2 = tr_from_dh_rationally(r_one, r_zero, r_zero, r_zero)

# Linkage closure adjustment
# By default, the DH parameters place links in series along the global X axis
# Adjust the second and fourth joints to close the linkage (twice 90° rotation is 180°)
# Rotate both (first and third) axes by 180° around Z-axis
local_tm[1] = rotate_z_pi2 * rotate_z_pi2 * local_tm[1]
local_tm[3] = rotate_z_pi2 * rotate_z_pi2 * local_tm[3]

# Compute global transformation matrices
global_tm = [local_tm[0]]
for i in range(1, len(local_tm)):
    global_tm.append(global_tm[i - 1] * local_tm[i])

# The linkage closure is satisfied if the last global_tm is identity matrix
assert all(sp.simplify(global_tm[-1][i,j] - sp.eye(4)[i,j]) == 0
           for i in range(4) for j in range(4))

# Initialize the first joint axis (Plücker coordinates)
screw_axes_rat = [sp.Matrix([0, 0, 1, 0, 0, 0])]

# Compute the remaining joint axes
for tm in global_tm[:-1]:
    tm_z_vector = tm[1:4, 3]
    tm_t_vector = tm[1:4, 0]
    tm_z_vector = [el for el in tm_z_vector]
    tm_t_vector = [el for el in tm_t_vector]
    screw_axes_rat.append(normalized_line_rationally(tm_t_vector, tm_z_vector))

# Print the results
print("Screw axes (Plücker coordinates):")
for i, screw in enumerate(screw_axes_rat):
    print(f"Screw axis {i}: {screw.T}")
