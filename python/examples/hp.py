import sympy as sy
import biquaternion_py as bq
from biquaternion_py import II, JJ, KK, EE

from rational_linkages import DualQuaternion, Plotter, RationalMechanism, MotionFactorization, TransfMatrix

# Create a planar motion polynomial
t = sy.symbols('t')
h1 = KK
h2 = 2*KK + EE * II
C = bq.Poly(bq.Poly(t - h1, t) * bq.Poly(t - h2, t))

# Compute its two factorizations
M1 = bq.Poly(bq.Poly(t - h1, *C.indets).norm().poly.scal, t)
M2 = bq.Poly(bq.Poly(t - h2, *C.indets).norm().poly.scal, t)
F1 = bq.factorize_from_list(C, [M1, M2])
F2 = bq.factorize_from_list(C, [M2, M1])

####################################################################
# END OF BQ PART
#
# DualQuaternion objects can be created from biquaternions directly:
# dq_from_bq = DualQuaternion.from_bq_biquaternion(2*KK + EE * II)
####################################################################

h1 = DualQuaternion.from_bq_poly(F1[0], t)
h2 = DualQuaternion.from_bq_poly(F1[1], t)
k1 = DualQuaternion.from_bq_poly(F2[0], t)
k2 = DualQuaternion.from_bq_poly(F2[1], t)

f1 = MotionFactorization([h1, h2])
f2 = MotionFactorization([k1, k2])

# specify tool frame (from TransfMatrix or directly as DualQuaternion)
tool = TransfMatrix.from_rpy_xyz([45, 33, -90], [0.5, -0.1, 0.3], units='deg')
tool_dq = DualQuaternion(tool.matrix2dq())

# create a mechanism from the two factorizations, tool frame is optional, but must be an object of type DualQuaternion
m = RationalMechanism([f1, f2], tool=tool_dq)

# initialize the plotter
plt = Plotter(interactive=True, steps=200, arrows_length=0.2)
plt.plot(m, show_tool=True)

# plot also the original motion path
plt.plot(m.get_motion_curve(), interval='closed', color='red', linewidth=0.5, linestyle=':')

plt.show()
