import sympy as sy
import numpy as np
import biquaternion_py as bq
from biquaternion_py import II, JJ, KK, EE

from rational_linkages import DualQuaternion, Plotter, RationalMechanism, MotionFactorization

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

h1_org = DualQuaternion.from_bq_biquaternion(h1)
h2_org = DualQuaternion.from_bq_biquaternion(2*KK + EE * II)

h1 = DualQuaternion.from_bq_poly(F1[0], t)
h2 = DualQuaternion.from_bq_poly(F1[1], t)
k1 = DualQuaternion.from_bq_poly(F2[0], t)
k2 = DualQuaternion.from_bq_poly(F2[1], t)



f1 = MotionFactorization([h1, h2])
f2 = MotionFactorization([k1, k2])

m = RationalMechanism([f1, f2])
plt = Plotter(interactive = True, steps = 500, arrows_length = 0.2)
plt.plot(m, show_tool = True)
plt.show()
