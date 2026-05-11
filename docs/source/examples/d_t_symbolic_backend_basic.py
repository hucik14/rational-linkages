import rational_linkages
rational_linkages.set_backend("sympy")

from rational_linkages import DualQuaternion
from sympy import symbols, Rational

# Declare eight real-valued symbols for the eight Study parameters.
p0, p1, p2, p3, d0, d1, d2, d3 = symbols("p0 p1 p2 p3 d0 d1 d2 d3", real=True)

dq = DualQuaternion([p0, p1, p2, p3, d0, d1, d2, d3])
print(dq)
# DQ([p0, p1, p2, p3, d0, d1, d2, d3])

# substitute inner parameters
dq_eval = dq.eval({p0: 1, p1: 0, p2: Rational(1,3), p3: 0, d0: 0, d1: 0, d2: 0, d3: 0})
print(dq_eval)

# evaluate numerically
dq_eval_num = dq_eval.evalf()
print(dq_eval_num)
