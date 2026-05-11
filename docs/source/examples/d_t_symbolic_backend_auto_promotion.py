from rational_linkages import DualQuaternion
from sympy import Rational, symbols

dq_numeric = DualQuaternion()
print(type(dq_numeric).__name__)  # DualQuaternion (numeric)

# Rational coefficients: no set_backend() call needed.
dq_rational = DualQuaternion([Rational(1, 2), 0, 0, 0, 0, 0, 0, 0])
print(type(dq_rational).__name__)  # DualQuaternionSymbolic

# Symbolic coefficients: again promoted automatically.
t = symbols("t")
dq_sym = DualQuaternion([1, t, 0, 0, 0, t**2, 0, 0])
print(type(dq_sym).__name__)  # DualQuaternionSymbolic

