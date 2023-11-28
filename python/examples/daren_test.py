import biquaternion_py as bq
import sympy as sy

from FactorizationProvider import FactorizationProvider
from DualQuaternion import DualQuaternion
from MotionFactorization import MotionFactorization



# Setting up the symbols
t = sy.Symbol("t")

# Given a Curve as a polynomial in the dual quaternions
# we first need to find the irreducible factors of the normpolynomial

# For this we construct a polynomial from linear factors (t-h)

h1bq = bq.BiQuaternion(
    [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]
)
h2bq = bq.BiQuaternion(
    [-3 / 10, 833 / 240, -451 / 160, 19 / 24, 0, -427 / 480, -1609 / 720, -1217 / 300]
)
h3bq = bq.BiQuaternion(
    [9 / 4, -96 / 385, -3 / 11, 12 / 121, 0, -9 / 22, 18 / 77, -27 / 70]
)

# h1 = bq.rand_line()
# h2 = bq.rand_line()
# h3 = bq.rand_line()

poly = bq.Poly((t - h1bq) * (t - h2bq) * (t - h3bq), t)

# Next we calculate the norm polynomial. To avoid numerical problems, we extract
# the scalar part, since the norm should be purely real anyhow.

norm_poly = poly.norm()
norm_poly = bq.Poly(norm_poly.poly.scal, *norm_poly.indets)

# From this we can calculate the irreducible factors, that then determine the different
# factorizations

_, factors = bq.irreducible_factors(norm_poly)

# The different permutations of the irreducible factors then generate the different
# factorizations of the motion.

factorization1 = bq.factorize_from_list(poly, factors)
factorization2 = bq.factorize_from_list(poly, [factors[1], factors[2], factors[0]])

# We can now calculate the direct kinematics of the motion

h1 = DualQuaternion([-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5], is_rotation=True)
h2 = DualQuaternion([-3 / 10, 833 / 240, -451 / 160, 19 / 24, 0, -427 / 480, -1609 / 720, -1217 / 300], is_rotation=True)
h3 = DualQuaternion([9 / 4, -96 / 385, -3 / 11, 12 / 121, 0, -9 / 22, 18 / 77, -27 / 70], is_rotation=True)

my_f = MotionFactorization([h1, h2, h3])
factorize = FactorizationProvider.factorize_for_motion_factorization(my_f)