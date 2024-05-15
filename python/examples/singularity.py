from rational_linkages.models import collisions_free_6r
from rational_linkages import SingularityAnalysis

import numpy as np

m = collisions_free_6r()
m.update_segments()

from sympy import Symbol
t = Symbol('t')


sa = SingularityAnalysis()

r = sa.check_singularity(m)

print(r)

