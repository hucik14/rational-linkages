import rational_linkages
rational_linkages.set_backend("sympy")

from rational_linkages import DualQuaternion, TransfMatrix
from sympy import pprint, symbols

# Pure translation along x by distance 'a'.
a = symbols("a", positive=True)
dq_trans = DualQuaternion([1, 0, 0, 0, 0, -a/2, 0, 0])

mat = TransfMatrix(dq_trans.dq2matrix())
pprint(mat)
# [[1, 0, 0, 0],
#  [a, 1, 0, 0],
#  [0, 0, 1, 0],
#  [0, 0, 0, 1]]
# note the convention with projective coordinates on the top row
# and the translation in the first column