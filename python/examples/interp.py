import numpy as np

from DualQuaternion import DualQuaternion
from TransfMatrix import TransfMatrix
from RationalCurve import RationalCurve
import sympy as sp
from Plotter import Plotter
from FactorizationProvider import FactorizationProvider


x = sp.symbols("x:4")
y = sp.symbols("y:4")
lam = sp.symbols("lam")
t = sp.symbols("t:4")

P1 = DualQuaternion.as_rational()
P2 = DualQuaternion.as_rational([3135, 31350, 6270, 10450, 4296, -4065, -3948, 13275])
P3 = DualQuaternion.as_rational([-2/5, 1/5, -3/2, 4, 1409/5535, 5753/5535, 998/369, 5467/5535])
P4 = DualQuaternion.as_rational([2, 1/2, 5/2, -3, -3/130, -289/520, -3/104, -103/780])

P1 = DualQuaternion.as_rational()
P2 = DualQuaternion.as_rational([0, 0, 0, 1, 1, 0, 1, 0])
P3 = DualQuaternion.as_rational([1, 2, 0, 0, -2, 1, 0, 0])
P4 = DualQuaternion.as_rational([3, 0, 1, 0, 1, 0, -3, 0])

K = P1.array() + x[0] * P2.array() + x[1] * P3.array() + x[2] * P4.array()
Kdq = DualQuaternion(K)

eqs = [K[0], K[4], Kdq.norm().array()[4]]

#sol = sp.solve(eqs, x, domain=sp.S.Reals)
sol = sp.solve(eqs, x)

K = [sp.Expr(el) for el in K]

K1 = [el.subs({x[0]: sol[0][0], x[1]: sol[0][1], x[2]: sol[0][2]}) for el in K]

K1 = DualQuaternion([el.args[0] for el in K1])
print(K1)
#div = sp.gcd(*[K1[1].args[0], K1[2].args[0], K1[3].args[0], K1[5].args[0], K1[6].args[0], K1[7].args[0]], domain='RR')
#K2 = [el.subs({x[0]: sol[1][0], x[1]: sol[1][1], x[2]: sol[1][2], x[3]: sol[1][3]}) for el in K]

#ii = sp.symbols("ii:8")
#K1factored = [sp.together(element.subs(x[3], 1)) for i, element in enumerate(K1)]
#K1factored = [sp.Expr(K1factored[i].args[0] * ii[i]) for i in range(len(K1factored))]

#result = sp.together(*K1factored)

###################################

K1 = DualQuaternion([0, 4, -1, 5, 0, 2, 8, 0])
K2 = DualQuaternion([0, 2, 1, 1, 0, 1, -2, 0])

P5 = DualQuaternion([lam, 0, 0, 0, 0, 0, 0, 0]) - K1

study_cond_mat = sp.Matrix([[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]])

dq = sp.Matrix([x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]])

study_cond = dq.transpose() @ study_cond_mat @ dq

t1 = DualQuaternion([t[1], 0, 0, 0, 0, 0, 0, 0])
t2 = DualQuaternion([t[2], 0, 0, 0, 0, 0, 0, 0])
t3 = DualQuaternion([t[3], 0, 0, 0, 0, 0, 0, 0])

eqs2 = [sp.Matrix((t1 - K1).array()).transpose() @ study_cond_mat @ sp.Matrix(P2.array()),
        sp.Matrix((t2 - K1).array()).transpose() @ study_cond_mat @ sp.Matrix(P3.array()),
        sp.Matrix((t3 - K1).array()).transpose() @ study_cond_mat @ sp.Matrix(P4.array())]

solst = sp.solve(eqs2, t)
# covert to list
solst = [val for i, (key, val) in enumerate(solst.items())]


def lagrange_polynomial(n, k, x, t):
    # n is the degree of the polynomial
    #n = 3
    # k is the index of the Lagrange polynomial

    L = 1
    for i in range(n + 1):
        if i != k:
            L *= (x - t[i]) / (t[k] - t[i])
    return L

def interpolation_polynomial(P, x):
    # Interpolation polynomial with indeterminate x and parameter values x_i.
    # P... list of Poses
    n = len(P) - 1
    #result = sp.Add(*[P[k] * lagrange_polynomial(n, k, x, list(range(n+1))) for k in range(n+1)])
    result = sp.Matrix([0, 0, 0, 0, 0, 0, 0, 0])
    t = sp.symbols("t:4")
    for i in range(n+1):
        result += P[i] * lagrange_polynomial(n, i, x, t)
    return result

lams = sp.symbols("lams:5")
x = sp.symbols("x")
P = [sp.Matrix(P1.array()), sp.Matrix(lams[1] * P2.array()), sp.Matrix(lams[2] * P3.array()), sp.Matrix(lams[3] * P4.array())]

temp = interpolation_polynomial(P, x)

# I added x^3 already here:
temp = [element.subs(t[0], 0) for element in temp]
temp2 = [element.subs(x, 1/x) for element in temp]
temp3 = [sp.together(element * x**3) for element in temp2]

temp4 = [sp.together(element.subs({t[1]: 1/solst[0], t[2]: 1/solst[1], t[3]: 1/solst[2]}))
         for element in temp3]

eqslambda = [element.subs(x, lam) - lams[-1] * P5.array()[i] for i, element in enumerate(temp4)]

solslambda = sp.solve(eqslambda, lams[1:5])

Q = [element.subs(solslambda) for element in temp4]

Q_as_dq = DualQuaternion(Q)

Q = [element.subs(lam, 0) for element in Q]
t = sp.Symbol("t")
Q = [element.subs(x, t) for element in Q]

Q_poly = [sp.Poly(element, t) for element in Q]

#print(sp.simplify(Q_as_dq.norm()[4]))

c = RationalCurve(Q_poly)

p = Plotter(interactive=False, steps=500)
#p.plot(c, interval=[-10, 10])
f = FactorizationProvider()

f_res = f.factorize_motion_curve(c)

f_res = f.factorize_for_motion_factorization(f_res[0])

rational_dq = DualQuaternion.as_rational([1, 5, 2, 3, 0, 1, 0, 0])

print(f_res)

#p.plot(f_res[0])

matrix = TransfMatrix.from_rpy_xyz([0, 0, 90], [1, -80, 20], units="deg")