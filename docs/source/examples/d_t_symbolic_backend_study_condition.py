from rational_linkages import DualQuaternion, set_backend
from sympy import symbols


set_backend("sympy")


p0, p1, p2, p3 = symbols("p0 p1 p2 p3", real=True)

# Pure-rotation dual quaternion: dual part is zero, so p*d = 0 trivially.
dq_rot = DualQuaternion([p0, p1, p2, p3, 0, 0, 0, 0])
print(dq_rot.is_on_study_quadric())  # True

# Back-project an arbitrary dual quaternion onto the Study quadric.
p0, p1, p2, p3, d0, d1, d2, d3 = symbols("p0 p1 p2 p3 d0 d1 d2 d3", real=True)
dq_arb = DualQuaternion([p0, p1, p2, p3, d0, d1, d2, d3])
dq_proj = dq_arb.back_projection()
print(dq_proj.is_on_study_quadric())  # True


