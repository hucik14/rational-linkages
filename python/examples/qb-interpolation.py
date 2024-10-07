import numpy as np

from rational_linkages import (Quaternion, PointHomogeneous, RationalBezier, Plotter,
                               RationalMechanism, RationalCurve, DualQuaternion,
                               MotionInterpolation)

q0 = PointHomogeneous([1, 0, 0, 0])
q1 = PointHomogeneous([1, 1, 0, -2])
q2 = PointHomogeneous([1, 2, -1, 0])
q3 = PointHomogeneous([1, -3, 0, 3])
q4 = PointHomogeneous([1, 2, 1, -1])
pts = [q0, q1, q2, q3, q4]

a0 = Quaternion([0, q0.array()[1] / -2, q0.array()[2] / -2, q0.array()[3] / -2])
a1 = Quaternion([0, q1.array()[1] / -2, q1.array()[2] / -2, q1.array()[3] / -2])
a2 = Quaternion([0, q2.array()[1] / -2, q2.array()[2] / -2, q2.array()[3] / -2])
a3 = Quaternion([0, q3.array()[1] / -2, q3.array()[2] / -2, q3.array()[3] / -2])
a4 = Quaternion([0, q4.array()[1] / -2, q4.array()[2] / -2, q4.array()[3] / -2])

d41 = a4 - a1
d21 = a2 - a1
d43 = a4 - a3
d23 = a2 - a3
d10 = a1 - a0
d30 = a3 - a0

d32 = a3 - a2
d14 = a1 - a4

if (d43.inv()*d32*d21.inv()*d14)[0] == -3.0:
    print("Not possible to interpolate")

w0 = Quaternion()
w2 = (-9*d41.inv()*d21 - 3*d43.inv()*d23).inv() * (9*d41.inv()*d10 - d43.inv()*d30) * w0
w4 = (-1*d21.inv()*d41 - 3*d23.inv()*d43).inv() * (3*d21.inv()*d10 + d23.inv()*d30) * w0

u0 = w0
p0 = a0
u1 = (-1/2)*(w0 + 2*w2 + w4)
p1 = (-1/2)*(a0*w0 + 2*a2*w2 + a4*w4)*u1.inv()
u2 = w4
p2 = a4

# cp0 = PointHomogeneous(np.concatenate([np.atleast_1d(x) for x in (u0.array()[0], (p0*u0).array()[1:], (p0*u0).array()[0], u0.array()[1:])]))
# cp1 = PointHomogeneous(np.concatenate([np.atleast_1d(x) for x in (u1.array()[0], (p1*u1).array()[1:], (p1*u1).array()[0], u1.array()[1:])]))
# cp2 = PointHomogeneous(np.concatenate([np.atleast_1d(x) for x in (u2.array()[0], (p2*u2).array()[1:], (p2*u2).array()[0], u2.array()[1:])]))

cp0 = PointHomogeneous(np.concatenate((u0.array(), (p0*u0).array())))
cp1 = PointHomogeneous(np.concatenate((u1.array(), (p1*u1).array())))
cp2 = PointHomogeneous(np.concatenate((u2.array(), (p2*u2).array())))
cp = [cp0, cp1, cp2]

b = RationalBezier([cp2, cp1, cp0])

i = MotionInterpolation.interpolate(pts)
print(i)
c = RationalCurve(b.set_of_polynomials)
print(c)
m = RationalMechanism(i.factorize())

p = Plotter(interactive=True, steps=500)
# p.plot(m, show_tool=True)
p.plot(b, interval='closed')
p.plot(pts)
# p.plot([DualQuaternion(cp0.array()), DualQuaternion(cp1.array()), DualQuaternion(cp2.array())])
p.show()
