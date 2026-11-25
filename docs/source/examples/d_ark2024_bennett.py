from rational_linkages import RationalCurve, RationalMechanism, Plotter, DualQuaternion, NormalizedLine
import numpy as np


# input poses (not used for calculation, only for plotting)
p0 = DualQuaternion([0, 17, -33, -89, 0, -6, 5, -3])
p1 = DualQuaternion([0, 84, -21, -287, 0, -30, 3, -9])
p2 = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3])

coeffs = np.array([[0, 0, 0],
                   [4440, 39870, 22134],
                   [16428, 9927, -42966],
                   [-37296, -73843, -115878],
                   [0, 0, 0],
                   [-1332, -14586, -7812],
                   [-2664, -1473, 6510],
                   [-1332, -1881, -3906]])

# define a rational curve object
c = RationalCurve.from_coeffs(coeffs)

# define a mechanism object from curve factorization
m = RationalMechanism(c.factorize())

# define a plotter object, set mechanism, backend, number of discrete steps, frame arrows length,
# and base pose - the base pose is used change the reference frame of the mechanism,
# according to the methodology, it equals p2
p = Plotter(mechanism=m, backend='matplotlib', steps=500, arrows_length=0.05, base=p2)

# plot poses
p.plot(DualQuaternion(), label='world (identity)')
p.plot(p0, label='a')
p.plot(p1, label='b')
p.plot(p2, label='c')

# la = NormalizedLine(p2.dq2screw())
# p.plot(la, label='l_a', interval=(-0.1, 0.1), color='orange')

# plot the mechanism
p.show()