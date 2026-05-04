from rational_linkages import RationalCurve, RationalMechanism, FactorizationProvider, Plotter
import numpy as np

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

# factorize the curve
factors = c.factorize()

# define a mechanism object
m = RationalMechanism(factors)

# define a plotter object, set interactive mode and number of discrete steps
# to plot the curve
p = Plotter(mechanism=m, steps=500, arrows_length=0.05)

# plot the mechanism
p.show()