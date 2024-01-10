from rational_linkages import RationalMechanism
from rational_linkages import FactorizationProvider
from rational_linkages import DualQuaternion
from rational_linkages import NormalizedLine
from rational_linkages import RationalCurve
from rational_linkages import Plotter

import numpy as np


if __name__ == '__main__':
    l1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0.5, 0, 0])
    l2 = NormalizedLine.from_direction_and_point([0, -0.6, 0.4], [0, 0, 0.04])

    l1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0.5, 0, 0])
    l2 = NormalizedLine.from_direction_and_point([0, -0.6, 0.4], [0, 0, 0.04])


    h1 = l1.line2dq_array()
    h2 = l2.line2dq_array()
    h2[0] = 0.2
    h1 = DualQuaternion.as_rational(h1, is_rotation=True)
    h2 = DualQuaternion.as_rational(h2, is_rotation=True)

    coeffs = np.array([[0, 0, 0],
                       [4440, 39870, 22134],
                       [16428, 9927, -42966],
                       [-37296,-73843,-115878],
                       [0, 0, 0],
                       [-1332, -14586, -7812],
                       [-2664, -1473, 6510],
                       [-1332, -1881, -3906]])

    # define a rational curve object
    c = RationalCurve.from_coeffs(coeffs)

    # factorize the curve
    factors = FactorizationProvider().factorize_motion_curve(c)

    # define a mechanism object
    m = RationalMechanism(factors)

    #m = RationalMechanism.from_saved_file('bennett')

    p = Plotter(interactive=True, steps=500)
    p.plot(m, show_tool=True)
    p.show()

