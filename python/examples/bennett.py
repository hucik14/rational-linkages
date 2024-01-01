from RationalMechanism import RationalMechanism
from MotionFactorization import MotionFactorization
from FactorizationProvider import FactorizationProvider
from DualQuaternion import DualQuaternion
from NormalizedLine import NormalizedLine
from RationalCurve import RationalCurve
from Plotter import Plotter

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

    c = RationalCurve.from_coeffs(coeffs)


    #h1 = DualQuaternion.as_rational([0, 0, 0, 1, 0, 0, 0.5, 0], is_rotation=True)
    #h2 = DualQuaternion.as_rational([0.2, 0, -0.6, 0.8, 0, 0, -0.8, -0.6], is_rotation=True)

    f = MotionFactorization([h1, h2])
    fs = FactorizationProvider().factorize_for_motion_factorization(f)
    fs = FactorizationProvider().factorize_motion_curve(c)
    m = RationalMechanism(fs)

    p = Plotter(interactive=True, steps=500)
    p.plot(m, show_tool=True)
    p.show()

"""
m.factorizations[0].linkage[0].points_params
Out[6]: [-0.3354329386075423, -0.16746451667086637]
m.factorizations[0].linkage[1].points_params
Out[7]: [0.08112874779541412, 0.0005039052658095677]
m.factorizations[1].linkage[0].points_params
Out[8]: [0.20878474846728778, 0.07570207570207588]
m.factorizations[1].linkage[1].points_params
Out[9]: [-0.10699588477366317, 0.0005039052658095677]
"""

