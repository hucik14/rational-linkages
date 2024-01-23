from rational_linkages import (NormalizedLine, FactorizationProvider, RationalMechanism,
                               Plotter, TransfMatrix, RationalCurve, DualQuaternion,
                               PointHomogeneous)
from rational_linkages.models import bennett_ark24 as bennett

import numpy as np


if __name__ == '__main__':
    # coeffs = np.array([[0, 0, 0],
    #                    [4440, 39870, 22134],
    #                    [16428, 9927, -42966],
    #                    [-37296,-73843,-115878],
    #                    [0, 0, 0],
    #                    [-1332, -14586, -7812],
    #                    [-2664, -1473, 6510],
    #                    [-1332, -1881, -3906]])
    #
    # # define a rational curve object
    # c = RationalCurve.from_coeffs(coeffs)
    #
    # # factorize the curve
    # factors = FactorizationProvider().factorize_motion_curve(c)
    # factors[0].set_joint_connection_points(
    #     [PointHomogeneous([1, -0.09766832, 0.04903487, -0.43889732]),
    #      PointHomogeneous([1, -0.00154614, -0.05103096, -0.34090308]),
    #      PointHomogeneous([1, -0.06544225, -0.19215265, -0.05421835]),
    #      PointHomogeneous([1, -4.53371421e-03, -1.93493200e-01, 3.07273305e-04])])
    # factors[1].set_joint_connection_points(
    #     [PointHomogeneous([1, -0.12325908, -0.05714298, 0.21982814]),
    #      PointHomogeneous([1, -0.03221804, -0.01370668, 0.18784829]),
    #      PointHomogeneous([1, -0.06937755, 0.22790725, 0.07179236]),
    #      PointHomogeneous([1, 0.00681783, 0.2248944, -0.00213832])])
    # # define a mechanism object
    # m = RationalMechanism(factors)

    m = bennett()

    p = Plotter(interactive=True, steps=500, arrows_length=0.05)
    p.plot(m, show_tool=True)

    p0 = TransfMatrix.from_rpy_xyz([-90, 0, 0], [0.15, 0, 0], units='deg')
    p.plot(p0)

    p.show()

