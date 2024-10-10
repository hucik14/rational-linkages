import numpy as np
import sympy as sp

from rational_linkages import (
    Plotter,
    RationalCurve,
    RationalBezier,
    RationalMechanism,
    AffineMetric,
    PointHomogeneous,
    DualQuaternion,
    CollisionAnalyser,
)
from rational_linkages.models import bennett_ark24, collisions_free_6r, plane_fold_6r, interp_4poses_6r

from time import time

if __name__ == '__main__':
    m = interp_4poses_6r()
    m = collisions_free_6r()
    m = plane_fold_6r()
    #m = bennett_ark24()

    m.update_segments()
    m._relative_motions = None
    m._metric = None

    start_time = time()
    ca = CollisionAnalyser(m)
    print(f'{time() - start_time:.3f} sec for generating Bezier segments')


    l0 = 'l_01'
    l1 = 't_13'
    orbits0 = ca.get_segment_orbit(l0)
    orbits1 = ca.get_segment_orbit(l1)

    start_time = time()
    res = m._check_given_pair([2, 6])
    print(f'{time() - start_time:.3f} sec for checking collisions in standard way')

    start_time = time()
    ca.check_two_segments(l0, l1)
    print(f'{time() - start_time:.3f} sec for checking collision')

    p = Plotter(interactive=True, arrows_length=0.1, joint_range_lim=2, steps=300)
    p.plot(m)

    for orbit in orbits0:
        p.plot(orbit[1:])
    for orbit in orbits1:
        p.plot(orbit[1:])

    # for orbit in orbits0[0][1:]:
    #     p.plot(orbit)
    # for orbit in orbits1[0][1:]:
    #     p.plot(orbit)

    p.show()

