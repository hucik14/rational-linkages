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


if __name__ == '__main__':
    m = interp_4poses_6r()
    #m = collisions_free_6r()
    #m = plane_fold_6r()
    m = bennett_ark24()
    m.update_segments()

    m._relative_motions = None
    m._metric = None

    p = Plotter(interactive=True, arrows_length=0.1, joint_range_lim=2, steps=200)
    p.plot(m)

    ca = CollisionAnalyser(m)

    s = 't_12'
    o0, o1 = ca.get_segment_orbit(s)
    p.plot(ca.segments[s])

    for orbit in o0:
        p.plot(orbit)
    for orbit in o1:
        p.plot(orbit)

    p.show()

