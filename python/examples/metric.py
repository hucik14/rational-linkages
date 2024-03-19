import numpy as np

from rational_linkages import (
    Plotter,
    RationalCurve,
    RationalBezier,
    RationalMechanism,
    AffineMetric,
    PointHomogeneous,
    DualQuaternion,
)

if __name__ == '__main__':
    m = RationalMechanism.from_saved_file("johannes-interp.pkl")
    #m.collision_check(parallel=True, only_links=True)

    c = m.curve()



    mechanism_points = m.points_at_parameter(0, inverted_part=True, only_links=True)
    metric = AffineMetric(c, mechanism_points)

    b = RationalBezier(c.curve2bezier_control_points(reparametrization=True))

    p = Plotter(interactive=True, arrows_length=0.5, joint_range_lim=2, steps=200)
    p.plot(m, show_tool=False)
    p.plot(c, interval=(-1, 1))
    #p.plot(b, interval=(-1, 1))
    p.show()
