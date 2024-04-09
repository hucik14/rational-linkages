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
from rational_linkages.models import bennett_ark24, collisions_free_6r, plane_fold_6r

if __name__ == '__main__':
    m = RationalMechanism.from_saved_file("johannes-interp.pkl")
    #m = collisions_free_6r()
    #m = plane_fold_6r()
    #m = bennett_ark24()
    m.update_segments()

    #m.collision_check(parallel=True, only_links=True)

    c = m.curve()

    mechanism_points = m.points_at_parameter(0, inverted_part=True, only_links=True)
    metric = AffineMetric(c, mechanism_points)

    p = Plotter(interactive=True, arrows_length=0.5, joint_range_lim=2, steps=200)

    bezier_segments = c.split_in_beziers()

    # for bezier_curve in bezier_segments:
    #     p.plot(bezier_curve, interval=(-1, 1), plot_control_points=True)
    #     p.plot(bezier_curve.ball)

    id = 0
    #p.plot(bezier_segments[id], interval=(-1, 1), plot_control_points=True)
    #p.plot(bezier_segments[id].ball)

    p.plot(m)
    p.plot(m.segments[3])

    #p.plot(c, interval='closed')
    p.show()

