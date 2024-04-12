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
)
from rational_linkages.models import bennett_ark24, collisions_free_6r, plane_fold_6r

if __name__ == '__main__':
    m = RationalMechanism.from_saved_file("johannes-interp.pkl")
    #m = collisions_free_6r()
    # m = plane_fold_6r()
    #m = bennett_ark24()
    m.update_segments()

    #m.collision_check(parallel=True, only_links=True)

    c = m.curve()
    t = sp.symbols('t')

    mechanism_points = m.points_at_parameter(0, inverted_part=True, only_links=False)
    metric = AffineMetric(c, mechanism_points)

    p = Plotter(interactive=True, arrows_length=0.1, joint_range_lim=2, steps=200)

    # bezier_segments = c.split_in_beziers(metric=metric, min_splits=20)
    lower_c = m.factorizations[0].get_symbolic_factors()

    mflc = RationalCurve([sp.Poly(pol, t) for pol in (lower_c[0] * lower_c[1])])
    bezier_segments = mflc.split_in_beziers(metric=metric, min_splits=30)

    p.plot(m)

    s = 4
    p0 = 3
    p1 = 4
    p.plot(m.segments[s])
    p.plot(mechanism_points[p0])
    p.plot(mechanism_points[p1])

    p.plot(bezier_segments[0], plot_control_points=True)
    p.plot(bezier_segments[1], plot_control_points=True)

    for segment in bezier_segments:
        mechanism_points[p0].get_point_orbit(
            segment.ball.center,
            segment.ball.radius,
            metric)
        p.plot(mechanism_points[p0].orbit)

    for segment in bezier_segments:
        mechanism_points[p1].get_point_orbit(
            segment.ball.center,
            segment.ball.radius,
            metric)
        p.plot(mechanism_points[p1].orbit)

    # segment = bezier_segments[0]
    # mechanism_points[2].get_point_orbit(
    #     segment.ball.center,
    #     segment.ball.radius,
    #     metric)
    # p.plot(mechanism_points[2].orbit)
    #
    # mechanism_points[3].get_point_orbit(
    #     segment.ball.center,
    #     segment.ball.radius,
    #     metric)
    # p.plot(mechanism_points[3].orbit)
    # p.plot(segment, plot_control_points=True)
    p.show()

