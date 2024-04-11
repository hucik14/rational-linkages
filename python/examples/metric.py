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
    m = plane_fold_6r()
    #m = bennett_ark24()
    m.update_segments()

    #m.collision_check(parallel=True, only_links=True)

    c = m.curve()

    mechanism_points = m.points_at_parameter(0, inverted_part=True, only_links=False)
    metric = AffineMetric(c, mechanism_points)

    p = Plotter(interactive=True, arrows_length=0.1, joint_range_lim=2, steps=200)

    bezier_segments = c.split_in_beziers(metric=metric, min_splits=20)

    p.plot(m)

    # p.plot(m.segments[4])
    p.plot(mechanism_points[3])
    p.plot(mechanism_points[2])

    for segment in bezier_segments:
        mechanism_points[3].get_point_orbit(
            segment.ball.center,
            segment.ball.radius,
            metric)
        p.plot(mechanism_points[3].orbit)

    for segment in bezier_segments:
        mechanism_points[2].get_point_orbit(
            segment.ball.center,
            segment.ball.radius,
            metric)
        p.plot(mechanism_points[2].orbit)

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

