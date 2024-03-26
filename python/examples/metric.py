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
    #m = bennett_ark24()
    m = collisions_free_6r()
    #m = plane_fold_6r()

    #m.collision_check(parallel=True, only_links=True)

    c_org = m.curve()

    c = c_org.get_curve_in_pr12()
    c_inv = c.inverse_curve()

    mechanism_points = m.points_at_parameter(0, inverted_part=True, only_links=True)
    metric = AffineMetric(c, mechanism_points)

    p = Plotter(interactive=False, arrows_length=0.5, joint_range_lim=2, steps=200)
    # p.plot(m.curve(), interval='closed')
    # p.show()

    bcp = c.curve2bezier_control_points(reparametrization=True)
    b = RationalBezier(bcp, reparametrization=True)
    bcp_inv = c_inv.curve2bezier_control_points(reparametrization=True)
    b_inv = RationalBezier(bcp_inv, reparametrization=True)

    bezier_curve_segments = [b, b_inv]

    while True:
        new_bezier_curve_segments = []
        split_occurred = False

        for bezier_curve in bezier_curve_segments:
            if bezier_curve.check_for_control_points_at_infinity() or bezier_curve.check_for_negative_weights():
                left, right = bezier_curve.split_de_casteljau()
                new_bezier_curve_segments.append(left)
                new_bezier_curve_segments.append(right)
                split_occurred = True
            else:
                new_bezier_curve_segments.append(bezier_curve)

        bezier_curve_segments = new_bezier_curve_segments
        print(len(bezier_curve_segments))

        if not split_occurred:
            break

    for bezier_curve in bezier_curve_segments:
        if not (bezier_curve.check_for_control_points_at_infinity() or bezier_curve.check_for_negative_weights()):
            p.plot(bezier_curve, interval=(-1, 1), plot_control_points=True)

    #p.plot(c_org, interval='closed')
    p.show()
