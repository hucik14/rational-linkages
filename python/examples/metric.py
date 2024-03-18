import numpy as np

from rational_linkages import (
    FactorizationProvider,
    Plotter,
    RationalCurve,
    RationalMechanism,
    AffineMetric,
    PointHomogeneous,
    DualQuaternion,
)

if __name__ == '__main__':
    coeffs = np.array([[0, 0, 0],
                       [4440, 39870, 22134],
                       [16428, 9927, -42966],
                       [-37296, -73843, -115878],
                       [0, 0, 0],
                       [-1332, -14586, -7812],
                       [-2664, -1473, 6510],
                       [-1332, -1881, -3906]])

    # define a rational curve object
    c = RationalCurve.from_coeffs(coeffs)
    m = RationalMechanism(FactorizationProvider().factorize_motion_curve(c))
    #m.smallest_polyline(update_design=True)
    #m.collision_check(parallel=True, only_links=True)

    # p = Plotter(interactive=True, arrows_length=0.05, joint_range_lim=0.5)
    # p.plot(m, show_tool=True)
    # p.show()

    mechanism_points = m.points_at_parameter(0, inverted_part=True, only_links=True)

    # define an affine metric object
    metric = AffineMetric(c, mechanism_points)
    print("")
    print("Metric matrix:")
    print(metric.matrix)

    # get two poses
    a = DualQuaternion(c.evaluate(-1))
    b = DualQuaternion(c.evaluate(0, inverted_part=True))

    print("")
    print("Distance between two poses via inner product: ????")
    print(metric.distance(a, b))

    print("Distance between two poses via metric matrix:")
    print(metric.dist(a, b))