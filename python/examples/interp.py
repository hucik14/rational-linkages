from rational_linkages import DualQuaternion
from rational_linkages import TransfMatrix
from rational_linkages import Plotter
from rational_linkages import FactorizationProvider
from rational_linkages import RationalMechanism
from rational_linkages import MotionInterpolation


if __name__ == "__main__":

    # p1 = DualQuaternion.as_rational([3135, 31350, 6270, 10450, 4296, -4065, -3948, 13275])
    # p2 = DualQuaternion.as_rational([-2/5, 1/5, -3/2, 4, 1409/5535, 5753/5535, 998/369, 5467/5535])
    # p3 = DualQuaternion.as_rational([2, 1/2, 5/2, -3, -3/130, -289/520, -3/104, -103/780])

    p1 = DualQuaternion.as_rational([0, 0, 0, 1, 1, 0, 1, 0])
    p2 = DualQuaternion.as_rational([1, 2, 0, 0, -2, 1, 0, 0])
    p3 = DualQuaternion.as_rational([3, 0, 1, 0, 1, 0, -3, 0])

    # t1 = TransfMatrix.from_rpy_xyz([0, 90, 0], [1, -2, 2], units="deg")
    # p1 = DualQuaternion.as_rational(t1.matrix2dq())
    # t2 = TransfMatrix.from_rpy_xyz([-45, 90, 0], [0, 1, 1], units="deg")
    # p2 = DualQuaternion.as_rational(t2.matrix2dq())
    # t3 = TransfMatrix.from_rpy_xyz([90, 45, 0], [2, 1, 0], units="deg")
    # p3 = DualQuaternion.as_rational(t3.matrix2dq())

    p0 = DualQuaternion()

    c = MotionInterpolation.interpolate([p1, p2, p3])
    f = c.factorize()
    m = RationalMechanism(f)

    #m.collision_check(parallel=True)

    p = Plotter(interactive=True, steps=1000, arrows_length=0.5)
    p.plot(m, show_tool=True)

    for pose in [p0, p1, p2, p3]:
        p.plot(pose)

    p.show()
