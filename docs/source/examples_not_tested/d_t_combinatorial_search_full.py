# THIS CODE IS NOT TESTED
from rational_linkages import (DualQuaternion, MotionFactorization,
                               RationalMechanism, Plotter)

if __name__ == '__main__':
    h1 = DualQuaternion.as_rational([0, 1, 0, 0, 0, 0, 0, 0])
    h2 = DualQuaternion.as_rational([0, 0, 3, 0, 0, 0, 0, 1])
    h3 = DualQuaternion.as_rational([0, 1, 1, 0, 0, 0, 0, -2])

    f1 = MotionFactorization([h1, h2, h3])

    # find factorizations
    factorizations = f1.factorize()

    # create mechanism
    m = RationalMechanism(factorizations)
    m.collision_free_optimization(max_iters=10,
                                  min_joint_segment_length=0.3)

    # plot mechanism
    myplt = Plotter(mechanism=m, show_tool=False, steps=200, arrows_length=0.2,
                    joint_sliders_lim=3.0)
    myplt.show()