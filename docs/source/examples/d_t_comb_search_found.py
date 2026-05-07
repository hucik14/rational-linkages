from rational_linkages import (DualQuaternion, MotionFactorization,
                                   RationalMechanism, Plotter)


h1 = DualQuaternion.as_rational([0, 1, 0, 0, 0, 0, 0, 0])
h2 = DualQuaternion.as_rational([0, 0, 3, 0, 0, 0, 0, 1])
h3 = DualQuaternion.as_rational([0, 1, 1, 0, 0, 0, 0, -2])

f1 = MotionFactorization([h1, h2, h3])

# find factorizations
factorizations = f1.factorize()

# create mechanism
m = RationalMechanism(factorizations)
m.collision_free_optimization(max_iters=10,
                              min_joint_segment_length=0.3,
                              start_iteration=4,
                              combinations_links=[(0, 0, 0, 1, 1, 0)],
                              combinations_joints=[(-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1)])

# plot mechanism
myplt = Plotter(mechanism=m, show_tool=False, steps=200, arrows_length=0.2, joint_sliders_lim=3.0)
myplt.show()