from rational_linkages import (Plotter,
                               MotionInterpolation,
                               TransfMatrix,
                               RationalMechanism,
                               )

if __name__ == "__main__":
    # for simplification, p0 is the identity matrix
    p0 = TransfMatrix()
    p1 = TransfMatrix.from_rpy_xyz([-40, 0, 0], [0.030, 0.1, 0.02], unit='deg')
    p2 = TransfMatrix.from_rpy_xyz([0, 90, 0], [0.09, -0.05, -0.02], unit='deg')

    poses = [p0, p1, p2]  # list of the transformation matrices

    curve = MotionInterpolation.interpolate(poses)
    mechanism = RationalMechanism(curve.factorize())

    #mechanism.collision_free_optimization(max_iters=10)
    comb_links = [(0, 0, 0, -1)]
    comb_joints = [(-1, 1, 1, -1, 1, -1, -1, 1)]
    mechanism.collision_free_optimization(min_joint_segment_length=0.003,
                                          combinations_links=comb_links,
                                          combinations_joints=comb_joints)

    design_parameters = mechanism.get_design(scale=1000, unit='deg')

    myplt = Plotter(mechanism=mechanism, arrows_length=0.02, steps=300)
    for i, pose in enumerate(poses):
        myplt.plot(pose, label=f'p{i}')
    myplt.show()
