from rational_linkages import (
    DualQuaternion,
    MotionFactorization,
    Plotter,
    RationalMechanism,
    TransfMatrix,
)

if __name__ == '__main__':
    f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                              DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])

    f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                              DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

    tool_matrix = TransfMatrix.from_rpy_xyz([90, 0, 45], [-0.2, 0.5, 0], units='deg')
    tool_dq = DualQuaternion(tool_matrix.matrix2dq())

    m = RationalMechanism([f1, f2], tool=tool_dq)
    #m = RationalMechanism([f1, f2])

    p = Plotter(interactive=True, steps=200, arrows_length=0.2)
    p.plot(m, show_tool=True)
    p.plot(m.get_motion_curve(), label='motion curve', interval='closed', color='red',
           linewidth='0.7', linestyle=':')
    p.show()

