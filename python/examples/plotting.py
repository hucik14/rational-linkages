from rational_linkages import (
    DualQuaternion,
    NormalizedLine,
    Plotter,
    PointHomogeneous,
    TransfMatrix,
)
from rational_linkages.models import bennett_ark24

if __name__ == '__main__':
    # create plotter object, arg steps says how many descrete steps will be used for
    # plotting curves
    myplt = Plotter()

    # create two DualQuaternion objects
    identity = DualQuaternion()
    pose1 = DualQuaternion([0, 0, 1, 0, 0, -0.5, 1, 0])
    pose2 = TransfMatrix.from_rpy_xyz([0, -90, 0], [0, 0, 0.5], unit='deg')

    # create a point with homogeneous coordinates w = 1, x = 2, y = -3, z = 1.5
    point = PointHomogeneous([1, 2, -3, 1.5])

    # create a normalized line from direction vector and the previously specified point
    line = NormalizedLine.from_direction_and_point([0, 0, 1], point.normalized_in_3d())

    # plot the objects
    # 1-line command
    myplt.plot(identity, label='base')
    myplt.plot(point, label='pt')
    myplt.plot(line, label='l1')
    # or for cycle
    for i, obj in enumerate([pose1, pose2]):
        myplt.plot(obj, label='p{}'.format(i + 1))

    myplt.show()

    # load the mechanism
    m = bennett_ark24()

    myplt2 = Plotter(interactive=True, steps=500, arrows_length=0.05)

    point = PointHomogeneous([1, 0.5, -0.75, 0.25])
    myplt2.plot(point, label='pt')
    myplt2.plot(m, show_tool=True)
    myplt2.show()

