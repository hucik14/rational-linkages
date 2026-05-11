from rational_linkages import TransfMatrix, PointHomogeneous, Plotter
from rational_linkages.StaticMechanism import SnappingMechanism

p0 = TransfMatrix()
p1 = TransfMatrix.from_rpy_xyz([15, 0, -5], [0.15, -0.25, 0.05], unit='deg')

a2 = PointHomogeneous([1, -0.2, 0, 0])
a3 = PointHomogeneous([1, 0.2, 0, 0])
b2 = PointHomogeneous([1, -0.2, 0, 0.1])
b3 = PointHomogeneous([1, 0.2, 0.1, 0.1])

m = SnappingMechanism(p1, [a2, b2, a3, b3])

m.factorizations[0].set_joint_connection_points_by_parameters([[0., 0.001],
                                                               [0.001, 0.],
                                                               [0., 0.001],
                                                               [0.001, 0.]])

m.get_design(unit='deg', scale=150)

p = Plotter(arrows_length=0.1, backend='matplotlib')
p.plot(p0, label='origin')
p.plot(p1, label='pose')
p.plot_line_segments_between_points(m.points_discrete_poses[0] + [m.points_discrete_poses[0][0]], color='red')
p.plot_line_segments_between_points(m.points_discrete_poses[1] + [m.points_discrete_poses[1][0]], color='blue')

p.plot(m.screws[0], label='axis0', interval=(-0.1, 0.1))
p.plot(m.screws[1], label='axis1', interval=(-0.1, 0.1))
p.plot(m.screws[2], label='axis2', interval=(-0.1, 0.1))
p.plot(m.screws[3], label='axis3', interval=(-0.1, 0.1))

p.show()