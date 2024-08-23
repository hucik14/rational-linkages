from rational_linkages import (NormalizedLine, MotionFactorization, DualQuaternion,
                               RationalCurve, RationalMechanism, Plotter,
                               TransfMatrix, MotionInterpolation, PointHomogeneous,
                               StaticMechanism)
import numpy as np
import matplotlib.pyplot as plt


# sm = StaticMechanism.from_dh_parameters(theta=[0, 0], d=[0, 0], a=[0.2, 0.3], alpha=[np.deg2rad(180-36.86989898989898), np.deg2rad(90)])
#
# a0 = NormalizedLine.from_direction_and_point([0, 0, -1],
#                                              [0, 0.36, 0.08])
# a1 = NormalizedLine.from_direction_and_point([0.70710678, 0., 0.70710678],
#                                              [0, 0.160, 0])
#
# a0 = NormalizedLine([0, 0, 1, 0, 0, 0])
# a1 = NormalizedLine([0., -0.6, -0.8,  0.,  0.16, -0.12])
#
# fp = MotionFactorization([DualQuaternion.as_rational(a0.line2dq_array()),
#                           DualQuaternion.as_rational(a1.line2dq_array())])
#
# # c2 = fp.factorize()
# c2 = RationalCurve(fp.set_of_polynomials)


t0 = TransfMatrix()
t1 = TransfMatrix.from_vectors(approach_z=[-0.0362862, 0.400074, 0.915764],
                               normal_x=[0.988751, -0.118680, 0.0910266],
                               origin=[0.0376272, 0.0503519, 0.0435525])
t2 = TransfMatrix.from_vectors(approach_z=[-0.0463679, -0.445622, 0.894020],
                               normal_x=[0.985161, 0.127655, 0.114724],
                               origin=[-0.0477573, 0.00438766, -0.0701082])
poses = [t0, t1, t2]

c = MotionInterpolation.interpolate(poses)
m = RationalMechanism(c.factorize())


# p = Plotter(interactive=True, steps=500, arrows_length=0.05, joint_range_lim=0.1)
# #p.ax.init_view(elev=120, azim=0, roll=-150)
# #p.plot(m, show_tool=True)
# p.plot(t0, label='origin')
# p.plot(t1, label='p1')
# p.plot(t2, label='p2')
# p.plot(c, label='Motion path of origin', interval='closed')
# # p.plot(c2, label='Bad', interval='closed')
# # p.plot(a0, label='a0', interval=(-0.1, 0.1))
# # p.plot(a1, label='a1', interval=(-0.1, 0.1))

# p.show()

#########################################################
# Control

m.factorizations[0].t_param_to_joint_angle(1.0)

theta1 = m.inverse_kinematics(t1)
theta2 = m.inverse_kinematics(t2)

theta_corr = theta2 - theta1

pose1 = m.forward_kinematics(theta1).normalize()
pose2 = m.forward_kinematics(theta2).normalize()

total_time = 4.
frequency = 20
num_points = int(total_time * frequency)

traj = m.traj_p2p_joint_space(joint_angle_start=theta1,
                              joint_angle_end=theta2,
                              time_sec=total_time,
                              num_points=num_points,
                              generate_csv=False)

# traj = m.traj_smooth_tool(joint_angle_start=theta1,
#                           joint_angle_end=theta2,
#                           time_sec=total_time,
#                           num_points=num_points,
#                           generate_csv=False)


# p = Plotter(interactive=True, steps=500, arrows_length=0.05, joint_range_lim=0.1)
# #p.ax.init_view(elev=120, azim=0, roll=-150)
# p.plot(m, show_tool=True)
# p.plot(t0, label='origin')
# p.plot(t1, label='p1')
# p.plot(t2, label='p2')
# #p.plot(c, label='Motion path of origin', interval='closed')
#
# p.show()



# plot the trajectory
vel = np.diff(traj, axis=0) * frequency
#vel = np.append(vel, vel[-1])
acc = np.diff(vel, axis=0) * frequency
#acc = np.append(acc, acc[-1])

plt.plot(traj)
plt.plot(vel)
plt.plot(acc)
plt.xlabel('Time-step')
plt.ylabel('Joint pos [rad], vel [rad/s], acc [rad/s^2]')
plt.title('Smooth joint-space trajectory')
plt.legend(['Position [rad]', 'Velocity [rad/s]', 'Acceleration [rad/s^2]'])
plt.grid()
plt.show()
