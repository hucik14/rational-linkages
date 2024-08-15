from rational_linkages import (NormalizedLine, MotionFactorization, DualQuaternion,
                               RationalCurve, RationalMechanism, Plotter,
                               TransfMatrix, MotionInterpolation, PointHomogeneous)
import numpy as np
import matplotlib.pyplot as plt

# a0 = NormalizedLine.from_direction_and_point([0, 0, -1],
#                                              [0, 0.36, 0.08])
# a1 = NormalizedLine.from_direction_and_point([0.70710678, 0., 0.70710678],
#                                              [0, 0.160, 0])
#
# fp = MotionFactorization([DualQuaternion(a0.line2dq_array()),
#                           DualQuaternion(a1.line2dq_array())])
# c2 = fp.factorize()

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

#########################################################
# Control

theta0 = m.inverse_kinematics(t1)
theta1 = m.inverse_kinematics(t2)

total_time = 3.
num_points = 100

traj = m.traj_p2p_joint_space(joint_angle_start=theta0,
                              joint_angle_end=theta1,
                              time_sec=total_time,
                              num_points=num_points)

# traj = m.traj_smooth_tool(joint_angle_start=theta0,
#                           joint_angle_end=theta1,
#                           time_sec=total_time,
#                           num_points=num_points)




# p = Plotter(interactive=True, steps=500, arrows_length=0.05, joint_range_lim=0.1)
# p.plot(m, show_tool=True)
# p.plot(poses)
# p.show()
#
# p.animate_angles(traj, sleep_time=0.2)

# plot the trajectory
vel = np.diff(traj, axis=0) / (total_time / (num_points - 1))
acc = np.diff(vel, axis=0) / (total_time / (num_points - 1))

plt.plot(traj)
plt.plot(vel)
plt.plot(acc)
plt.xlabel('Time')
plt.ylabel('Joint Angle')
plt.title('Smooth joint-space trajectory')
plt.legend(['Position [rad]', 'Velocity [rad/s]', 'Acceleration [rad/s^2]'])
plt.grid()
plt.show()
