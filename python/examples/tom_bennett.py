from rational_linkages import (RationalMechanism,
                               Plotter,
                               TransfMatrix,
                               MotionInterpolation)
import numpy as np
import matplotlib.pyplot as plt


# define poses as input for Bennett synthesis
t0 = TransfMatrix()
t1 = TransfMatrix.from_vectors(approach_z=[-0.0362862, 0.400074, 0.915764],
                               normal_x=[0.988751, -0.118680, 0.0910266],
                               origin=[0.0376272, 0.0503519, 0.0435525])
t2 = TransfMatrix.from_vectors(approach_z=[-0.0463679, -0.445622, 0.894020],
                               normal_x=[0.985161, 0.127655, 0.114724],
                               origin=[-0.0477573, 0.00438766, -0.0701082])
poses = [t0, t1, t2]
print('Poses (using European convention; projective coordinates are on'
      'first line and translation vector is in the left column):')
for pose in poses:
    print(pose)
print('')  # empty line

# construct C(t) from poses
c = MotionInterpolation.interpolate(poses)

# factorize C(t) and obtain mechanism
m = RationalMechanism(c.factorize())
print('C(t):')
print(m.symbolic)
print('')  # empty line
print('DH parameters of the mechanism (length scaled by 1000 to milimeters):')
m.get_design(unit='deg', scale=1000)
print('')  # empty line

# visualize the mechanism and the poses
p = Plotter(interactive=True, steps=500, arrows_length=0.05, joint_range_lim=0.1)
p.plot(m, show_tool=True)
p.plot(t0, label='origin')
p.plot(t1, label='p1')
p.plot(t2, label='p2')
p.show()

# inverse kinematics
theta1 = m.inverse_kinematics(t1)
theta2 = m.inverse_kinematics(t2)
print('Joint angle for pose 1 in rad:', theta1)
print('Joint angle for pose 2 in rad:', theta2)
print('')  # empty line

# direct (forward) kinematics
pose1_as_dq = m.forward_kinematics(theta1)
pose2_as_dq = m.forward_kinematics(theta2)

# check if the poses are the same
pose1_dk = pose1_as_dq.dq2matrix()
pose2_dk = pose2_as_dq.dq2matrix()

error_pose1 = np.linalg.norm(pose1_dk - t1.array())
error_pose2 = np.linalg.norm(pose2_dk - t2.array())
print('Error in direct kinematics for pose 1:', error_pose1)
print('Error in direct kinematics for pose 2:', error_pose2)

# motion planning
total_time_of_motion = 4.  # seconds
frequency = 20  # Hz
num_points = int(total_time_of_motion * frequency)

# joint-space point-to-point trajectory
pos, vel, acc = m.traj_p2p_joint_space(joint_angle_start=theta1,
                                       joint_angle_end=theta2,
                                       time_sec=total_time_of_motion,
                                       num_points=num_points,
                                       generate_csv=False)

plt.figure()
plt.plot(pos)
plt.plot(vel)
plt.plot(acc)
plt.xlabel('Time-step')
plt.ylabel('Joint pos [rad], vel [rad/s], acc [rad/s^2]')
plt.title('Smooth joint-space trajectory')
plt.legend(['Position [rad]', 'Velocity [rad/s]', 'Acceleration [rad/s^2]'])
plt.grid()
plt.show()

pos, vel, acc = m.traj_smooth_tool(joint_angle_start=theta1,
                                   joint_angle_end=theta2,
                                   time_sec=total_time_of_motion,
                                   num_points=num_points,
                                   generate_csv=False)

plt.figure()
plt.plot(pos)
plt.plot(vel)
plt.plot(acc)
plt.xlabel('Time-step')
plt.ylabel('Joint pos [rad], vel [rad/s], acc [rad/s^2]')
plt.title('Smooth tool motion trajectory')
plt.legend(['Position [rad]', 'Velocity [rad/s]', 'Acceleration [rad/s^2]'])
plt.grid()
plt.show()
