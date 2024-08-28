from rational_linkages import (RationalMechanism, Plotter,
                               TransfMatrix, MotionInterpolation,
                               PointHomogeneous)
import numpy as np
import matplotlib.pyplot as plt


# define poses as input for Bennett synthesis
p0 = TransfMatrix()
p1 = TransfMatrix.from_vectors(approach_z=[-0.0362862, 0.400074, 0.915764],
                               normal_x=[0.988751, -0.118680, 0.0910266],
                               origin=[0.033635718, 0.09436004, 0.03428654])
p2 = TransfMatrix.from_vectors(approach_z=[-0.0463679, -0.445622, 0.894020],
                               normal_x=[0.985161, 0.127655, 0.114724],
                               origin=[-0.052857769, -0.04463076, -0.081766])

poses = [p0, p1, p2]
print('Poses (using European convention; projective coordinates are on'
      'first line and translation vector is in the left column):')
for pose in poses:
    print('Pose p' + str(poses.index(pose)) + ' as matrix:')
    print(pose)
    print('Pose p' + str(poses.index(pose)) + ' as dual quaternion:')
    print(pose.matrix2dq())

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

# inverse kinematics
theta1 = m.inverse_kinematics(p1)
theta2 = m.inverse_kinematics(p2)
print('Joint angle for pose 1 in rad:', theta1)
print('Joint angle for pose 2 in rad:', theta2)
print('')  # empty line

# direct (forward) kinematics
pose1_as_dq = m.forward_kinematics(theta1)
pose2_as_dq = m.forward_kinematics(theta2)

# check if the poses are the same
pose1_as_matrix = pose1_as_dq.dq2matrix()
pose2_as_matrix = pose2_as_dq.dq2matrix()

error_pose1 = np.linalg.norm(pose1_as_matrix - p1.array())
error_pose2 = np.linalg.norm(pose2_as_matrix - p2.array())
print('Error in direct kinematics for pose 1:', error_pose1)
print('Error in direct kinematics for pose 2:', error_pose2)

# motion planning
total_time_of_motion = 4.  # seconds
frequency = 20  # Hz
num_points = int(total_time_of_motion * frequency)

# visualize the mechanism and the poses
p = Plotter(interactive=True, steps=500, arrows_length=0.05, joint_range_lim=0.1)
# p.plot(m, show_tool=True)
p.plot(m.curve(), interval='closed')
p.plot(p0, label='p0 (origin)')
p.plot(p1, label='p1')
p.plot(p2, label='p2')
p.show()

# joint-space point-to-point trajectory
pos, vel, acc = m.traj_p2p_joint_space(joint_angle_start=theta1,
                                       joint_angle_end=theta2,
                                       time_sec=total_time_of_motion,
                                       num_points=num_points,
                                       generate_csv=False)

# animate the motion
#p.animate_angles(pos, sleep_time=0.2)

plt.figure()
plt.plot(pos)
plt.plot(vel)
plt.plot(acc)
plt.xlabel('Time-step')
plt.ylabel('Joint pos [rad], vel [rad/s], acc [rad/s^2]')
plt.legend(['Pos (origin)', 'Vel (origin)', 'Acc (origin)'])
plt.title('Smooth joint-space trajectory')
plt.grid()
plt.show()

pos, vel, acc = m.traj_smooth_tool(joint_angle_start=theta1,
                                   joint_angle_end=theta2,
                                   time_sec=total_time_of_motion,
                                   num_points=num_points,
                                   generate_csv=False)

# new tool is translated along the y-axis by -170 mm
t_new_tool = PointHomogeneous([1, 0, -0.17, 0])

pos2, vel2, acc2 = m.traj_smooth_tool(joint_angle_start=theta1,
                                      joint_angle_end=theta2,
                                      time_sec=total_time_of_motion,
                                      point_of_interest=t_new_tool,
                                      num_points=num_points,
                                      generate_csv=False)

plt.figure()
plt.plot(pos, 'C0')
plt.plot(vel, 'C1')
plt.plot(acc, 'C2')
plt.plot(pos2, 'C3', linestyle=':')
plt.plot(vel2, 'C5', linestyle=':')
plt.plot(acc2, 'C4', linestyle=':')
plt.xlabel('Time-step')
plt.ylabel('Joint pos [rad], vel [rad/s], acc [rad/s^2]')
plt.title('Smooth tool motion trajectory')
plt.legend(['Pos (origin)', 'Vel (origin)', 'Acc (origin)',
            'Pos (tool on side)', 'Vel (tool on side)', 'Acc (tool on side)'])
plt.grid()
plt.show()

# animate the motion
#p.animate_angles(pos, sleep_time=0.2)
