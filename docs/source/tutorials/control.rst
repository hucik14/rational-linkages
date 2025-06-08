Direct (Forward) Kinematics
===========================

Direct/forward kinematics is the process of calculating the position and orientation of the end-effector (tool)
of a robot given its joint angles. In the context of a rational mechanism,
this involves computing the tool's pose based on the configuration of the mechanism's joints. As only 1 degree
of freedom is present in a rational mechanism, the forward kinematics is straightforward. However, the
parameterization has to be handled carefully. For more details, see the section :ref:`joint-angle-to-t` and related
paper :footcite:t:`Huczala2025kinematics`. The implementation is in
:meth:`.RationalMechanism.forward_kinematics()` moethod or it's alias :meth:`.RationalMechanism.direct_kinematics()`.

.. testcode:: [direct_kinematics_example]

    from rational_linkages.models import bennett_ark24
    from rational_linkages import TransfMatrix

    m = bennett_ark24()

    theta = 2.3  # rad
    pose_as_dq = m.forward_kinematics(theta)
    pose_as_matrix = TransfMatrix(pose_as_dq.dq2matrix())

.. testcleanup :: [direct_kinematics_example]

    del bennett_ark24, TransfMatrix, m, theta, pose_as_dq, pose_as_matrix



Inverse Kinematics (Numerical)
==============================

Algebraically, the inverse kinematics problem may be tricky due to the rational nature of the mechanism. The given
pose has to lie on the motion curve that parameterizes the mechanism (and therefore on Study quadric to be a proper
rigid body displacement). If the input suffers from numerical errors,
the solution may not be found. Therefore, the method :meth:`.RationalMechanism.inverse_kinematics()`
implements the numerical inverse kinematics problem using the Newton-Raphson method,
see :footcite:t:`Huczala2025kinematics` for details. As an optimization problem, the given pose can lie outside the
motion curve, and the nearest point on the curve will be returned.

The implemented method has an argument ``robust`` that, when set to ``True``, will use more initial guesses if
the first guess does not converge.

.. testcode:: [inverse_kinematics_example]

    from rational_linkages.models import bennett_ark24
    from rational_linkages import TransfMatrix

    m = bennett_ark24()

    theta = 2.3  # rad
    pose_as_dq = m.forward_kinematics(theta)
    theta_in_rad = m.inverse_kinematics(pose_as_dq, robust=True)

.. testcleanup :: [inverse_kinematics_example]

    del bennett_ark24, TransfMatrix, m, theta, pose_as_dq, theta_in_rad


Velocity Motion Planning
========================

Velocity motion planning is the process of generating a joint-space trajectory for the driven (motorized) joint
of a rational mechanism. There are two main methods for generating such trajectories:
:meth:`.RationalMechanism.traj_p2p_joint_space()` that generates a point-to-point trajectory in joint space using
quintic polynomial interpolation (see :footcite:p:`Lynch2017`). It implements various arguments, see the docstring
documentation and example of the method. The usage is also demonstrated in the jupyter notebook `trajectory_planning`
tutorial. The output joint coordinates can be generated as CSV file for further processing to your control software.

The method :meth:`.RationalMechanism.traj_smooth_tool()` delivers tool-space trajectory that provides smooth equidistant
travel of the tool along the path (the tool velocity is approximately constant). The method uses arc-length
reparameterization of the path. See more details in :footcite:t:`Huczala2025kinematics` and the jupyter notebook
`trajectory_planning` tutorial. Again, the output can be generated as CSV file.