from rational_linkages import (RationalMechanism, DualQuaternion, Plotter,
                               MotionFactorization, PointHomogeneous)
import exudyn as exu
from exudyn.itemInterface import *
from exudyn.utilities import *  # includes graphics and rigid body utilities
import numpy as np


if __name__ == '__main__':
    # define mechanism factorizations (branches) and its axes
    f0 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                              DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])

    f1 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                              DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

    # adjust points on links
    l0_p0 = PointHomogeneous([1, 0, 0, 0])
    l0_p1 = PointHomogeneous([1, -0.16666667, 0, 0])
    l1_p0 = PointHomogeneous([1, 0, 0, 0.2])
    l1_p1 = PointHomogeneous([1, -0.5, 0, 0.2])
    l2_p0 = PointHomogeneous([1, -0.5, 0, 0.1])
    l2_p1 = PointHomogeneous([1, -0.66666667, 0, 0])
    l3_p0 = PointHomogeneous([1, -0.66666667, 0, -0.1])
    l3_p1 = PointHomogeneous([1, -0.16666667, 0, -0.1])

    f0.set_joint_connection_points([l0_p0, l1_p0, l1_p1, l2_p0])
    f1.set_joint_connection_points([l0_p1, l3_p1, l3_p0, l2_p1])

    # create mechanism object, place tool frame in the middle of the last link
    m = RationalMechanism([f0, f1], tool='mid_of_last_link')

    # p = Plotter(interactive=True, steps=200, arrows_length=0.08)
    # p.plot(m, show_tool=True)
    # p.plot(DualQuaternion())
    # p.show()

    ###################################################################################
    ############################# Exudyn part #########################################

    useGraphics = True
    case = 2
    caseText = 'non-redundant constraints'

    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    # %%++++++++++++++++++++++++++++++++++++++++++++++++++++
    # physical parameters
    g = [0, -1, -9.81]  # gravity + disturbance

    L = [0.16666667, 0.5, 0.194365066, 0.5]
    w = 0.06  # width

    bodyDim = [[L[1], w, w], [L[2], w, w], [L[3], w, w]]  # body dimensions

    pMid0 = np.array([-0.25, 0, 0.2])  # center of mass, body0  #TODO global frame
    pMid1 = np.array([-0.58333333, 0, 0.05])  # center of mass, body1
    pMid2 = np.array([-0.41666667, 0, -0.1])  # center of mass, body2
    ref_positions = [pMid0, pMid1, pMid2]

    # ground body
    # graphics data for checkerboard background (not required):
    gGround0 = GraphicsDataCheckerBoard(point=[-0.4, 0, -0.4], normal=[0, 0, 1], size=1)
    # add ground object and background graphics; visualization is optional
    #oGround = mbs.CreateGround(graphicsDataList=[gGround0])  # TODO not working
    oGround = mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=[gGround0])))

    inertias = [InertiaCuboid(density=5000, sideLengths=bodyDim[0]),
                InertiaCuboid(density=5000, sideLengths=bodyDim[1]),
                InertiaCuboid(density=5000, sideLengths=bodyDim[2])]

    # graphics for body # TODO local frame from center of mass?
    graphicsBody0 = GraphicsDataRigidLink(p0=[0.5 * L[1], 0, 0], p1=[-0.5 * L[1], 0, 0],
                                          axis0=[0, 0, 1], axis1=[0, 0, 1],
                                          radius=[0.5 * w, 0.5 * w],
                                          thickness=w, width=[1.2 * w, 1.2 * w],
                                          color=color4red)
    graphicsBody1 = GraphicsDataRigidLink(p0=[0.08333333, 0, 0.05], p1=[-0.08333333, 0, -0.05],
                                          axis0=[0, 0, 1], axis1=[0, 0, 1],
                                          radius=[0.5 * w, 0.5 * w],
                                          thickness=w, width=[1.2 * w, 1.2 * w],
                                          color=color4green)
    graphicsBody2 = GraphicsDataRigidLink(p0=[-0.5 * L[3], 0, 0], p1=[0.5 * L[3], 0, 0],
                                          axis0=[0, 0, 1], axis1=[0, 0, 1],
                                          radius=[0.5 * w, 0.5 * w],
                                          thickness=w, width=[1.2 * w, 1.2 * w],
                                          color=color4steelblue)

    b0 = mbs.CreateRigidBody(inertia=inertias[0],
                             referencePosition=ref_positions[0],
                             initialVelocity=[0, 0, 0],
                             initialAngularVelocity=[0, 0, 0],
                             gravity=g,
                             graphicsDataList=[graphicsBody0])

    b1 = mbs.CreateRigidBody(inertia=inertias[1],
                             referencePosition=ref_positions[1],
                             initialVelocity=[0, 0, 0],
                             initialAngularVelocity=[0, 0, 0],
                             gravity=g,
                             graphicsDataList=[graphicsBody1])

    b2 = mbs.CreateRigidBody(inertia=inertias[2],
                             referencePosition=ref_positions[2],
                             initialVelocity=[0, 0, 0],
                             initialAngularVelocity=[0, 0, 0],
                             gravity=g,
                             graphicsDataList=[graphicsBody2])

    mbs.CreateRevoluteJoint(bodyNumbers=[oGround, b0],
                            position=[0, 0, 0],
                            axis=[0, 0, 1],  # rotation along global z-axis
                            useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)

    mbs.CreateRevoluteJoint(bodyNumbers=[b0, b1],
                            position=[-0.5, 0, 0.2],
                            axis=[0, 0, 1],  # rotation along global z-axis
                            useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)

    mbs.CreateRevoluteJoint(bodyNumbers=[b1, b2],
                            position=[-0.66666667, 0, 0],
                            axis=[0, 0, 1],  # rotation along global z-axis
                            useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)

    mbs.CreateRevoluteJoint(bodyNumbers=[b2, oGround],
                            position=[-0.16666667, 0, -0.1],
                            axis=[0, 0, 1],  # rotation along global z-axis
                            useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)

    mbs.Assemble()

    # some simulation parameters:
    simulationSettings = exu.SimulationSettings()  # takes currently set values or default values
    simulationSettings.timeIntegration.numberOfSteps = 1000
    simulationSettings.timeIntegration.endTime = 5

    # for redundant constraints, the following two settings:
    simulationSettings.linearSolverSettings.ignoreSingularJacobian = True
    simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense  # use EigenSparse for larger systems alternatively

    ### ADDED BY ME
    mbs.ComputeSystemDegreeOfFreedom(verbose=True)
    #mbs.DrawSystemGraph(useItemTypes=True)  # TODO not working
    #mbs.systemData.Info()
    ###

    mbs.SolveDynamic(simulationSettings)

    # visualize results after simulation:
    mbs.SolutionViewer()
