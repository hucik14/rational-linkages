
from rational_linkages import (RationalMechanism, DualQuaternion, Plotter,
                               MotionFactorization, PointHomogeneous, ExudynAnalysis)

from rational_linkages.models import collisions_free_6r
import exudyn as exu
from exudyn.itemInterface import *
from exudyn.utilities import *  # includes graphics and rigid body utilities
import numpy as np


if __name__ == '__main__':
    m = collisions_free_6r()

    if False:
        p = Plotter(interactive=True, steps=200, arrows_length=0.2)
        p.plot(m, show_tool=True)
        p.plot(DualQuaternion())
        p.show()

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

    links_pts, links_lengths, body_dim, links_masses_pts, joint_axes, rel_links_pts \
        = ExudynAnalysis().get_exudyn_params(m)
    L = links_lengths
    w = 0.06  # width of link

    # ground body
    # graphics data for checkerboard background (not required):
    gGround0 = GraphicsDataCheckerBoard(point=[-0.4, 0, -0.4], normal=[0, 0, 1], size=1)
    # add ground object and background graphics; visualization is optional
    # oGround = mbs.CreateGround(graphicsDataList=[gGround0])
    oGround = mbs.AddObject \
        (ObjectGround(visualization=VObjectGround(graphicsData=[gGround0])))

    inertias = [InertiaCuboid(density=5000, sideLengths=body_dim[0]),
                InertiaCuboid(density=5000, sideLengths=body_dim[1]),
                InertiaCuboid(density=5000, sideLengths=body_dim[2]),
                InertiaCuboid(density=5000, sideLengths=body_dim[3]),
                InertiaCuboid(density=5000, sideLengths=body_dim[4]),
                InertiaCuboid(density=5000, sideLengths=body_dim[5])]

    # graphics for body # TODO local frame from center of mass?
    graphicsBody0 = GraphicsDataRigidLink(p0=rel_links_pts[1][0], p1=rel_links_pts[1][1],
                                          axis0=joint_axes[0], axis1=joint_axes[1],
                                          radius=[0.5 * w, 0.5 * w],
                                          thickness=w, width=[1.2 * w, 1.2 * w],
                                          color=color4red)
    graphicsBody1 = GraphicsDataRigidLink(p0=rel_links_pts[2][0], p1=rel_links_pts[2][1],
                                          axis0=joint_axes[1], axis1=joint_axes[2],
                                          radius=[0.5 * w, 0.5 * w],
                                          thickness=w, width=[1.2 * w, 1.2 * w],
                                          color=color4green)
    graphicsBody2 = GraphicsDataRigidLink(p0=rel_links_pts[3][0], p1=rel_links_pts[3][1],
                                          axis0=joint_axes[2], axis1=joint_axes[3],
                                          radius=[0.5 * w, 0.5 * w],
                                          thickness=w, width=[1.2 * w, 1.2 * w],
                                          color=color4steelblue)
    graphicsBody3 = GraphicsDataRigidLink(p0=rel_links_pts[4][0], p1=rel_links_pts[4][1],
                                            axis0=joint_axes[3], axis1=joint_axes[4],
                                            radius=[0.5 * w, 0.5 * w],
                                            thickness=w, width=[1.2 * w, 1.2 * w],
                                            color=color4steelblue)
    graphicsBody4 = GraphicsDataRigidLink(p0=rel_links_pts[5][0], p1=rel_links_pts[5][1],
                                            axis0=joint_axes[4], axis1=joint_axes[5],
                                            radius=[0.5 * w, 0.5 * w],
                                            thickness=w, width=[1.2 * w, 1.2 * w],
                                            color=color4steelblue)

    b1 = mbs.CreateRigidBody(inertia=inertias[1],
                             # initialAngularVelocity=[5,6,7],
                             referencePosition=links_masses_pts[1],
                             gravity=g,
                             graphicsDataList=[graphicsBody0])

    b2 = mbs.CreateRigidBody(inertia=inertias[2],
                             referencePosition=links_masses_pts[2],
                             gravity=g,
                             graphicsDataList=[graphicsBody1])

    b3 = mbs.CreateRigidBody(inertia=inertias[3],
                             referencePosition=links_masses_pts[3],
                             gravity=g,
                             graphicsDataList=[graphicsBody2])

    b4 = mbs.CreateRigidBody(inertia=inertias[4],
                                referencePosition=links_masses_pts[4],
                                gravity=g,
                                graphicsDataList=[graphicsBody3])

    b5 = mbs.CreateRigidBody(inertia=inertias[5],
                                referencePosition=links_masses_pts[5],
                                gravity=g,
                                graphicsDataList=[graphicsBody4])


    mbs.CreateRevoluteJoint(bodyNumbers=[oGround, b1],
                            position=links_pts[0][1],
                            axis=joint_axes[0],  # rotation along global z-axis
                            useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)

    mbs.CreateRevoluteJoint(bodyNumbers=[b1, b2],
                            position=links_pts[1][1],
                            axis=joint_axes[1],  # rotation along global z-axis
                            useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)

    mbs.CreateRevoluteJoint(bodyNumbers=[b2, b3],
                            position=links_pts[2][1],
                            axis=joint_axes[2],  # rotation along global z-axis
                            # axis=[0, 0, 1],
                            useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)

    mbs.CreateRevoluteJoint(bodyNumbers=[b3, b4],
                            position=links_pts[3][1],
                            axis=joint_axes[3],  # rotation along global z-axis
                            # axis=[0, 0, 1],
                            useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)

    mbs.CreateRevoluteJoint(bodyNumbers=[b4, b5],
                            position=links_pts[4][1],
                            axis=joint_axes[4],  # rotation along global z-axis
                            # axis=[0, 0, 1],
                            useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)

    if False:
        mbs.CreateRevoluteJoint(bodyNumbers=[b5, oGround],
                                position=links_pts[5][1],
                                axis=joint_axes[5],  # rotation along global z-axis
                                # axis=[0, 0, 1],
                                useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)
    else:
        joint5Frame = ComputeOrthonormalBasis(joint_axes[5])

        mbs.CreateGenericJoint(bodyNumbers=[b5, oGround],
                               position=links_pts[5][1],
                               rotationMatrixAxes=joint5Frame,
                               constrainedAxes=[0, 1, 1, 1, 1, 0],
                               # axis=joint_axes[3],  # rotation along global z-axis
                               # axis=[0, 0, 1],
                               useGlobalFrame=True,
                               axesRadius=0.02,
                               axesLength=0.14)

    mbs.Assemble()

    # some simulation parameters:
    simulationSettings = exu.SimulationSettings()  # takes currently set values or default values
    simulationSettings.timeIntegration.numberOfSteps = 1000
    simulationSettings.timeIntegration.endTime = 5
    simulationSettings.timeIntegration.verboseMode = 1

    # simulationSettings.timeIntegration.newton.numericalDifferentiation.doSystemWideDifferentiation = True
    # simulationSettings.timeIntegration.newton.relativeTolerance = 1e-6
    # simulationSettings.timeIntegration.newton.absoluteTolerance = 1e-4

    # for redundant constraints, the following two settings:
    simulationSettings.linearSolverSettings.ignoreSingularJacobian = True
    simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense  # use EigenSparse for larger systems alternatively

    ### ADDED BY ME
    mbs.ComputeSystemDegreeOfFreedom(verbose=True, useSVD=True)
    # mbs.DrawSystemGraph(useItemTypes=True)  # TODO not working
    # mbs.systemData.Info()
    ###
    if useGraphics:
        exu.StartRenderer()
        print('wait')
        mbs.WaitForUserToContinue()

    mbs.SolveDynamic(simulationSettings,
                     # solverType=exu.DynamicSolverType.TrapezoidalIndex2, #index 2
                     )
    # , showHints=True, showCausingItems=True)

    ## stop graphics
    if useGraphics:
        SC.WaitForRenderEngineStopFlag()
        exu.StopRenderer() # safely close rendering window!


    # visualize results after simulation:
    mbs.SolutionViewer()


