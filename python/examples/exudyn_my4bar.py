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
    g = [0.1, -9.81, 0]  # gravity + disturbance

    L = [0.16666667, 0.5, 0.194365066, 0.5]


    w = 0.06  # width
    bodyDim = [[L[1], w, w], [L[2], w, w], [L[3], w, w]]  # body dimensions
    # p0 =    [0,0,0]
    pMid0 = np.array([-0.25, 0, 0.2])  # center of mass, body0                       #TODO global frame
    pMid1 = np.array([-0.58333333, 0, 0.05])  # center of mass, body1
    pMid2 = np.array([-0.41666667, 0, -0.1])  # center of mass, body2

    # ground body
    graphicsCOM0 = GraphicsDataBasis(origin=[0, 0, 0], length=4 * w)
    oGround = mbs.AddObject(
        ObjectGround(visualization=VObjectGround(graphicsData=[graphicsCOM0])))
    markerGround0 = mbs.AddMarker(
        MarkerBodyRigid(bodyNumber=oGround, localPosition=[0, 0, 0]))
    markerGround1 = mbs.AddMarker(
        MarkerBodyRigid(bodyNumber=oGround, localPosition=[-L[0], 0, 0]))

    # %%++++++++++++++++++++++++++++++++++++++++++++++++++++
    # first link:
    iCube = [InertiaCuboid(density=5000, sideLengths=bodyDim[0]),
             InertiaCuboid(density=5000, sideLengths=bodyDim[1]),
             InertiaCuboid(density=5000, sideLengths=bodyDim[2])]

    # graphics for body
    graphicsBody0 = GraphicsDataRigidLink(p0=[0.5 * L[1], 0, 0], p1=[-0.5 * L[1], 0, 0],      # TODO local frame from center of mass?
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

    [n0, b0] = AddRigidBody(mainSys=mbs,
                            inertia=iCube[0],  # includes COM
                            nodeType=exu.NodeType.RotationEulerParameters,
                            position=pMid0,
                            #rotationMatrix=RotationMatrixZ(0.5 * pi),
                            gravity=g,
                            graphicsDataList=[graphicsBody0])

    markerBody0J0 = mbs.AddMarker(  # TODO what is a marker?
        MarkerBodyRigid(bodyNumber=b0, localPosition=[0.5 * L[1], 0, 0]))
    markerBody0J1 = mbs.AddMarker(
        MarkerBodyRigid(bodyNumber=b0, localPosition=[-0.5 * L[1], 0, 0]))

    [n1, b1] = AddRigidBody(mainSys=mbs,
                            inertia=iCube[1],  # includes COM
                            nodeType=exu.NodeType.RotationEulerParameters,
                            position=pMid1,
                            #rotationMatrix=RotationMatrixZ(0.),
                            gravity=g,
                            graphicsDataList=[graphicsBody1])
    markerBody1J0 = mbs.AddMarker(
        MarkerBodyRigid(bodyNumber=b1, localPosition=[0.08333333, 0, 0.05]))
    markerBody1J1 = mbs.AddMarker(
        MarkerBodyRigid(bodyNumber=b1, localPosition=[-0.08333333, 0, -0.05]))

    [n2, b2] = AddRigidBody(mainSys=mbs,                                                   # TODO is this doing anything in the background, or only getting g2
                            inertia=iCube[2],  # includes COM
                            nodeType=exu.NodeType.RotationEulerParameters,
                            position=pMid2,
                            #rotationMatrix=RotationMatrixZ(-0.5 * pi),
                            gravity=g,
                            graphicsDataList=[graphicsBody2])

    markerBody2J0 = mbs.AddMarker(
        MarkerBodyRigid(bodyNumber=b2, localPosition=[-0.5 * L[3], 0, 0]))
    markerBody2J1 = mbs.AddMarker(
        MarkerBodyRigid(bodyNumber=b2, localPosition=[0.5 * L[3], 0, 0]))

    # revolute joint option 1:
    mbs.AddObject(GenericJoint(markerNumbers=[markerGround0, markerBody0J0],
                               constrainedAxes=[1, 1, 1, 1, 1, 0],
                               visualization=VObjectJointGeneric(axesRadius=0.2 * w,
                                                                 axesLength=1.4 * w)))

    mbs.AddObject(GenericJoint(markerNumbers=[markerBody0J1, markerBody1J0],
                               constrainedAxes=[1, 1, 1, 1, 1, 0],
                               visualization=VObjectJointGeneric(axesRadius=0.2 * w,
                                                                 axesLength=1.4 * w)))

    mbs.AddObject(GenericJoint(markerNumbers=[markerBody1J1, markerBody2J0],
                               constrainedAxes=[1, 1, 1, 1, 1, 0],
                               visualization=VObjectJointGeneric(axesRadius=0.2 * w,
                                                                 axesLength=1.4 * w)))

    constrainedAxes3 = [1, 1, 1, 1, 1, 0]
    if case == 2:
        constrainedAxes3 = [1, 1, 0, 0, 0, 0]  # only these constraints are needed for closing loop!
        print('use non-redundant constraints for last joint:', constrainedAxes3)

    mbs.AddObject(GenericJoint(markerNumbers=[markerBody2J1, markerGround1],                    # TODO joint goes to some default pos?
                               constrainedAxes=constrainedAxes3,
                               visualization=VObjectJointGeneric(axesRadius=0.2 * w,
                                                                 axesLength=1.4 * w)))

    # position sensor at tip of body1
    sens1 = mbs.AddSensor(
        SensorBody(bodyNumber=b1, localPosition=[0, 0, 0.5],                                # TODO why Z-axis?
                   fileName='solution/sensorPos.txt',
                   outputVariableType=exu.OutputVariableType.Position))

    # %%++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # assemble system before solving
    mbs.Assemble()
    if False:
        mbs.systemData.Info()  # show detailed information
    if False:
        # from exudyn.utilities import DrawSystemGraph                                          # TODO returns error
        mbs.DrawSystemGraph(useItemTypes=True)  # draw nice graph of system

    simulationSettings = exu.SimulationSettings()  # takes currently set values or default values

    tEnd = 10  # simulation time
    h = 2e-3  # step size
    simulationSettings.timeIntegration.numberOfSteps = int(tEnd / h)
    simulationSettings.timeIntegration.endTime = tEnd
    simulationSettings.timeIntegration.verboseMode = 1
    # simulationSettings.timeIntegration.simulateInRealtime = True
    # simulationSettings.timeIntegration.realtimeFactor = 4

    if case == 1:
        simulationSettings.linearSolverSettings.ignoreSingularJacobian = True  # for redundant constraints

    simulationSettings.timeIntegration.newton.useModifiedNewton = True
    simulationSettings.solutionSettings.writeSolutionToFile = False
    # simulationSettings.solutionSettings.solutionWritePeriod = 0.005 #store every 5 ms

    SC.visualizationSettings.window.renderWindowSize = [1200, 1024]
    SC.visualizationSettings.openGL.multiSampling = 4
    SC.visualizationSettings.general.autoFitScene = False

    SC.visualizationSettings.nodes.drawNodesAsPoint = False
    SC.visualizationSettings.nodes.showBasis = True

    if useGraphics:
        exu.StartRenderer()
        if 'renderState' in exu.sys:  # reload old view
            SC.SetRenderState(exu.sys['renderState'])

        mbs.WaitForUserToContinue()  # stop before simulating

    try:  # solver will raise exception in case 1
        mbs.SolveDynamic(simulationSettings=simulationSettings, showHints=True)
    except:
        pass

    # mbs.SolveDynamic(simulationSettings = simulationSettings,
    #                  solverType=exu.DynamicSolverType.TrapezoidalIndex2)
    if useGraphics:
        SC.WaitForRenderEngineStopFlag()  # stop before closing
        exu.StopRenderer()  # safely close rendering window!

    # check redundant constraints and DOF:
    mbs.ComputeSystemDegreeOfFreedom(verbose=True)
