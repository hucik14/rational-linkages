from rational_linkages import ExudynAnalysis, DualQuaternion, Plotter
from rational_linkages.models import bennett_ark24

# import numpy as np

# import exudyn as exu
# from exudyn.itemInterface import *
from exudyn.utilities import *


if __name__ == '__main__':
    m = bennett_ark24()

    if False:
        p = Plotter(interactive=True, steps=200, arrows_length=0.08)
        p.plot(m, show_tool=True)
        p.plot(DualQuaternion())
        p.show()

    ###################################################################################
    ############################# Exudyn part #########################################

    useGraphics = True

    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    # physical parameters
    g = [0, 0, -9.81]
    number_of_links = 4

    links_pts, links_lengths, body_dim, links_masses_pts, joint_axes, rel_links_pts \
        = ExudynAnalysis().get_exudyn_params(m)
    w = 0.06  # width of link

    inertias = [InertiaCuboid(density=5000, sideLengths=body_dim[i])
                for i in range(number_of_links)]

    colors = [color4red, color4green, color4steelblue]

    # GRAPIHCS BODIES
    # ground link
    gGround0 = GraphicsDataRigidLink(p0=links_pts[0][0], p1=links_pts[0][1],
                                     axis0=joint_axes[-1], axis1=joint_axes[0],
                                     radius=[0.5 * w, 0.5 * w],
                                     thickness=w, width=[1.2 * w, 1.2 * w],
                                     color=color4darkgrey)
    graphics_bodies = [gGround0]

    # other links
    for i in range(1, number_of_links):
        graphics_body = GraphicsDataRigidLink(
            p0=rel_links_pts[i][0],
            p1=rel_links_pts[i][1],
            axis0=joint_axes[i - 1],
            axis1=joint_axes[i],
            radius=[0.5 * w, 0.5 * w],
            thickness=w,
            width=[1.2 * w, 1.2 * w],
            color=colors[i - 1]
        )
        graphics_bodies.append(graphics_body)

    # RIGID BODIES
    # ground body
    oGround = mbs.AddObject(
        ObjectGround(visualization=VObjectGround(graphicsData=[gGround0])))
    bodies = [oGround]

    # other links
    for i in range(1, number_of_links):
        body = mbs.CreateRigidBody(
            inertia=inertias[i],
            referencePosition=links_masses_pts[i],
            gravity=g,
            graphicsDataList=[graphics_bodies[i]]
        )
        bodies.append(body)

    # REVOLUTE JOINTS
    for i in range(number_of_links - 1):
        mbs.CreateRevoluteJoint(
            bodyNumbers=[bodies[i], bodies[i + 1]],
            position=links_pts[i][1],
            axis=joint_axes[i],
            useGlobalFrame=True,
            axisRadius=0.02,
            axisLength=0.14
        )

    # TORQUE
    mBody = mbs.AddMarker(MarkerNodeRigid(nodeNumber=mbs.GetObject(bodies[-1])['nodeNumber']))
    mbs.AddLoad(Torque(markerNumber=mBody, loadVector=[0, 0, 50]))

    # GENERIC JOINT - last joint
    if False:
        mbs.CreateRevoluteJoint(bodyNumbers=[bodies[-1], bodies[0]],
                                position=links_pts[3][1],
                                axis=joint_axes[3],
                                useGlobalFrame=True, axisRadius=0.02, axisLength=0.14)
    else:
        joint3Frame = ComputeOrthonormalBasis(joint_axes[3])

        mbs.CreateGenericJoint(bodyNumbers=[bodies[-1], bodies[0]],
                               position=links_pts[3][1],
                               rotationMatrixAxes=joint3Frame,
                               constrainedAxes=[1, 1, 0, 0, 0, 0],
                               useGlobalFrame=True,
                               axesRadius=0.02,
                               axesLength=0.14)

    mbs.Assemble()

    # simulation parameters:
    simulationSettings = exu.SimulationSettings()

    simulationSettings.timeIntegration.numberOfSteps = 1000
    simulationSettings.timeIntegration.endTime = 1.5
    simulationSettings.timeIntegration.verboseMode = 1

    simulationSettings.linearSolverSettings.ignoreSingularJacobian = True
    simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense

    mbs.ComputeSystemDegreeOfFreedom(verbose=True, useSVD=True)
    # mbs.systemData.Info()

    if useGraphics:
        exu.StartRenderer()
        print('wait')
        mbs.WaitForUserToContinue()

    mbs.SolveDynamic(simulationSettings,
                     #solverType=exu.DynamicSolverType.TrapezoidalIndex2, #index 2
                     )
    # , showHints=True, showCausingItems=True)

    ## stop graphics
    if useGraphics:
        SC.WaitForRenderEngineStopFlag()
        exu.StopRenderer()

    # visualize results after simulation:
    mbs.SolutionViewer()


