from exudyn.utilities import *

from rational_linkages import ExudynAnalysis, TransfMatrix
from rational_linkages.models import bennett_ark24, collisions_free_6r

# choose 4-bar or 6-bar mechanism
mechanism_case = '4r'
#mechanism_case = '6r'


if __name__ == '__main__':
    # define initial parameters for simulation
    match mechanism_case:
        case '4r':
            m = bennett_ark24()
            number_of_links = 4

            # exudyn parameters
            w = 0.06  # width of link
            simulation_time = 1.5
            torque_load = [0, 0, 50]

            # last joint is "generic" with these constraints
            constrained_axes = [1, 1, 0, 0, 0, 0]
        case '6r':
            m = collisions_free_6r()
            number_of_links = 6

            # exudyn parameters
            w = 0.1  # width of link
            simulation_time = 5
            torque_load = [6000, 0, 0]

            # last joint is "generic" with these constraints
            constrained_axes = [1, 1, 1, 0, 1, 0]

    # get parameters from rational_linkages mechanism model
    (links_pts, links_lengths, body_dim, links_masses_pts, joint_axes,
     rel_links_pts) = ExudynAnalysis().get_exudyn_params(m, link_radius=w)

    ###################################################################################
    # EXUDYN PART #####################################################################

    useGraphics = True
    g = [0, 0, -9.81]  # gravity
    colors = [color4red, color4green, color4blue, color4magenta, color4yellow]
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    inertias = [InertiaCuboid(density=5000, sideLengths=body_dim[i])
                for i in range(number_of_links)]

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

    f = m.get_frames()
    f1 = f[1] * TransfMatrix.from_rpy([0, 0, -0.35 * np.pi])
    stlGrafics = GraphicsDataFromSTLfile('link0.stl',
                                         color=colors[0],
                                         scale=5,
                                         pOff=rel_links_pts[1][0] - joint_axes[0] * 10.458809/200,
                                         Aoff=f1.rot_matrix(),
                                         )
    graphics_bodies[1] = stlGrafics

    #A = TransfMatrix.from_vectors(normal_x=[0.1, 0.8, 0], approach_z=joint_axes[1])
    f = m.get_frames()
    f2 = f[2] * TransfMatrix.from_rpy([0, 0, 0.485 * np.pi])
    stlGrafics = GraphicsDataFromSTLfile('link1.stl',
                                         color=[0, 1, 0, 1],
                                         scale=5,
                                         pOff=rel_links_pts[2][0] + joint_axes[1] / 200,
                                         Aoff=f2.rot_matrix(),
                                         )
    graphics_bodies[2] = stlGrafics

    f3 = f[3] * TransfMatrix.from_rpy([0, 0, 0.72*np.pi])
    stlGrafics = GraphicsDataFromSTLfile('link2.stl',
                                         color=[0, 0, 1, 1],
                                         scale=5,
                                         pOff=rel_links_pts[3][0] + joint_axes[2] *22/200,
                                         Aoff=f3.rot_matrix(),
                                         )
    graphics_bodies[3] = stlGrafics

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
    mBody = mbs.AddMarker(
        MarkerNodeRigid(nodeNumber=mbs.GetObject(bodies[-1])['nodeNumber']))
    mbs.AddLoad(Torque(markerNumber=mBody, loadVector=torque_load))

    last_joint_frame = ComputeOrthonormalBasis(joint_axes[-1])

    mbs.CreateGenericJoint(bodyNumbers=[bodies[-1], bodies[0]],
                           position=links_pts[-1][1],
                           rotationMatrixAxes=last_joint_frame,
                           constrainedAxes=constrained_axes,
                           useGlobalFrame=True,
                           axesRadius=0.02,
                           axesLength=0.14)

    mbs.Assemble()

    # simulation parameters:
    simulationSettings = exu.SimulationSettings()

    simulationSettings.timeIntegration.numberOfSteps = 1000
    simulationSettings.timeIntegration.endTime = simulation_time
    simulationSettings.timeIntegration.verboseMode = 1

    simulationSettings.linearSolverSettings.ignoreSingularJacobian = True
    simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense

    mbs.ComputeSystemDegreeOfFreedom(verbose=True, useSVD=True)
    # mbs.systemData.Info()

    if useGraphics:
        exu.StartRenderer()
        print('wait')
        mbs.WaitForUserToContinue()

    mbs.SolveDynamic(simulationSettings)#, showHints=True, showCausingItems=True)

    # stop graphics
    if useGraphics:
        SC.WaitForRenderEngineStopFlag()
        exu.StopRenderer()

    # visualize results after simulation:
    mbs.SolutionViewer()
