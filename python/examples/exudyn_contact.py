from exudyn.utilities import *

from rational_linkages import ExudynAnalysis, TransfMatrix
from rational_linkages.models import bennett_ark24

mechanism_case = '4r'

if __name__ == '__main__':
    m = bennett_ark24()
    number_of_links = 4

    # exudyn parameters
    w = 0.015  # width of link
    simulation_time = 1
    torque_load = [0, 0, 0.04]

    # last joint is "generic" with these constraints
    constrained_axes = [1, 1, 0, 0, 0, 0]

    # get parameters from rational_linkages mechanism model
    (links_pts, links_lengths, body_dim, links_masses_pts, joint_axes,
     rel_links_pts) = ExudynAnalysis().get_exudyn_params(m,
                                                         link_radius=w,
                                                         scale=0.2)

    ###################################################################################
    # EXUDYN PART #####################################################################

    useGraphics = True
    g = [0, 0, -9.81]  # gravity
    colors = [color4red, color4green, color4blue, color4magenta, color4yellow]
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()

    inertias = [InertiaCuboid(density=3000, sideLengths=body_dim[i])
                for i in range(number_of_links)]

    # GRAPIHCS BODIES
    # ground link
    A = TransfMatrix.from_vectors(normal_x=[-0.55, 0.4, 0], approach_z=joint_axes[0])
    gGround0 = GraphicsDataFromSTLfile('link3.stl',
                                       color=[0.5, 0.5, 0, 1],
                                       scale=0.001,
                                       pOff=links_pts[0][0],
                                       Aoff=A.rot_matrix()
                                       )
    graphics_bodies = [gGround0]

    A = TransfMatrix.from_vectors(normal_x=[-0.55, 0.4, 0], approach_z=joint_axes[0])
    stlGrafics = GraphicsDataFromSTLfile('link0.stl',
                                         color=[1, 0, 0, 1],
                                         scale=1.0,
                                         pOff=rel_links_pts[1][0],
                                         Aoff=A.rot_matrix()
                                         )
    graphics_bodies.append(stlGrafics)

    A = TransfMatrix.from_vectors(normal_x=[0, 1, 0], approach_z=joint_axes[1])
    stlGrafics = GraphicsDataFromSTLfile('link1.stl',
                                         color=[0, 1, 0, 1],
                                         scale=1.0,
                                         pOff=rel_links_pts[2][0],
                                         Aoff=A.rot_matrix()
                                         )
    graphics_bodies.append(stlGrafics)

    A = TransfMatrix.from_vectors(normal_x=[0.5, -0.5, 0], approach_z=joint_axes[2])
    stlGrafics = GraphicsDataFromSTLfile('link2.stl',
                                         color=[0, 0, 1, 1],
                                         scale=1.0,
                                         pOff=rel_links_pts[3][0],
                                         Aoff=A.rot_matrix()
                                         )
    graphics_bodies.append(stlGrafics)

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
            axisRadius=0.5*w,
            axisLength=w
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
                           axesRadius=0.5*w,
                           axesLength=w)

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
