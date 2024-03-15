from exudyn.utilities import *

from rational_linkages import ExudynAnalysis, TransfMatrix
from rational_linkages.models import bennett_ark24

mechanism_case = '4r'

if __name__ == '__main__':
    m = bennett_ark24()
    number_of_links = 4

    # exudyn parameters
    w = 0.06  # width of link
    simulation_time = 1.5
    torque_load = [0, 0, 50]

    # last joint is "generic" with these constraints
    constrained_axes = [1, 1, 0, 0, 0, 0]

    # get parameters from rational_linkages mechanism model
    (links_pts, links_lengths, body_dim, links_masses_pts, joint_axes,
     rel_links_pts) = ExudynAnalysis().get_exudyn_params(m,
                                                         link_radius=w,
                                                         #scale=0.2
                                                         )

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
    f = m.get_frames()


    # ground link
    A = TransfMatrix.from_vectors(normal_x=[-1, -0.8, 0], approach_z=joint_axes[-1])
    A = TransfMatrix()
    gGround0 = GraphicsDataFromSTLfile('link3.stl',
                                       color=[0.5, 0.5, 0, 1],
                                       scale=0.005,
                                       pOff=links_pts[0][0] - 0.29 * joint_axes[-1],
                                       Aoff=A.rot_matrix()
                                       )
    graphics_bodies = [gGround0]

    f1 = f[1] * TransfMatrix.from_rpy([0, 0, -0.35 * np.pi])
    stlGrafics = GraphicsDataFromSTLfile('link0.stl',
                                         color=colors[0],
                                         scale=5,
                                         pOff=rel_links_pts[1][0] - joint_axes[
                                             0] * 10.458809 / 200,
                                         Aoff=f1.rot_matrix(),
                                         )
    graphics_bodies.append(stlGrafics)

    f2 = f[2] * TransfMatrix.from_rpy([0, 0, 0.485 * np.pi])
    stlGrafics = GraphicsDataFromSTLfile('link1.stl',
                                         color=[0, 1, 0, 1],
                                         scale=5,
                                         pOff=rel_links_pts[2][0] + joint_axes[1] / 200,
                                         Aoff=f2.rot_matrix(),
                                         )
    graphics_bodies.append(stlGrafics)

    f3 = f[3] * TransfMatrix.from_rpy([0, 0, 0.72 * np.pi])
    stlGrafics = GraphicsDataFromSTLfile('link2.stl',
                                         color=[0, 0, 1, 1],
                                         scale=5,
                                         pOff=rel_links_pts[3][0] + joint_axes[
                                             2] * 22 / 200,
                                         Aoff=f3.rot_matrix(),
                                         )
    graphics_bodies.append(stlGrafics)

    # RIGID BODIES
    # ground body
    oGround = mbs.AddObject(
        ObjectGround(visualization=VObjectGround(graphicsData=[gGround0])))
    bodies = [oGround]

    body_markers = [None] #[mbs.AddMarker(MarkerNodeRigid(nodeNumber=mbs.GetObject(oGround)['nodeNumber']))]

    # other links
    for i in range(1, number_of_links):
        body = mbs.CreateRigidBody(
            inertia=inertias[i],
            referencePosition=links_masses_pts[i],
            gravity=g,
            graphicsDataList=[graphics_bodies[i]]
        )
        bodies.append(body)
        body_markers.append(mbs.AddMarker(MarkerNodeRigid(nodeNumber=mbs.GetObject(body)['nodeNumber'])))

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

    body = mbs.CreateGenericJoint(bodyNumbers=[bodies[-1], bodies[0]],
                                  position=links_pts[-1][1],
                                  rotationMatrixAxes=last_joint_frame,
                                  constrainedAxes=constrained_axes,
                                  axesRadius=0.5*w,
                                  axesLength=w)
    bodies.append(body)

    gContact = mbs.AddGeneralContact()
    gContact.verboseMode = 1
    gContact.resetSearchTreeInterval = 10000  # interval at which search tree memory is cleared
    frictionCoeff = 0
    gContact.SetFrictionPairings(frictionCoeff * np.eye(1))

    k = 2e4 * 4
    d = 0.002 * k

    for i in range(1, number_of_links):
        [meshPoints, meshTrigs] = GraphicsData2PointsAndTrigs(graphics_bodies[i])

        if i == 1:
            #[meshPoints, meshTrigs] = RefineMesh(meshPoints, meshTrigs)
            [meshPoints2, meshTrigs2] = ShrinkMeshNormalToSurface(meshPoints, meshTrigs, 0.1 * w)
            gCube = GraphicsDataFromPointsAndTrigs(meshPoints, meshTrigs,
                                                   color=colors[i])
            gList = [gCube]

            # add points for contact to visualization (shrinked)
            for p in meshPoints2:
                gList += [GraphicsDataSphere(point=p, radius=w, color=color4red)]

            print(len(gList))

            [nMassCube0, oMassCube0] = AddRigidBody(mainSys=mbs,
                                                    inertia=inertias[i],
                                                    nodeType=exu.NodeType.RotationRotationVector,
                                                    position=rel_links_pts[i][0],
                                                    # rotationMatrix=RotationMatrixZ(0.),
                                                    angularVelocity=[0, 0, 0],
                                                    gravity=g,
                                                    graphicsDataList=gList,
                                                    )

            nCube0 = nMassCube0
            mCube0 = mbs.AddMarker(MarkerNodeRigid(nodeNumber=nMassCube0))

            gContact.AddTrianglesRigidBodyBased(rigidBodyMarkerIndex=body_markers[i],
                                                contactStiffness=k,
                                                contactDamping=d,
                                                frictionMaterialIndex=1,
                                                pointList=meshPoints,
                                                triangleList=meshTrigs)

            for p in meshPoints2:
                mPoint = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oMassCube0, localPosition=p))
                gContact.AddSphereWithMarker(mPoint, radius=w, contactStiffness=k,
                                             contactDamping=d, frictionMaterialIndex=1)
        else:
            gContact.AddTrianglesRigidBodyBased(rigidBodyMarkerIndex=body_markers[i],
                                                contactStiffness=k,
                                                contactDamping=d,
                                                frictionMaterialIndex=0,
                                                pointList=meshPoints,
                                                triangleList=meshTrigs)

    # # SPHERE
    # color4node = color4steelblue
    # pS0 = [0, 0, 0]
    # r = 0.1
    # gList = [GraphicsDataSphere(point=[0, 0, 0], radius=0.1, color=color4node, nTiles=24)]
    #
    # RBinertia = InertiaSphere(1, r * 1)
    #
    # [nMass, oMass] = AddRigidBody(mainSys=mbs, inertia=RBinertia,
    #                               # nodeType=exu.NodeType.RotationRxyz,
    #                               nodeType=exu.NodeType.RotationRotationVector,
    #                               position=[0, 0, 0.5],
    #                               rotationMatrix=np.eye(3),
    #                               gravity=[0., 0., -9.81],
    #                               graphicsDataList=gList,
    #                               )
    #
    # mNode = mbs.AddMarker(MarkerNodeRigid(nodeNumber=nMass))
    # mBody = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oMass, localPosition=pS0))
    # gContact.AddSphereWithMarker(mBody, radius=r, contactStiffness=k, contactDamping=d,
    #                              frictionMaterialIndex=0)

    def PreStepUserFunction(mbs, t):
        gContact.UpdateContacts(mbs)
        l = gContact.GetActiveContacts(exu.ContactTypeIndex.IndexSpheresMarkerBased, -1)
        # l = gContact.GetActiveContacts(exu.ContactTypeIndex.IndexTrigsRigidBodyBased, -1)
        print('t=', t, 'active contact spheres=', sum(l))
        return True

    mbs.SetPreStepUserFunction(PreStepUserFunction)

    mbs.Assemble()

    # simulation parameters:
    simulationSettings = exu.SimulationSettings()

    simulationSettings.timeIntegration.numberOfSteps = 1000
    simulationSettings.timeIntegration.endTime = simulation_time
    simulationSettings.timeIntegration.verboseMode = 1

    simulationSettings.linearSolverSettings.ignoreSingularJacobian = True
    simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense

    mbs.ComputeSystemDegreeOfFreedom(verbose=True, useSVD=True)
    #mbs.systemData.Info()

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
