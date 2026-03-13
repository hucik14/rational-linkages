.. _cad_tutorial:

CAD Modeling
============

Before generating solid or mesh models, see the tutorial :ref:`mechanism_with_tool` which explains
how the line model of the mechanisms look like in the backend and how the design parameters can be
generated.

Generating CAD Models
---------------------

There are three methods for exporting the mechanism as a mesh or solid model. The Python libraries
``build123d`` are required for STEP files generation, while
``trimesh`` and ``manifold3d`` are required for STL generation. You can install them using pip:

.. code-block:: bash

    pip install build123d trimesh manifold3d


:meth:`.RationalMechanism.export_single_mesh()` returns an STL mesh file of the whole mechanisms. Similarly,
:meth:`.RationalMechanism.export_single_solid()` returns a STEP file. In both cases, the mechanism bodies
are merged in a single body.

The method :meth:`.RationalMechanism.export_solids()` returns a STEP file, but it consists of assembled solids.
The joints of two links overlap. When importing in your CAD, keep in mind to import it as assembly.

All methods accept various arguments for customization of the output. For example,
they accept arguments for size of the joints and links cylinders, and also can
add the tool link and frame meshes. Be careful with units (e.g. in :meth:`.RationalMechanism.export_single_solid()`)
it has to be specified if the result and inputs are in millimeters or meters.

.. testcode:: [cad-export]

    from rational_linkages.models import bennett_ark24

    m = bennett_ark24()

    m.export_single_mesh(scale=1.0,  # mind that this example will produce a tiny model
                         link_diameter=0.01,
                         joint_diameter=0.02,
                         add_tool_frame=True,
                         file_name='mechanism.stl')

    m.export_solids(units="mm",
                    link_diameter=10,  # 10 mm if units="mm", otherwise 10 m
                    joint_diameter=20,  # 20 mm if units="mm"
                    add_tool_frame=True,
                    file_name="mechanism.step")

.. testoutput:: [cad-export]

    Mesh exported as 'mechanism.stl'
    CAD solids exported to 'mechanism.step'

.. testcleanup:: [cad-export]

        del bennett_ark24, m

.. _stl_preview-motion_designer:

Visualizing Custom STL in MotionDesigner
----------------------------------------

Any STL file can be imported as solid body in :ref:`motion_designer`. This can be useful for visualizing
the mechanism in the context of the task, e.g., your work-cell.

It is important that the STL file coordinates are already in the world frame
(``p0`` of your mechanism in MotionDesigner). Place the STL file in the same directory as your script
and use the following pattern:

.. testcode:: [cad-import-motion-designer]

    from rational_linkages import MotionDesigner


    md = MotionDesigner(method='quadratic_from_poses',
                        preview_mechanism=True)
    md.add_mesh_from_stl("cnc_lathe.stl", scale=0.001)
    md.show()

.. testoutput:: [cad-import-motion_designer]

    Closing the window... generated points for interpolation:
    [1, 0, 0, 0, 0, 0, 0, 0]
    [ 1.          , -0.207522406 , -0.0333866662, -0.0691741237, -0.0625113682, -0.141265791 , -0.4478576802, -0.2637268902]
    [ 1.          ,  0.2333739522, -0.0427838517,  0.0777914503, -0.0839342318,  0.2991396249,  0.2980046603,  0.345444421 ]

.. testcleanup:: [cad-import-motion_designer]

    del MotionDesigner, md



As you can notice, argument ``preview_mechanism`` is set to ``True``. This will visualize
the mechanism in the MotionDesigner window, so you can check if the size and position of the mechanism is
adequate in the context of your task. However, the previewing has impact on rendering performance.