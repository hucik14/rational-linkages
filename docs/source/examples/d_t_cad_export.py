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
