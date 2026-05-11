from rational_linkages import Plotter
from rational_linkages.models import bennett_ark24

m = bennett_ark24()
dh, design_params, design_points = m.get_design(return_point_homogeneous=True,
                                                pretty_print=False)

# obtain points on joint0
base_joint0_pts = design_points[0]

# obtain points on the last joint (joint3)
base_joint3_pts = design_points[-1]

p = Plotter(m, arrows_length=0.1, backend='matplotlib')
# plot the first joint points
for i, pt in enumerate(base_joint0_pts):
    p.plot(pt, label=f'j0{i}')

# plot the last joint points
for i, pt in enumerate(base_joint3_pts):
    p.plot(pt, label=f'j3{i}')
p.show()

m.export_single_mesh(add_tool_frame=True,
                     file_name='mesh_bennett_ark24.stl')
