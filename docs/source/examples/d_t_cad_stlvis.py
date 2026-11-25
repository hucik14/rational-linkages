from rational_linkages import MotionDesigner, TransfMatrix
from rational_linkages.models import cart_stl

md = MotionDesigner(method='quadratic_from_poses',
                    preview_mechanism=True,
                    sliders_range=2.,
                    arrows_length=0.1)
path_to_stl = cart_stl()  # replace with path to your STL, for example:
# path_to_stl = "cart.stl"

# add transform for the mesh if needed
tr = TransfMatrix.from_rpy_xyz([0, 0, -90], [0.23, -0.44, -1.2], unit="deg")

md.add_mesh_from_stl(path_to_stl, scale=1, transform=tr)
md.show()