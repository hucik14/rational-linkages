import numpy as np


class LinkageCAD:
    """
    This class is responsible for generating CAD models of the linkage designs.
    """
    def __init__(self,
                 design_points,
                 tool = None,):
        """

        :param design_points:
        :param tool:
        """
        self.design_points = design_points
        self.tool = tool

        self.num_joints = int(len(design_points - 1) / 2)

    def export_single_mesh(self,
                           link_diameter: float = 0.01,
                           joint_diameter: float = 0.02,
                           add_tool_frame: bool = True,
                           file_name: str = 'mechanism_mesh.stl'):
        """
        Export a single STL mesh of the mechanism at home configuration.

        :param float scale: scaling factor of the mechanism
        :param float link_diameter: radius of the link cylinders
        :param float joint_diameter: radius of the joint cylinders
        :param bool smallest_polyline: if True, use the linkage design following the
            smallest polyline between the connection points
        :param bool add_tool_frame: if True, add a tool link with frame representing
            the tool frame
        :param str file_name: name of the output STL file

        """
        try:
            # lazy import
            import trimesh
        except ImportError:
            raise ImportError(
                "To create meshes that can be exported as STL files, the packages 'trimesh' and 'manifold3d' are required."
            )

        mesh_points = self.design_points

        cylinders = []
        for i in range(self.num_joints):
            cylinders.append(self._cylinder_between_points(mesh_points[2*i],
                                                           mesh_points[2*i+1],
                                                           joint_diameter / 2))
            cylinders.append(self._cylinder_between_points(mesh_points[2*i+1],
                                                           mesh_points[2*i+2],
                                                           link_diameter / 2))
        # filter None cylinders
        cylinders = [c for c in cylinders if c is not None]

        if add_tool_frame:
            # add tool link and frame mesh
            tool_origin = np.array([0, 0, 0])
            tool_axes = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])

            # middle points
            idx = len(mesh_points) / 2

            pt0 = mesh_points[int(idx)]
            pt1 = mesh_points[int(idx)-1]

            length_tool_link = np.linalg.norm(pt1 - tool_origin)

            cylinders.append(self._cylinder_between_points(tool_origin,
                                                           pt0,
                                                           link_diameter / 2))
            cylinders.append(self._cylinder_between_points(tool_origin,
                                                           pt1,
                                                           link_diameter / 2))
            for axis in tool_axes:
                cylinders.append(self._cylinder_between_points(
                    tool_origin,
                    axis * length_tool_link * 0.2,
                    link_diameter / 4))

        # boolean union of the cylinders
        combined = trimesh.boolean.union(cylinders, engine='manifold')

        # export as STL
        try:
            combined.export(file_name)
            print("Mesh exported as", file_name)
        except Exception as e:
            print("Failed to export mesh:", e)


    @staticmethod
    def _cylinder_between_points(p0, p1, radius=1.0):
        """
        Create a cylinder mesh between two points.

        :param p0: first point
        :param p1: second point
        :param radius: radius of the cylinder

        :return: cylinder mesh
        :rtype: trimesh.Trimesh
        """
        try:
            import trimesh  # lazy import
        except ImportError:
            raise ImportError(
                "To create meshes that can be exported as STL files, the packages 'trimesh' and 'manifold3d' are required."
            )

        p0 = np.asarray(p0)
        p1 = np.asarray(p1)

        # Vector and length
        vec = p1 - p0
        length = np.linalg.norm(vec)

        if length < 1e-9:
            return None

        direction = vec / length

        # Create cylinder aligned with Z axis
        cylinder = trimesh.creation.cylinder(
            radius=radius,
            height=length,
        )

        # Align Z axis to direction vector
        tr = trimesh.geometry.align_vectors([0, 0, 1], direction)
        cylinder.apply_transform(tr)

        # Move to midpoint
        midpoint = (p0 + p1) / 2
        cylinder.apply_translation(midpoint)

        return cylinder

    def export_single_solid(
            self,
            units: str = 'mm',
            link_diameter: float = 10,
            joint_diameter: float = 20,
            smallest_polyline: bool = False,
            add_tool_frame: bool = True,
            file_name: str = "mechanism.step",
    ):
        """
        Export a single CAD solid (STEP) of the mechanism at home configuration.

        :param str units: unit of the scale, can be 'm' or 'mm'
        :param float scale: scaling factor
        :param float link_diameter: diameter of link cylinders, default is 10 units
        :param float joint_diameter: diameter of joint cylinders, default is 20 units
        :param float smallest_polyline: use smallest polyline linkage
        :param float add_tool_frame: add tool frame geometry
        :param float file_name: output file name (.step recommended)
        """

        try:
            import build123d  # lazy import
        except ImportError:
            raise ImportError(
                "Build123d is required for CAD export. Use: pip install build123d"
            )

        if units == "m":
            scale = 1.0
        elif units == "mm":
            scale = 1000.0
        else:
            raise ValueError("Unsupported unit.")

        if smallest_polyline:
            self.smallest_polyline(update_design=True)

        _, _, mesh_points = self.get_design(pretty_print=False, update_design=True)
        mesh_points = np.vstack(mesh_points) * scale
        mesh_points = np.vstack([mesh_points, mesh_points[0]])

        solids = []

        # --- helper ---
        def cylinder_between_points(p0, p1, radius):
            p0 = np.asarray(p0)
            p1 = np.asarray(p1)

            vec = p1 - p0
            length = np.linalg.norm(vec)

            if length < 1e-9:
                return None

            direction = vec / length

            # Cylinder centered at origin, aligned with Z
            cyl = build123d.Cylinder(radius=radius, height=length)

            z_axis = np.array([0, 0, 1])
            axis = np.cross(z_axis, direction)
            axis_norm = np.linalg.norm(axis)

            if axis_norm > 1e-9:
                axis /= axis_norm
                angle = np.degrees(
                    np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                )

                cyl = cyl.rotate(
                    build123d.Axis((0, 0, 0), build123d.Vector(*axis)),
                    angle,
                )

            midpoint = (p0 + p1) / 2
            cyl = cyl.locate(build123d.Location(midpoint))

            return cyl

        # --- build mechanism ---
        for i in range(self.num_joints):
            solids.append(
                cylinder_between_points(
                    mesh_points[2 * i],
                    mesh_points[2 * i + 1],
                    joint_diameter / 2,
                )
            )
            solids.append(
                cylinder_between_points(
                    mesh_points[2 * i + 1],
                    mesh_points[2 * i + 2],
                    link_diameter / 2,
                )
            )

        solids = [s for s in solids if s is not None]

        # --- tool frame ---
        if add_tool_frame:
            tool_origin = np.array([0, 0, 0])
            tool_axes = np.eye(3)

            idx = len(mesh_points) // 2
            pt0 = mesh_points[idx]
            pt1 = mesh_points[idx - 1]

            length_tool_link = np.linalg.norm(pt1 - tool_origin)

            solids.append(
                cylinder_between_points(tool_origin, pt0, link_diameter / 2)
            )
            solids.append(
                cylinder_between_points(tool_origin, pt1, link_diameter / 2)
            )

            for axis in tool_axes:
                solids.append(
                    cylinder_between_points(
                        tool_origin,
                        axis * length_tool_link * 0.2,
                        link_diameter / 4,
                    )
                )

        # --- boolean fuse ---
        combined = solids[0]
        for s in solids[1:]:
            combined = combined.fuse(s)

        # --- export ---
        build123d.export_step(combined, file_name)
        print("STEP file exported as", file_name)
