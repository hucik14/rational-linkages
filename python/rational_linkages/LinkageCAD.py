from __future__ import annotations

from typing import Iterator

import numpy as np


class LinkageCAD:
    """
    Generate CAD models of linkage designs.
    """

    def __init__(self, design_points, tool=None):
        """
        Initialize the CAD helper for a linkage design.

        :param design_points: Sequence of design points describing the linkage.
        :param tool: Optional tool definition associated with the linkage.
        """
        self.design_points = np.asarray(design_points, dtype=float)
        self.tool = tool

    @property
    def num_joints(self) -> int:
        """
        Return the number of joints in the linkage.

        :return: Number of joints.
        :rtype: int
        """
        return (len(self.design_points) - 1) // 2

    def export_single_mesh(
            self,
            link_diameter: float = 0.01,
            joint_diameter: float = 0.02,
            add_tool_frame: bool = True,
            file_name: str = "mechanism_mesh.stl") -> None:
        """
        Export a single STL mesh of the mechanism at home configuration.

        :param float link_diameter: Diameter of the cylindrical links (in meters).
        :param float joint_diameter: Diameter of the cylindrical joints (in meters).
        :param bool add_tool_frame: Whether to include a simple tool frame geometry.
        :param str file_name: Output STL file name.
        """
        try:
            import trimesh  # lazy import
        except ImportError as exc:
            raise ImportError(
                "To export STL meshes, install 'trimesh' and 'manifold3d'."
            ) from exc

        points = self._scaled_points(units="m")
        segments = self._iter_all_segments(
            points=points,
            link_radius=link_diameter / 2,
            joint_radius=joint_diameter / 2,
            add_tool_frame=add_tool_frame,
        )

        cylinders = [
            self._trimesh_cylinder(p0, p1, radius)
            for p0, p1, radius in segments
        ]
        cylinders = [c for c in cylinders if c is not None]

        if not cylinders:
            raise ValueError("No valid cylinders were generated.")

        combined = trimesh.boolean.union(cylinders, engine="manifold")
        combined.export(file_name)
        print(f"Mesh exported to {file_name!r}")

    def export_single_solid(
            self,
            units: str = "mm",
            link_diameter: float = 10,
            joint_diameter: float = 20,
            add_tool_frame: bool = True,
            file_name: str = "mechanism.step",) -> None:
        """
        Export a single CAD solid (STEP) of the mechanism at home configuration.

        :param str units: Units for the design (e.g., "mm" or "m").
        :param float link_diameter: Diameter of the cylindrical links (default 10; i.e. mm).
        :param float joint_diameter: Diameter of the cylindrical joints (default 20; i.e. mm).
        :param bool add_tool_frame: Whether to include a simple tool frame geometry.
        :param str file_name: Output STEP file name.
        """
        try:
            import build123d  # lazy import
        except ImportError as exc:
            raise ImportError(
                "Build123d is required for CAD export. Use: pip install build123d"
            ) from exc

        points = self._scaled_points(units=units)
        segments = self._iter_all_segments(
            points=points,
            link_radius=link_diameter / 2,
            joint_radius=joint_diameter / 2,
            add_tool_frame=add_tool_frame,
        )

        solids = [
            self._build123d_cylinder(p0, p1, radius, build123d)
            for p0, p1, radius in segments
        ]
        solids = [s for s in solids if s is not None]

        if not solids:
            raise ValueError("No valid solids were generated.")

        combined = solids[0]
        for solid in solids[1:]:
            combined = combined.fuse(solid)

        build123d.export_step(combined, file_name)
        print(f"CAD solid exported to {file_name!r}")

    def export_solids(self,
                      units: str = "mm",
                      link_diameter: float = 10,
                      joint_diameter: float = 20,
                      add_tool_frame: bool = True,
                      file_name: str = "mechanism_parts.step",) -> None:
        """
        Export mechanism assembly with individual CAD solids (STEP).

        :param str units: Units for the design (e.g., "mm" or "m").
        :param float link_diameter: Diameter of the cylindrical links (default 10; i.e. mm).
        :param float joint_diameter: Diameter of the cylindrical joints (default 20; i.e. mm).
        :param bool add_tool_frame: Whether to include a simple tool frame geometry.
        :param str file_name: Output STEP file name.
        """
        try:
            import build123d  # lazy import
        except ImportError as exc:
            raise ImportError(
                "Build123d is required for CAD export. Use: pip install build123d"
            ) from exc

        points = self._scaled_points(units=units)
        segments = self._iter_all_segments(
            points=points,
            link_radius=link_diameter / 2,
            joint_radius=joint_diameter / 2,
            add_tool_frame=add_tool_frame,
        )

        solids = [
            self._build123d_cylinder(p0, p1, radius, build123d)
            for p0, p1, radius in segments
        ]

        if any(s is None for s in solids):
            raise ValueError("Degenerate segment encountered while building solids.")

        tool = None
        if add_tool_frame and solids:
            tool_parts = solids[-4:]
            tool = self._fuse_solids(tool_parts)
            solids = solids[:-4]

        if not solids:
            raise ValueError("No valid solids were generated.")

        links = self._build_link_solids(solids, tool=tool)

        assembly = build123d.Compound(label="assembly",
                                      children=links)

        build123d.export_step(assembly, file_name)
        print(f"CAD solids exported to {file_name!r}")

    def _build_link_solids(self,
                           solids: list,
                           tool=None,) -> list:
        """
        Build fused solids for all mechanism links.

        Each link is composed of three consecutive mechanism cylinders in the
        repeating pattern:
            [2*i-2, 2*i-1, 2*i]  (with cyclic indexing)

        :param list solids: Mechanism solids without tool-frame parts.
        :param tool: Optional fused tool solid attached to the middle link.

        :return: List of fused link solids.
        :rtype: list
        """
        n_links = self.num_joints
        n_segments = len(solids)

        if n_segments != 2 * n_links:
            raise ValueError(
                f"Expected {2 * n_links} mechanism solids, got {n_segments}."
            )

        middle_link_idx = n_links // 2
        links = []

        for i in range(n_links):
            idx0 = (2 * i - 2) % n_segments
            idx1 = (2 * i - 1) % n_segments
            idx2 = (2 * i) % n_segments

            parts = [solids[idx0], solids[idx1], solids[idx2]]

            if tool is not None and i == middle_link_idx:
                parts.append(tool)

            link = self._fuse_solids(parts)
            links.append(link)

        return links

    @staticmethod
    def _fuse_solids(solids: list):
        """
        Fuse a list of solids into a single solid.

        :param list solids: Solids to be fused.

        :return: Single fused solid.
        """
        if not solids:
            raise ValueError("No solids provided for fusion.")

        fused = solids[0]
        for solid in solids[1:]:
            fused = fused.fuse(solid)

        return fused

    def _scaled_points(self, units: str = "m") -> np.ndarray:
        """
        Return design points scaled to the requested units.

        :param str units: Units for the returned points, either "m" or "mm".

        :return: Scaled design points.
        :rtype: np.ndarray
        """
        if units == "m":
            scale = 1.0
        elif units == "mm":
            scale = 1000.0
        else:
            raise ValueError(f"Unsupported unit: {units!r}")

        return np.asarray(self.design_points, dtype=float) * scale

    def _iter_all_segments(
            self,
            points: np.ndarray,
            link_radius: float,
            joint_radius: float,
            add_tool_frame: bool,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
        """
        Yield all cylindrical segments of the mechanism and optional tool frame.

        :param np.ndarray points: Scaled design points.
        :param float link_radius: Radius of the link cylinders.
        :param float joint_radius: Radius of the joint cylinders.
        :param bool add_tool_frame: Whether to include the tool frame segments.

        :return: Iterator over segment tuples (p0, p1, radius).
        :rtype: Iterator[tuple[np.ndarray, np.ndarray, float]]
        """
        yield from self._iter_mechanism_segments(points, link_radius, joint_radius)

        if add_tool_frame:
            yield from self._iter_tool_segments(points, link_radius)

    def _iter_mechanism_segments(
            self,
            points: np.ndarray,
            link_radius: float,
            joint_radius: float,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
        """
        Yield cylindrical segments for the mechanism links and joints.

        :param np.ndarray points: Scaled design points.
        :param float link_radius: Radius of the link cylinders.
        :param float joint_radius: Radius of the joint cylinders.

        :return: Iterator over segment tuples (p0, p1, radius).
        :rtype: Iterator[tuple[np.ndarray, np.ndarray, float]]
        """
        for i in range(self.num_joints):
            yield points[2 * i], points[2 * i + 1], joint_radius
            yield points[2 * i + 1], points[2 * i + 2], link_radius

    def _iter_tool_segments(
            self,
            points: np.ndarray,
            link_radius: float,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
        """
        Yield cylindrical segments for the optional tool frame geometry.

        :param np.ndarray points: Scaled design points.
        :param float link_radius: Radius of the link cylinders.

        :return: Iterator over segment tuples (p0, p1, radius).
        :rtype: Iterator[tuple[np.ndarray, np.ndarray, float]]
        """
        tool_origin = np.zeros(3)
        tool_axes = np.eye(3)

        idx = len(points) // 2
        pt0 = points[idx]
        pt1 = points[idx - 1]

        mid_point = (pt0 + pt1) / 2

        length_tool_link = np.linalg.norm(pt1 - tool_origin)

        yield tool_origin, mid_point, link_radius
        # yield tool_origin, pt1, link_radius / 2

        for axis in tool_axes:
            yield tool_origin, axis * length_tool_link * 0.2, link_radius / 2

    @staticmethod
    def _segment_direction_and_length(p0: np.ndarray,
                                      p1: np.ndarray,
                                      tol: float = 1e-9,
                                      ) -> tuple[np.ndarray | None, float]:
        """
        Compute the unit direction vector and segment length between two points.

        :param np.ndarray p0: First point.
        :param np.ndarray p1: Second point.
        :param float tol: Tolerance below which the segment is treated as degenerate.

        :return: Tuple of unit direction vector and length. If the segment is too short,
            return None and 0.0.
        :rtype: tuple[np.ndarray | None, float]
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)

        vec = p1 - p0
        length = np.linalg.norm(vec)

        if length < tol:
            return None, 0.0

        return vec / length, length

    @staticmethod
    def _trimesh_cylinder(p0, p1, radius):
        """
        Create a trimesh cylinder between two points.

        :param p0: First point.
        :param p1: Second point.
        :param float radius: Cylinder radius.

        :return: Cylinder mesh, or None for a degenerate segment.
        """
        try:
            import trimesh  # lazy import
        except ImportError as exc:
            raise ImportError(
                "To export STL meshes, install 'trimesh' and 'manifold3d'."
            ) from exc

        direction, length = LinkageCAD._segment_direction_and_length(p0, p1)
        if direction is None:
            return None

        cylinder = trimesh.creation.cylinder(radius=radius, height=length)
        transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
        cylinder.apply_transform(transform)
        cylinder.apply_translation((np.asarray(p0) + np.asarray(p1)) / 2)
        return cylinder

    @staticmethod
    def _build123d_cylinder(p0, p1, radius, build123d):
        """
        Create a build123d cylinder between two points.

        :param p0: First point.
        :param p1: Second point.
        :param float radius: Cylinder radius.
        :param build123d: Imported build123d module.

        :return: Cylinder solid, or None for a degenerate segment.
        """
        direction, length = LinkageCAD._segment_direction_and_length(p0, p1)
        if direction is None:
            return None

        cyl = build123d.Cylinder(radius=radius, height=length)

        z_axis = np.array([0.0, 0.0, 1.0])
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

        midpoint = (np.asarray(p0) + np.asarray(p1)) / 2
        return cyl.locate(build123d.Location(tuple(midpoint)))