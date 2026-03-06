from __future__ import annotations

from typing import Iterator

import numpy as np


class LinkageCAD:
    """
    Generate CAD models of linkage designs.
    """

    def __init__(self, design_points, tool=None):
        self.design_points = np.asarray(design_points, dtype=float)
        self.tool = tool

    @property
    def num_joints(self) -> int:
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

        :param units: Units for the design (e.g., "mm" or "m").
        :param link_diameter: Diameter of the cylindrical links (default 10; i.e. mm).
        :param joint_diameter: Diameter of the cylindrical joints (default 20; i.e. mm).
        :param add_tool_frame: Whether to include a simple tool frame geometry.
        :param file_name: Output STEP file name.
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
        Export mehanism assembly with individual CAD solids (STEP).

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
        # TODO later


    def _scaled_points(self, units: str = "m") -> np.ndarray:
        """
        Return design points scaled to the requested units.
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
        Yield (p0, p1, radius) for all mechanism cylinders.
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
        Yield (p0, p1, radius) for the optional tool frame geometry.
        """
        tool_origin = np.zeros(3)
        tool_axes = np.eye(3)

        idx = len(points) // 2
        pt0 = points[idx]
        pt1 = points[idx - 1]

        length_tool_link = np.linalg.norm(pt1 - tool_origin)

        yield tool_origin, pt0, link_radius
        yield tool_origin, pt1, link_radius

        for axis in tool_axes:
            yield tool_origin, axis * length_tool_link * 0.2, link_radius / 2

    @staticmethod
    def _segment_direction_and_length(p0: np.ndarray,
                                      p1: np.ndarray,
                                      tol: float = 1e-9,
                                      ) -> tuple:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)

        vec = p1 - p0
        length = np.linalg.norm(vec)

        if length < tol:
            return None, 0.0

        return vec / length, length

    @staticmethod
    def _trimesh_cylinder(p0, p1, radius):
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