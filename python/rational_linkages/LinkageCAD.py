from __future__ import annotations

from typing import Iterator

import numpy


class LinkageCAD:
    """Generate CAD models for linkage designs.

    This helper provides routines to export the mechanism geometry as a mesh
    (STL) or as CAD solids (STEP) using optional backends such as ``trimesh``
    and ``build123d``.
    """

    def __init__(self, design_points, tool=None):
        """Create a LinkageCAD for a set of design points.

        Parameters
        ----------
        design_points
            Sequence of design points describing the linkage (list or array-like).
        tool, optional
            Optional tool definition associated with the linkage.
        """
        self.design_points = numpy.asarray(design_points, dtype=float)
        self.tool = tool

    @property
    def num_joints(self) -> int:
        """Return the number of joints in the linkage.

        Returns
        -------
        int
            Number of joints inferred from the design points.
        """
        return (len(self.design_points) - 1) // 2

    def export_single_mesh(
            self,
            link_diameter: float = 0.01,
            joint_diameter: float = 0.02,
            add_tool_frame: bool = True,
            file_name: str = "mechanism_mesh.stl") -> None:
        """Export a single STL mesh of the mechanism at the home configuration.

        Parameters
        ----------
        link_diameter, optional
            Diameter of the cylindrical links (in meters).
        joint_diameter, optional
            Diameter of the cylindrical joints (in meters).
        add_tool_frame, optional
            Whether to include a simple tool-frame geometry.
        file_name, optional
            Output STL file name.
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
        """Export a single CAD solid (STEP) of the mechanism.

        Parameters
        ----------
        units, optional
            Units for the design (e.g., ``"mm"`` or ``"m"``).
        link_diameter, optional
            Diameter of the cylindrical links (default 10; units match ``units``).
        joint_diameter, optional
            Diameter of the cylindrical joints (default 20; units match ``units``).
        add_tool_frame, optional
            Whether to include a simple tool-frame geometry.
        file_name, optional
            Output STEP file name.
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
        """Export the mechanism as individual CAD solids (STEP).

        Parameters
        ----------
        units, optional
            Units for the design (e.g., ``"mm"`` or ``"m"``).
        link_diameter, optional
            Diameter for link cylinders (default 10).
        joint_diameter, optional
            Diameter for joint cylinders (default 20).
        add_tool_frame, optional
            Whether to include the tool-frame geometry.
        file_name, optional
            Output STEP file name for the assembled parts.
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
        """Create fused solids representing each mechanism link.

        Each link is formed by fusing three consecutive cylinder solids. The
        repeating pattern uses cyclic indexing over the list of mechanism
        solids.

        Parameters
        ----------
        solids
            Mechanism solids (build123d solids) excluding tool-frame parts.
        tool, optional
            Optional fused tool solid to attach to the middle link.

        Returns
        -------
        list
            A list of fused link solids.
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
        """Fuse a sequence of solids into a single solid.

        Parameters
        ----------
        solids
            Iterable of solids (objects exposing a ``fuse`` method).

        Returns
        -------
        object
            The fused solid composed from the provided parts.
        """
        if not solids:
            raise ValueError("No solids provided for fusion.")

        fused = solids[0]
        for solid in solids[1:]:
            fused = fused.fuse(solid)

        return fused

    def _scaled_points(self, units: str = "m") -> numpy.ndarray:
        """Return design points scaled to the requested units.

        Parameters
        ----------
        units, optional
            Units for the returned points; supported values are ``"m"`` and
            ``"mm"``.

        Returns
        -------
        numpy.ndarray
            Scaled design points as a floating-point array.
        """
        if units == "m":
            scale = 1.0
        elif units == "mm":
            scale = 1000.0
        else:
            raise ValueError(f"Unsupported unit: {units!r}")

        return numpy.asarray(self.design_points, dtype=float) * scale

    def _iter_all_segments(
            self,
            points: numpy.ndarray,
            link_radius: float,
            joint_radius: float,
            add_tool_frame: bool,
    ) -> Iterator[tuple[numpy.ndarray, numpy.ndarray, float]]:
        """Yield all cylindrical segments for the mechanism and tool frame.

        Parameters
        ----------
        points
            Scaled design points as an (N,3) array.
        link_radius
            Radius for link cylinders.
        joint_radius
            Radius for joint cylinders.
        add_tool_frame
            Whether to include tool-frame segments.

        Yields
        ------
        tuple
            Tuples of the form (p0, p1, radius) describing each cylindrical
            segment.
        """
        yield from self._iter_mechanism_segments(points, link_radius, joint_radius)

        if add_tool_frame:
            yield from self._iter_tool_segments(points, link_radius)

    def _iter_mechanism_segments(
            self,
            points: numpy.ndarray,
            link_radius: float,
            joint_radius: float,
    ) -> Iterator[tuple[numpy.ndarray, numpy.ndarray, float]]:
        """Yield cylindrical segments representing links and joint cylinders.

        Parameters
        ----------
        points
            Scaled design points as an (N,3) array.
        link_radius
            Radius for link cylinders.
        joint_radius
            Radius for joint cylinders.

        Yields
        ------
        tuple
            (p0, p1, radius) for each mechanism segment.
        """
        for i in range(self.num_joints):
            yield points[2 * i], points[2 * i + 1], joint_radius
            yield points[2 * i + 1], points[2 * i + 2], link_radius

    def _iter_tool_segments(
            self,
            points: numpy.ndarray,
            link_radius: float,
    ) -> Iterator[tuple[numpy.ndarray, numpy.ndarray, float]]:
        """Yield cylindrical segments for the optional tool-frame geometry.

        Parameters
        ----------
        points
            Scaled design points as an (N,3) array.
        link_radius
            Radius for link cylinders.

        Yields
        ------
        tuple
            (p0, p1, radius) tuples describing tool-frame cylinders.
        """
        tool_origin = numpy.zeros(3)
        tool_axes = numpy.eye(3)

        idx = len(points) // 2
        pt0 = points[idx]
        pt1 = points[idx - 1]

        mid_point = (pt0 + pt1) / 2

        length_tool_link = numpy.linalg.norm(pt1 - tool_origin)

        yield tool_origin, mid_point, link_radius
        # yield tool_origin, pt1, link_radius / 2

        for axis in tool_axes:
            yield tool_origin, axis * length_tool_link * 0.2, link_radius / 2

    @staticmethod
    def _segment_direction_and_length(p0: numpy.ndarray,
                                      p1: numpy.ndarray,
                                      tol: float = 1e-9,
                                      ) -> tuple[numpy.ndarray | None, float]:
        """Compute a unit direction vector and length between two points.

        Parameters
        ----------
        p0, p1
            Endpoint coordinates (array-like).
        tol, optional
            Tolerance below which the segment is considered degenerate.

        Returns
        -------
        tuple
            ``(direction, length)`` where ``direction`` is a unit vector or
            ``None`` when the segment is degenerate, and ``length`` is the
            Euclidean distance between the points.
        """
        p0 = numpy.asarray(p0, dtype=float)
        p1 = numpy.asarray(p1, dtype=float)

        vec = p1 - p0
        length = numpy.linalg.norm(vec)

        if length < tol:
            return None, 0.0

        return vec / length, length

    @staticmethod
    def _trimesh_cylinder(p0, p1, radius):
        """Create a trimesh cylinder mesh between two points.

        Parameters
        ----------
        p0, p1
            Endpoint coordinates of the cylinder axis.
        radius
            Cylinder radius.

        Returns
        -------
        trimesh.Trimesh or None
            The created cylinder mesh, or ``None`` if the segment is degenerate.
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
        cylinder.apply_translation((numpy.asarray(p0) + numpy.asarray(p1)) / 2)
        return cylinder

    @staticmethod
    def _build123d_cylinder(p0, p1, radius, build123d):
        """Create a build123d cylinder solid between two points.

        Parameters
        ----------
        p0, p1
            Endpoint coordinates of the cylinder axis.
        radius
            Cylinder radius.
        build123d
            The imported ``build123d`` module used for solid construction.

        Returns
        -------
        build123d.Solid or None
            A cylinder solid or ``None`` if the segment is degenerate.
        """
        direction, length = LinkageCAD._segment_direction_and_length(p0, p1)
        if direction is None:
            return None

        cyl = build123d.Cylinder(radius=radius, height=length)

        z_axis = numpy.array([0.0, 0.0, 1.0])
        axis = numpy.cross(z_axis, direction)
        axis_norm = numpy.linalg.norm(axis)

        if axis_norm > 1e-9:
            axis /= axis_norm
            angle = numpy.degrees(
                numpy.arccos(numpy.clip(numpy.dot(z_axis, direction), -1.0, 1.0))
            )
            cyl = cyl.rotate(
                build123d.Axis((0, 0, 0), build123d.Vector(*axis)),
                angle,
            )

        midpoint = (numpy.asarray(p0) + numpy.asarray(p1)) / 2
        return cyl.locate(build123d.Location(tuple(midpoint)))