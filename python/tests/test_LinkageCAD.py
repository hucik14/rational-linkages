import sys
from unittest.mock import MagicMock, patch
import numpy
import pytest
from rational_linkages.LinkageCAD import LinkageCAD
from rational_linkages.models import bennett_ark24
# ---------------------------------------------------------------------------
# Module-level fixtures (all based on the real bennett_ark24 4R mechanism)
#
# bennett_ark24 is a closed 4R linkage.  get_design_points() returns 9 points
# (8 unique joint-axis samples + the first point repeated to close the loop):
#   shape (9, 3)  →  num_joints = (9-1)//2 = 4
# ---------------------------------------------------------------------------
@pytest.fixture()
def mechanism():
    """Real bennett_ark24 RationalMechanism instance."""
    return bennett_ark24()
@pytest.fixture()
def design_points(mechanism):
    """Design points from the bennett_ark24 model – shape (9, 3)."""
    return mechanism.get_design_points()
@pytest.fixture()
def cad(design_points):
    """LinkageCAD built from the real bennett_ark24 design points."""
    return LinkageCAD(design_points)
# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_design_points_stored_as_ndarray(self, cad, design_points):
        assert isinstance(cad.design_points, numpy.ndarray)
    def test_design_points_values(self, cad, design_points):
        numpy.testing.assert_array_equal(cad.design_points, design_points)
    def test_design_points_dtype_float(self, cad):
        assert cad.design_points.dtype == numpy.float64
    def test_design_points_shape(self, design_points):
        """4-joint closed loop → 2*4 + 1 = 9 rows."""
        assert design_points.shape == (9, 3)
    def test_design_points_are_finite(self, design_points):
        assert numpy.all(numpy.isfinite(design_points))
    def test_loop_is_closed(self, design_points):
        """get_design_points appends the first point; first row must equal last."""
        numpy.testing.assert_array_almost_equal(design_points[0], design_points[-1])
    def test_tool_default_none(self, cad):
        assert cad.tool is None
    def test_tool_can_be_set(self, design_points):
        tool = object()
        lc = LinkageCAD(design_points, tool=tool)
        assert lc.tool is tool
    def test_num_joints_is_four(self, cad):
        """Bennett is a 4R mechanism → 4 joints."""
        assert cad.num_joints == 4
    def test_accepts_numpy_array(self, design_points):
        arr = numpy.asarray(design_points)
        lc = LinkageCAD(arr)
        numpy.testing.assert_array_equal(lc.design_points, arr)
# ---------------------------------------------------------------------------
# _scaled_points
# ---------------------------------------------------------------------------
class TestScaledPoints:
    def test_meters_returns_same_values(self, cad, design_points):
        scaled = cad._scaled_points(units="m")
        numpy.testing.assert_array_almost_equal(scaled, design_points)
    def test_millimeters_returns_scaled_values(self, cad, design_points):
        scaled = cad._scaled_points(units="mm")
        numpy.testing.assert_array_almost_equal(scaled, design_points * 1000.0)
    def test_default_unit_is_meters(self, cad, design_points):
        scaled = cad._scaled_points()
        numpy.testing.assert_array_almost_equal(scaled, design_points)
    def test_invalid_unit_raises_value_error(self, cad):
        with pytest.raises(ValueError, match="Unsupported unit"):
            cad._scaled_points(units="cm")
    def test_returns_ndarray(self, cad):
        assert isinstance(cad._scaled_points(units="m"), numpy.ndarray)
    def test_mm_points_are_1000x_larger(self, cad):
        pts_m = cad._scaled_points(units="m")
        pts_mm = cad._scaled_points(units="mm")
        numpy.testing.assert_array_almost_equal(pts_mm, pts_m * 1000.0)
    def test_mm_segment_lengths_greater_than_1mm(self, cad):
        """Real bennett segments in mm must all be > 1 mm (sanity check)."""
        points = cad._scaled_points(units="mm")
        for p0, p1, _ in cad._iter_mechanism_segments(points, 5.0, 10.0):
            _, length = LinkageCAD._segment_direction_and_length(p0, p1)
            assert length > 1.0, f"Suspiciously short segment: {length} mm"
# ---------------------------------------------------------------------------
# _segment_direction_and_length  (pure geometry – no mechanism dependency)
# ---------------------------------------------------------------------------
class TestSegmentDirectionAndLength:
    def test_normal_segment_length(self):
        p0 = numpy.array([0.0, 0.0, 0.0])
        p1 = numpy.array([3.0, 4.0, 0.0])
        _, length = LinkageCAD._segment_direction_and_length(p0, p1)
        assert pytest.approx(length) == 5.0
    def test_normal_segment_direction_is_unit_vector(self):
        p0 = numpy.array([0.0, 0.0, 0.0])
        p1 = numpy.array([1.0, 0.0, 0.0])
        direction, _ = LinkageCAD._segment_direction_and_length(p0, p1)
        assert pytest.approx(numpy.linalg.norm(direction)) == 1.0
    def test_normal_segment_direction_values(self):
        p0 = numpy.array([0.0, 0.0, 0.0])
        p1 = numpy.array([0.0, 0.0, 5.0])
        direction, _ = LinkageCAD._segment_direction_and_length(p0, p1)
        numpy.testing.assert_array_almost_equal(direction, [0.0, 0.0, 1.0])
    def test_degenerate_same_points_returns_none(self):
        p0 = numpy.array([1.0, 2.0, 3.0])
        p1 = numpy.array([1.0, 2.0, 3.0])
        direction, length = LinkageCAD._segment_direction_and_length(p0, p1)
        assert direction is None
        assert length == 0.0
    def test_degenerate_below_tolerance_returns_none(self):
        p0 = numpy.array([0.0, 0.0, 0.0])
        p1 = numpy.array([0.0, 0.0, 1e-12])
        direction, _ = LinkageCAD._segment_direction_and_length(p0, p1)
        assert direction is None
    def test_custom_tolerance(self):
        p0 = numpy.array([0.0, 0.0, 0.0])
        p1 = numpy.array([0.0, 0.0, 0.5])
        direction, _ = LinkageCAD._segment_direction_and_length(p0, p1, tol=1.0)
        assert direction is None
    def test_accepts_list_input(self):
        direction, length = LinkageCAD._segment_direction_and_length(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
        )
        assert direction is not None
        assert pytest.approx(length) == 1.0
# ---------------------------------------------------------------------------
# _iter_mechanism_segments
# ---------------------------------------------------------------------------
class TestIterMechanismSegments:
    def test_yields_correct_number_of_segments(self, cad):
        """4-joint Bennett → 4 joints × 2 segments = 8 mechanism segments."""
        points = cad._scaled_points()
        segments = list(cad._iter_mechanism_segments(points, 0.005, 0.01))
        assert len(segments) == 8
    def test_joint_radius_used_for_even_segments(self, cad):
        points = cad._scaled_points()
        joint_r, link_r = 0.05, 0.02
        segments = list(cad._iter_mechanism_segments(points, link_r, joint_r))
        radii = [r for _, _, r in segments]
        for idx in range(0, len(radii), 2):   # 0, 2, 4, 6
            assert radii[idx] == pytest.approx(joint_r)
    def test_link_radius_used_for_odd_segments(self, cad):
        points = cad._scaled_points()
        joint_r, link_r = 0.05, 0.02
        segments = list(cad._iter_mechanism_segments(points, link_r, joint_r))
        radii = [r for _, _, r in segments]
        for idx in range(1, len(radii), 2):   # 1, 3, 5, 7
            assert radii[idx] == pytest.approx(link_r)
    def test_radii_alternate_across_all_segments(self, cad):
        """All 8 segments must alternate joint/link radius correctly."""
        points = cad._scaled_points()
        joint_r, link_r = 0.02, 0.01
        segments = list(cad._iter_mechanism_segments(points, link_r, joint_r))
        for idx, (_, _, radius) in enumerate(segments):
            expected = joint_r if idx % 2 == 0 else link_r
            assert radius == pytest.approx(expected)
    def test_segment_start_points(self, cad):
        """First segment must run from design_points[0] to design_points[1]."""
        points = cad._scaled_points()
        segments = list(cad._iter_mechanism_segments(points, 0.005, 0.01))
        p0_first, p1_first, _ = segments[0]
        numpy.testing.assert_array_equal(p0_first, points[0])
        numpy.testing.assert_array_equal(p1_first, points[1])
    def test_all_segments_non_degenerate(self, cad):
        """Every mechanism segment of the real bennett model must be non-degenerate."""
        points = cad._scaled_points()
        for p0, p1, _ in cad._iter_mechanism_segments(points, 0.005, 0.01):
            direction, length = LinkageCAD._segment_direction_and_length(p0, p1)
            assert direction is not None, f"Degenerate segment: {p0} → {p1}"
            assert length > 0.0
# ---------------------------------------------------------------------------
# _iter_tool_segments
# ---------------------------------------------------------------------------
class TestIterToolSegments:
    def test_yields_four_segments(self, cad):
        """Tool frame = 1 main link + 3 axis indicators."""
        points = cad._scaled_points()
        segments = list(cad._iter_tool_segments(points, 0.005))
        assert len(segments) == 4
    def test_all_tool_segments_start_at_origin(self, cad):
        points = cad._scaled_points()
        for p0, _, _ in cad._iter_tool_segments(points, 0.005):
            numpy.testing.assert_array_almost_equal(p0, [0.0, 0.0, 0.0])
    def test_axis_segments_use_half_link_radius(self, cad):
        points = cad._scaled_points()
        link_radius = 0.02
        segments = list(cad._iter_tool_segments(points, link_radius))
        for _, _, radius in segments[1:]:
            assert radius == pytest.approx(link_radius / 2)
    def test_main_segment_uses_full_link_radius(self, cad):
        points = cad._scaled_points()
        link_radius = 0.02
        segments = list(cad._iter_tool_segments(points, link_radius))
        _, _, radius = segments[0]
        assert radius == pytest.approx(link_radius)
# ---------------------------------------------------------------------------
# _iter_all_segments
# ---------------------------------------------------------------------------
class TestIterAllSegments:
    def test_without_tool_frame(self, cad):
        """Without tool: only the 8 mechanism segments."""
        points = cad._scaled_points()
        segments = list(cad._iter_all_segments(points, 0.005, 0.01,
                                               add_tool_frame=False))
        assert len(segments) == 8
    def test_with_tool_frame_total(self, cad):
        """With tool: 8 mechanism + 4 tool = 12 total."""
        points = cad._scaled_points()
        segments = list(cad._iter_all_segments(points, 0.005, 0.01,
                                               add_tool_frame=True))
        assert len(segments) == 12
    def test_with_tool_frame_adds_exactly_four(self, cad):
        points = cad._scaled_points()
        no_tool = list(cad._iter_all_segments(points, 0.005, 0.01,
                                              add_tool_frame=False))
        with_tool = list(cad._iter_all_segments(points, 0.005, 0.01,
                                                add_tool_frame=True))
        assert len(with_tool) == len(no_tool) + 4
    def test_all_segments_are_tuples_of_three(self, cad):
        points = cad._scaled_points()
        for seg in cad._iter_all_segments(points, 0.005, 0.01,
                                          add_tool_frame=True):
            assert len(seg) == 3
# ---------------------------------------------------------------------------
# _fuse_solids  (pure mock – no mechanism dependency)
# ---------------------------------------------------------------------------
class TestFuseSolids:
    def test_single_solid_returned_unchanged(self):
        solid = MagicMock()
        result = LinkageCAD._fuse_solids([solid])
        assert result is solid
    def test_two_solids_fused(self):
        solid_a = MagicMock()
        solid_b = MagicMock()
        fused = MagicMock()
        solid_a.fuse.return_value = fused
        result = LinkageCAD._fuse_solids([solid_a, solid_b])
        solid_a.fuse.assert_called_once_with(solid_b)
        assert result is fused
    def test_multiple_solids_fused_in_order(self):
        solids = [MagicMock() for _ in range(4)]
        for i in range(3):
            solids[i].fuse.return_value = solids[i + 1]
        result = LinkageCAD._fuse_solids(solids)
        solids[0].fuse.assert_called_once_with(solids[1])
        assert result is solids[3]
    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="No solids"):
            LinkageCAD._fuse_solids([])
# ---------------------------------------------------------------------------
# _build_link_solids
# ---------------------------------------------------------------------------
class TestBuildLinkSolids:
    @staticmethod
    def _make_solids(n):
        """Return n MagicMock solids with a deep-enough fuse chain."""
        solids = []
        for _ in range(n):
            m = MagicMock()
            m.fuse.return_value = MagicMock()
            m.fuse.return_value.fuse.return_value = MagicMock()
            m.fuse.return_value.fuse.return_value.fuse.return_value = MagicMock()
            solids.append(m)
        return solids
    def test_correct_number_of_links(self, cad):
        """Bennett 4R → 4 joints → 4 link solids."""
        solids = self._make_solids(8)   # 4 joints × 2 segments
        links = cad._build_link_solids(solids)
        assert len(links) == 4
    def test_wrong_number_of_solids_raises_value_error(self, cad):
        """Passing the wrong number of solids must raise ValueError."""
        solids = self._make_solids(5)   # expected 8 for a 4-joint linkage
        with pytest.raises(ValueError, match="Expected"):
            cad._build_link_solids(solids)
    def test_tool_attached_to_middle_link(self, cad):
        """For a 4-joint linkage the tool is fused into the link at index 2."""
        solids = self._make_solids(8)
        tool = MagicMock()
        tool.fuse.return_value = MagicMock()
        captured = []
        def mock_fuse(parts):
            captured.append(list(parts))
            result = MagicMock()
            result.fuse.return_value = MagicMock()
            return result
        with patch.object(LinkageCAD, '_fuse_solids', staticmethod(mock_fuse)):
            cad._build_link_solids(solids, tool=tool)
        middle_link_idx = cad.num_joints // 2   # 4 // 2 == 2
        assert tool in captured[middle_link_idx]
    def test_no_tool_each_link_has_three_parts(self, cad):
        """Without a tool every link is fused from exactly 3 cylinder solids."""
        captured = []
        def mock_fuse(parts):
            captured.append(list(parts))
            result = MagicMock()
            result.fuse.return_value = MagicMock()
            return result
        solids = self._make_solids(8)
        with patch.object(LinkageCAD, '_fuse_solids', staticmethod(mock_fuse)):
            cad._build_link_solids(solids, tool=None)
        for parts in captured:
            assert len(parts) == 3
# ---------------------------------------------------------------------------
# Export methods – ImportError when optional dependencies are missing
# ---------------------------------------------------------------------------
class TestExportImportErrors:
    def test_export_single_mesh_missing_trimesh(self, cad):
        with patch.dict(sys.modules, {"trimesh": None}):
            with pytest.raises(ImportError, match="trimesh"):
                cad.export_single_mesh()
    def test_export_single_solid_missing_build123d(self, cad):
        with patch.dict(sys.modules, {"build123d": None}):
            with pytest.raises(ImportError, match="[Bb]uild123d"):
                cad.export_single_solid()
    def test_export_solids_missing_build123d(self, cad):
        with patch.dict(sys.modules, {"build123d": None}):
            with pytest.raises(ImportError, match="[Bb]uild123d"):
                cad.export_solids()
    def test_export_single_mesh_via_mechanism_missing_trimesh(self, mechanism):
        """RationalMechanism.export_single_mesh must raise when trimesh is absent."""
        with patch.dict(sys.modules, {"trimesh": None}):
            with pytest.raises(ImportError, match="trimesh"):
                mechanism.export_single_mesh()
    def test_export_single_solid_via_mechanism_missing_build123d(self, mechanism):
        """RationalMechanism.export_single_solid must raise when build123d is absent."""
        with patch.dict(sys.modules, {"build123d": None}):
            with pytest.raises(ImportError, match="[Bb]uild123d"):
                mechanism.export_single_solid()
# ---------------------------------------------------------------------------
# Tests inspired by docs/source/examples/d_t_cad_export.py
#
# m.export_single_mesh(scale=1.0, link_diameter=0.01, joint_diameter=0.02, ...)
# m.export_solids(units="mm", link_diameter=10, joint_diameter=20, ...)
# ---------------------------------------------------------------------------
class TestDocsExportParams:
    """Verify that the exact parameters from the docs export example produce
    geometrically correct segment data, without requiring trimesh / build123d."""
    # Exact values used in d_t_cad_export.py
    LINK_DIAMETER = 0.01    # metres  → radius 0.005
    JOINT_DIAMETER = 0.02   # metres  → radius 0.010
    LINK_DIAMETER_MM = 10   # mm      → radius 5
    JOINT_DIAMETER_MM = 20  # mm      → radius 10
    # --- export_single_mesh parameters (metres) ---
    def test_mesh_export_joint_radius(self, cad):
        """Joint cylinders carry radius = joint_diameter / 2 = 0.010 m."""
        points = cad._scaled_points(units="m")
        link_r = self.LINK_DIAMETER / 2
        joint_r = self.JOINT_DIAMETER / 2
        segments = list(cad._iter_mechanism_segments(points, link_r, joint_r))
        for idx, (_, _, r) in enumerate(segments):
            if idx % 2 == 0:
                assert r == pytest.approx(joint_r)
    def test_mesh_export_link_radius(self, cad):
        """Link cylinders carry radius = link_diameter / 2 = 0.005 m."""
        points = cad._scaled_points(units="m")
        link_r = self.LINK_DIAMETER / 2
        joint_r = self.JOINT_DIAMETER / 2
        segments = list(cad._iter_mechanism_segments(points, link_r, joint_r))
        for idx, (_, _, r) in enumerate(segments):
            if idx % 2 == 1:
                assert r == pytest.approx(link_r)
    def test_mesh_export_all_segments_count(self, cad):
        """export_single_mesh with add_tool_frame=True yields 12 segments."""
        points = cad._scaled_points(units="m")
        segments = list(cad._iter_all_segments(
            points,
            link_radius=self.LINK_DIAMETER / 2,
            joint_radius=self.JOINT_DIAMETER / 2,
            add_tool_frame=True,
        ))
        assert len(segments) == 12
    def test_mesh_export_all_segments_non_degenerate(self, cad):
        """Every segment with docs-example params must be non-degenerate."""
        points = cad._scaled_points(units="m")
        for p0, p1, _ in cad._iter_all_segments(
            points,
            link_radius=self.LINK_DIAMETER / 2,
            joint_radius=self.JOINT_DIAMETER / 2,
            add_tool_frame=True,
        ):
            direction, length = LinkageCAD._segment_direction_and_length(p0, p1)
            assert direction is not None, f"Degenerate segment: {p0} → {p1}"
            assert length > 0.0
    # --- export_solids parameters (millimetres) ---
    def test_solids_export_mm_joint_radius(self, cad):
        """With units='mm', joint radius = joint_diameter_mm / 2 = 10 mm."""
        points = cad._scaled_points(units="mm")
        joint_r = self.JOINT_DIAMETER_MM / 2
        link_r = self.LINK_DIAMETER_MM / 2
        segments = list(cad._iter_mechanism_segments(points, link_r, joint_r))
        for idx, (_, _, r) in enumerate(segments):
            if idx % 2 == 0:
                assert r == pytest.approx(joint_r)
    def test_solids_export_mm_link_radius(self, cad):
        """With units='mm', link radius = link_diameter_mm / 2 = 5 mm."""
        points = cad._scaled_points(units="mm")
        joint_r = self.JOINT_DIAMETER_MM / 2
        link_r = self.LINK_DIAMETER_MM / 2
        segments = list(cad._iter_mechanism_segments(points, link_r, joint_r))
        for idx, (_, _, r) in enumerate(segments):
            if idx % 2 == 1:
                assert r == pytest.approx(link_r)
    def test_solids_export_mm_segment_lengths_reasonable(self, cad):
        """Segment lengths in mm must all be > 1 mm."""
        points = cad._scaled_points(units="mm")
        for p0, p1, _ in cad._iter_mechanism_segments(
            points, self.LINK_DIAMETER_MM / 2, self.JOINT_DIAMETER_MM / 2
        ):
            _, length = LinkageCAD._segment_direction_and_length(p0, p1)
            assert length > 1.0, f"Suspiciously short segment in mm: {length}"
    def test_solids_export_build_link_solids_four_links(self, cad):
        """export_solids must produce 4 link solids for the 4R Bennett."""
        def make_mock():
            m = MagicMock()
            m.fuse.return_value = MagicMock()
            m.fuse.return_value.fuse.return_value = MagicMock()
            m.fuse.return_value.fuse.return_value.fuse.return_value = MagicMock()
            return m
        links = cad._build_link_solids([make_mock() for _ in range(8)])
        assert len(links) == 4
# ---------------------------------------------------------------------------
# End-to-end file-creation tests (skipped when optional backends are absent)
# Mirrors exactly the two calls in docs/source/examples/d_t_cad_export.py
# ---------------------------------------------------------------------------
class TestExportFileCreation:
    """Full-pipeline export tests writing real files to a tmp directory.
    Tests are auto-skipped when trimesh / build123d are not installed.
    """
    # --- STL mesh via trimesh (mirrors first call in d_t_cad_export.py) ---
    def test_export_single_mesh_creates_file(self, mechanism, tmp_path):
        pytest.importorskip("trimesh", reason="trimesh not installed")
        out = tmp_path / "mechanism.stl"
        mechanism.export_single_mesh(
            scale=1.0,
            link_diameter=0.01,
            joint_diameter=0.02,
            add_tool_frame=True,
            file_name=str(out),
        )
        assert out.exists(), "STL file was not created"
        assert out.stat().st_size > 0, "STL file is empty"
    def test_export_single_mesh_has_stl_extension(self, mechanism, tmp_path):
        pytest.importorskip("trimesh", reason="trimesh not installed")
        out = tmp_path / "test_out.stl"
        mechanism.export_single_mesh(file_name=str(out))
        assert out.suffix.lower() == ".stl"
    def test_export_single_mesh_without_tool_frame(self, mechanism, tmp_path):
        """add_tool_frame=False should still produce a valid STL."""
        pytest.importorskip("trimesh", reason="trimesh not installed")
        out = tmp_path / "no_tool.stl"
        mechanism.export_single_mesh(
            scale=1.0,
            link_diameter=0.01,
            joint_diameter=0.02,
            add_tool_frame=False,
            file_name=str(out),
        )
        assert out.exists()
        assert out.stat().st_size > 0
    def test_export_single_mesh_custom_diameters(self, mechanism, tmp_path):
        """Different diameter values should still produce a valid STL."""
        pytest.importorskip("trimesh", reason="trimesh not installed")
        out = tmp_path / "custom.stl"
        mechanism.export_single_mesh(
            scale=1.0,
            link_diameter=0.005,
            joint_diameter=0.015,
            add_tool_frame=True,
            file_name=str(out),
        )
        assert out.exists()
        assert out.stat().st_size > 0
    # --- STEP solid via build123d (mirrors second call in d_t_cad_export.py) ---
    def test_export_single_solid_creates_file(self, mechanism, tmp_path):
        pytest.importorskip("build123d", reason="build123d not installed")
        out = tmp_path / "mechanism.step"
        mechanism.export_single_solid(
            units="mm",
            link_diameter=10,
            joint_diameter=20,
            add_tool_frame=True,
            file_name=str(out),
        )
        assert out.exists(), "STEP file was not created"
        assert out.stat().st_size > 0, "STEP file is empty"
    def test_export_solids_creates_file(self, mechanism, tmp_path):
        """export_solids (multi-part STEP) — exact replica of the docs example."""
        pytest.importorskip("build123d", reason="build123d not installed")
        out = tmp_path / "mechanism_parts.step"
        mechanism.export_solids(
            units="mm",
            link_diameter=10,
            joint_diameter=20,
            add_tool_frame=True,
            file_name=str(out),
        )
        assert out.exists(), "Multi-part STEP file was not created"
        assert out.stat().st_size > 0, "Multi-part STEP file is empty"
