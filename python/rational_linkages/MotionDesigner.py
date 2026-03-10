import sys
import struct
import os
import numpy as np

from typing import Union
from warnings import warn

from .DualQuaternion import DualQuaternion
from .MotionInterpolation import MotionInterpolation
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve
from .RationalMechanism import RationalMechanism
from .TransfMatrix import TransfMatrix

# Try importing GUI components
try:
    import pyqtgraph.opengl as gl
    from PyQt6 import QtCore, QtWidgets
    from .PlotterPyqtgraph import (
        FramePlotHelper,
        InteractivePlotterWidget,
        PlotterPyqtgraph,
    )
except (ImportError, OSError):
    warn("Failed to import OpenGL or PyQt6. If you expect interactive GUI to work, "
         "please check the package installation.")

    gl = None
    QtCore = None
    QtWidgets = None
    FramePlotHelper = None
    InteractivePlotterWidget = None
    PlotterPyqtgraph = None


class MotionDesigner:
    """
    Main application class for the motion designer.

    Encapsulates the QApplication and the MotionDesigner widget.

    :examples:

    Run motion designer without initial points or poses:

    .. testcode:: [motiondesigner_ex1]

        from rational_linkages import MotionDesigner

        d = MotionDesigner(method='quadratic_from_poses')
        d.show()

    .. testoutput:: [motiondesigner_ex1]
        :hide:

        Closing the window... generated points for interpolation:
        [1, 0, 0, 0, 0, 0, 0, 0]
        [ 1.          , -0.207522406 , -0.0333866662, -0.0691741237, -0.0625113682, -0.141265791 , -0.4478576802, -0.2637268902]
        [ 1.          ,  0.2333739522, -0.0427838517,  0.0777914503, -0.0839342318,  0.2991396249,  0.2980046603,  0.345444421 ]

    .. testcleanup:: [motiondesigner_ex1]

        del d, MotionDesigner

    Run motion designer with initial points:

    .. code-block:: python

        # NOT TESTED

        from rational_linkages import MotionDesigner, PointHomogeneous


        chosen_points = [PointHomogeneous(pt) for pt in
                         [
                             [ 1.  , -0.2 ,  0.  ,  1.76],
                             [1., 1., 1., 2.],
                             [ 1.,  3., -3.,  1.],
                             [ 1.,  2., -4.,  1.],
                             [ 1., -2., -2.,  2.]
                         ]]

        d = MotionDesigner(method='quadratic_from_points', initial_points_or_poses=chosen_points)
        d.show()


    """
    def __init__(self,
                 method: str,
                 initial_points_or_poses: list[Union[PointHomogeneous, DualQuaternion]] = None,
                 arrows_length: float = 1.0,
                 sliders_range: int = 10,
                 show_grid: bool = True,
                 white_background: bool = False,
                 preview_mechanism: bool = False):
        """
        Initialize the application with the motion designer widget.

        :param str method: The method to use for interpolation, supported values are
            'cubic_from_points', 'quadratic_from_points', and 'quadratic_from_poses'.
        :param list[PointHomogeneous] or list[DualQuaternion] initial_points_or_poses:
            The initial points or poses to use for the motion curve.
        :param float arrows_length: The length of the arrows for the poses.
        :param int sliders_range: The plus/minus range for the sliders in the control
            panel (default is 10, i.e. from -10 to 10 units).
        :param bool show_grid: Whether to show a grid in the 3D view (default is True).
        :param bool white_background: Whether to use a white background for the plot.
        :param bool preview_mechanism: Whether to show a preview of the mechanism. The
            mechanism is visualized as shortest polyline connecting the axes, in
            default configuration. The computational time may be significant, leading
            to lagging rendering.
        """
        if method not in ['cubic_from_points',
                          'cubic_from_poses',
                          'quadratic_from_points',
                          'quadratic_from_poses',]:
            raise ValueError("Invalid method for motion designer.")

        if QtWidgets is None:
            raise RuntimeError(
                "Qt6 / pyqtgraph are not available. GUI MotionDesigner cannot be used."
            )

        # Reuse an existing QApplication if there is one (e.g. when using run())
        existing_app = QtWidgets.QApplication.instance()
        self.app = existing_app or QtWidgets.QApplication(sys.argv)

        self.window = MotionDesignerWidget(method=method,
                                           initial_pts=initial_points_or_poses,
                                           arrows_length=arrows_length,
                                           sliders_range=sliders_range,
                                           show_grid=show_grid,
                                           white_background=white_background,
                                           preview_mechanism=preview_mechanism)

    @classmethod
    def start(cls,
              initial_points_or_poses: list[Union[PointHomogeneous, DualQuaternion]] = None,
              arrows_length: float = 1.0,
              white_background: bool = False,
              sliders_range: int = 10,
              show_grid: bool = True,
              preview_mechanism: bool = False):
        """
        Start Motion Designer with GUI method options.

        :param list[PointHomogeneous] or list[DualQuaternion] initial_points_or_poses:
            The initial points or poses to use for the motion curve.
        :param float arrows_length: The length of the arrows for the poses.
        :param bool white_background: Whether to use a white background for the plot.
        :param int sliders_range: The plus/minus range for the sliders in the control
            panel (default is 10, i.e. from -10 to 10 units).
        :param bool show_grid: Whether to show a grid in the 3D view (default is True).
        :param bool preview_mechanism: Whether to show a preview of the mechanism. The
            mechanism is visualized as shortest polyline connecting the axes, in
            default configuration. The computational time may be significant, leading
            to lagging rendering.
        """
        if QtWidgets is None:
            raise RuntimeError(
                "Qt6 / pyqtgraph are not available. GUI MotionDesigner cannot be used."
            )

        # Make sure there is a QApplication
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        # ----- Build the selection dialog -----
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("MotionDesigner – choose input method")

        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel("Choose the input methodology:")
        layout.addWidget(label)

        combo = QtWidgets.QComboBox(dialog)

        # (internal_key, human_readable_description)
        methods = [
            ("quadratic_from_poses",  "3 poses → quadratic motion (4-bar linkage)"),
            ("cubic_from_poses",      "4 poses → cubic motion (6-bar linkage)"),
            ("quadratic_from_points", "5 points → quadratic motion (4-bar linkage)"),
            ("cubic_from_points",     "7 points → cubic motion (6-bar linkage)"),
        ]

        for key, desc in methods:
            combo.addItem(desc, userData=key)

        layout.addWidget(combo)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(button_box)

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        # modal dialog: runs its own event loop until closed
        result = dialog.exec()
        if result != QtWidgets.QDialog.DialogCode.Accepted:
            # user cancelled; do nothing
            return None

        chosen_method = combo.currentData()
        if chosen_method is None:
            # very defensive; should not happen
            return None

        # ----- Create and run the actual MotionDesigner -----
        designer = cls(method=chosen_method,
                       initial_points_or_poses=initial_points_or_poses,
                       arrows_length=arrows_length,
                       sliders_range=sliders_range,
                       show_grid=show_grid,
                       white_background=white_background,
                       preview_mechanism=preview_mechanism)
        designer.show()
        return designer

    def plot(self, *args, **kwargs):
        """
        Plot the given objects in the motion designer widget.

        :param args: The objects to plot.
        :param kwargs: Additional keyword arguments for the plotter.
        """
        self.window.plotter.plot(*args, **kwargs)

    def show(self):
        """
        Run the application, showing the motion designer widget.
        """
        self.window.show()
        try:
            self.app.exec()
        except SystemExit:
            pass

    def add_mesh_from_stl(self,
                          path: str,
                          scale: float = 1.0,
                          color: tuple = (0.4, 0.4, 0.4, 0.1),
                          name: str | None = None,
                          smooth: bool = False,
                          max_faces: int = None,
                          weld_tol: float = 1e-8) -> object:
        """
        Add a mesh from an STL file to the view.

        Load an STL file (ASCII or binary), produce (vertices, faces) arrays,
        optionally reduce triangle count for performance, and add it to the view
        via self.add_mesh\(\).

        :param str path: The file path to the STL file.
        :param float scale: Scale factor for mesh vertices.
        :param tuple color: RGBA color for the mesh (default is light gray with
            some transparency).
        :param str | None name: Optional name for the mesh.
        :param bool smooth: Whether to use smooth shading (default is False).
        :param int max_faces: If set, the maximum number of faces to display.
        :param float weld_tol: Tolerance for welding duplicate vertices (default
            is 1e-8).

        :return: The GL item created by self.add_mesh.
        :rtype: object
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        # read raw bytes
        with open(path, "rb") as f:
            data = f.read()

        verts = []
        faces = []

        # detect binary STL reliably by expected size
        is_binary = False
        if len(data) >= 84:
            try:
                tri_count = struct.unpack_from("<I", data, 80)[0]
                expected = 84 + tri_count * 50
                if expected == len(data):
                    is_binary = True
            except struct.error:
                is_binary = False

        if is_binary:
            # parse binary STL
            tri_count = struct.unpack_from("<I", data, 80)[0]
            offset = 84
            for i in range(tri_count):
                # 12 floats: normal(3) + v1(3) + v2(3) + v3(3) => 48 bytes, plus 2 byte attr
                vals = struct.unpack_from("<12f", data, offset)
                offset += 48
                # skip 2 byte attribute
                offset += 2
                # vertices are vals[3:12]
                v1 = (vals[3], vals[4], vals[5])
                v2 = (vals[6], vals[7], vals[8])
                v3 = (vals[9], vals[10], vals[11])
                base = len(verts)
                # scale vertices
                v1 = (v1[0] * scale, v1[1] * scale, v1[2] * scale)
                v2 = (v2[0] * scale, v2[1] * scale, v2[2] * scale)
                v3 = (v3[0] * scale, v3[1] * scale, v3[2] * scale)
                verts.extend([v1, v2, v3])
                faces.append([base, base + 1, base + 2])
        else:
            # parse ASCII STL
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                raise RuntimeError("Unable to decode ASCII STL")
            lines = text.splitlines()
            current_face = []
            for ln in lines:
                ln = ln.strip()
                if ln.lower().startswith("vertex"):
                    parts = ln.split()
                    if len(parts) >= 4:
                        try:
                            x = float(parts[1]) * scale
                            y = float(parts[2]) * scale
                            z = float(parts[3]) * scale
                            verts.append((x, y, z))
                            current_face.append(len(verts) - 1)
                            if len(current_face) == 3:
                                faces.append(current_face)
                                current_face = []
                        except ValueError:
                            continue
                elif ln.lower().startswith("endfacet"):
                    current_face = []

        if len(faces) == 0 or len(verts) == 0:
            raise RuntimeError("No triangles parsed from STL")

        verts = np.array(verts, dtype=float)
        faces = np.array(faces, dtype=int)
        if (max_faces is None) and (faces.shape[0]) > 200000:
            warn("Too many faces may lead to very slow rendering. Consider reducing it using max_faces=200000 parameter.")
            print('number of faces: {}'.format(faces.shape[0]))

        # optional subsample triangles for performance
        if (max_faces is not None) and (faces.shape[0] > max_faces):
            idx = np.linspace(0, faces.shape[0] - 1, max_faces, dtype=int)
            faces = faces[idx]

        # weld duplicate vertices within weld_tol
        decimals = max(0, int(-np.log10(weld_tol))) if weld_tol > 0 else 8
        key_map = {}
        unique_verts = []
        remap = np.empty(len(verts), dtype=int)
        for i, v in enumerate(verts):
            key = (round(float(v[0]), decimals),
                   round(float(v[1]), decimals),
                   round(float(v[2]), decimals))
            if key in key_map:
                remap[i] = key_map[key]
            else:
                idx_new = len(unique_verts)
                key_map[key] = idx_new
                unique_verts.append((v[0], v[1], v[2]))
                remap[i] = idx_new

        unique_verts = np.array(unique_verts, dtype=float)
        faces = remap[faces]

        # remove unused vertices and remap indices (compact the vertex array)
        used = np.unique(faces.reshape(-1))
        new_idx = -np.ones(unique_verts.shape[0], dtype=int)
        new_idx[used] = np.arange(used.shape[0], dtype=int)
        vertices_final = unique_verts[used]
        faces_final = new_idx[faces]

        # delegate to existing add_mesh
        return self.window.add_mesh(vertices_final,
                                    faces_final,
                                    color=color,
                                    name=name,
                                    smooth=smooth)

if QtWidgets is not None:
    class MotionDesignerWidget(QtWidgets.QWidget):
        """
        Interactive plotting widget for designing motion curves with interpolated points.

        A widget that displays a 3D view of a motion curve and control points,
        plus a side panel with controls for selecting and modifying one of the
        control points (p0 to p6). Moving the sliders adjusts the x, y, and z
        coordinates of the selected control point, which then updates the curve.
        """
        def __init__(self,
                     method: str = 'cubic_from_points',
                     initial_pts: Union[list[PointHomogeneous], list[DualQuaternion]] = None,
                     parent = None,
                     sliders_range: int = 10,
                     show_grid: bool = True,
                     steps: int = 1000,
                     interval: tuple = (0, 1),
                     arrows_length: float = 1.0,
                     white_background: bool = False,
                     preview_mechanism: bool = False):
            """
            Initialize the motion designer widget.
            """
            super().__init__(parent)
            self.setMinimumSize(900, 600)

            self.white_background = white_background
            self.preview_mechanism = preview_mechanism
            self.points = self._initialize_points(method, initial_pts)
            self.method = method
            self.arrows_length = arrows_length
            self.mi = MotionInterpolation()

            grid_size = sliders_range * 2

            # an instance of Pyqtgraph-based plotter
            self.plotter = PlotterPyqtgraph(steps=steps,
                                            interval=interval,
                                            arrows_length=self.arrows_length,
                                            white_background=self.white_background,
                                            show_grid=show_grid,
                                            grid_size=grid_size)

            self.mechanism_plotter = []

            if self.white_background:
                self.render_mode = 'opaque'
            else:
                self.render_mode = 'additive'

            self.previous_rpy_sliders_values = []

            # array of control point coordinates (in 3D)
            if method == 'quadratic_from_points' or method == 'cubic_from_points':
                self.plotted_points = np.array([pt.normalized_in_3d()
                                                for pt in self.points])

                # interpolated points markers
                self.markers = gl.GLScatterPlotItem(pos=self.plotted_points,
                                                    color=(1, 0, 1, 1),
                                                    glOptions=self.render_mode,
                                                    size=10)
                self.plotter.widget.addItem(self.markers)

                for i, pt in enumerate(self.plotted_points):
                    self.plotter.widget.add_label(pt, f"p{i}")

            elif method == 'quadratic_from_poses' or method == 'cubic_from_poses':
                poses_arrays = [TransfMatrix(pt.dq2matrix()) for pt in self.points]
                self.plotted_poses = [FramePlotHelper(transform=tr,
                                                      width=10,
                                                      length=2 * self.arrows_length)
                                      for tr in poses_arrays]
                for i, pose in enumerate(self.plotted_poses):
                    pose.addToView(self.plotter.widget)
                    self.plotter.widget.add_label(pose, f"p{i}")
                    self.previous_rpy_sliders_values.append(pose.tr.rpy() * 100)

            if method == 'cubic_from_points' or method == 'cubic_from_poses':
                self.num_lines = 12
            elif method == 'quadratic_from_points' or method == 'quadratic_from_poses':
                self.num_lines = 8

            self.lines = None  # mechanism preview lines
            self.curve_path_vis = None  # path of motion curve
            self.curve_frames_vis = None  # poses of motion curve
            self.lambda_val = 0.0
            self.motion_family_idx = 0
            self.update_curve_vis()  # initial curve update

            ###################################
            # --- build the Control Panel --- #
            def create_separator():
                """
                Create a horizontal line separator (QFrame).
                """
                separator = QtWidgets.QFrame()
                separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
                return separator

            # combo box to select one of the points
            self.point_combo = QtWidgets.QComboBox()
            for i in range(1, len(self.points)):
                self.point_combo.addItem(f"Point {i}")
            self.point_combo.currentIndexChanged.connect(self.on_point_selection_changed)

            # sliders for adjusting x, y, and z
            self.slider_x = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.textbox_x = QtWidgets.QLineEdit()
            self.textbox_x.editingFinished.connect(
                lambda: self.on_textbox_changed(self.textbox_x.text(), self.slider_x)
            )
            self.slider_y = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.textbox_y = QtWidgets.QLineEdit()
            self.textbox_y.editingFinished.connect(
                lambda: self.on_textbox_changed(self.textbox_y.text(), self.slider_y)
            )
            self.slider_z = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.textbox_z = QtWidgets.QLineEdit()
            self.textbox_z.editingFinished.connect(
                lambda: self.on_textbox_changed(self.textbox_z.text(), self.slider_z)
            )
            # slider range
            for slider, textbox in [(self.slider_x, self.textbox_x),
                                    (self.slider_y, self.textbox_y),
                                    (self.slider_z, self.textbox_z)]:
                slider.setMinimum(int(-100 * sliders_range))
                slider.setMaximum(int(100 * sliders_range))
                slider.setSingleStep(1)
                slider.valueChanged.connect(self.on_slider_value_changed)

            if method == 'quadratic_from_poses' or method == 'cubic_from_poses':
                # sliders for adjusting roll, pitch, and yaw with textboxes
                self.slider_roll = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                self.textbox_roll = QtWidgets.QLineEdit()
                self.textbox_roll.editingFinished.connect(
                    lambda: self.on_textbox_changed(self.textbox_roll.text(), self.slider_roll)
                )
                self.slider_pitch = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                self.textbox_pitch = QtWidgets.QLineEdit()
                self.textbox_pitch.editingFinished.connect(
                    lambda: self.on_textbox_changed(self.textbox_pitch.text(), self.slider_pitch)
                )
                self.slider_yaw = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                self.textbox_yaw = QtWidgets.QLineEdit()
                self.textbox_yaw.editingFinished.connect(
                    lambda: self.on_textbox_changed(self.textbox_yaw.text(), self.slider_yaw)
                )

                self.slider_roll_prev = 0
                self.slider_pitch_prev = 0
                self.slider_yaw_prev = 0

                # slider range
                for slider, textbox in [(self.slider_roll, self.textbox_roll),
                                        (self.slider_pitch, self.textbox_pitch),
                                        (self.slider_yaw, self.textbox_yaw)]:
                    slider.setMinimum(int(-np.pi * 100))
                    slider.setMaximum(int(np.pi * 100))
                    slider.setSingleStep(1)
                    slider.valueChanged.connect(self.on_slider_value_changed)

            # slider for lambda of cubic curve with textbox
            if method == 'cubic_from_poses':
                self.slider_lambda = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                self.textbox_lambda = QtWidgets.QLineEdit()

                self.slider_lambda.setMinimum(int(-500))
                self.slider_lambda.setMaximum(int(500))
                self.slider_lambda.setSingleStep(1)

                self.slider_lambda.valueChanged.connect(self.on_lambda_slider_value_changed)
                self.textbox_lambda.editingFinished.connect(
                    lambda: self.on_lambda_textbox_changed(self.textbox_lambda.text(),
                                                           self.slider_lambda))

                # add button for swapping family
                self.swap_family_check_box = QtWidgets.QCheckBox(text="Swap family")
                self.motion_family_idx = 0
                self.swap_family_check_box.stateChanged.connect(self.on_swap_family_check_box_changed)
            else:
                self.slider_lambda = None
                self.swap_family_check_box = None
                self.textbox_lambda = None

            # add button for mechanism synthesis
            self.synthesize_button = QtWidgets.QPushButton("Mechanism")
            self.synthesize_button.clicked.connect(self.on_synthesize_button_clicked)

            # initially for the first point
            self.set_sliders_for_point(0)

            # --- layout the 3D view and control panel ---
            main_layout = QtWidgets.QHBoxLayout(self)
            # add plotter (stored in self.plotter.widget)
            main_layout.addWidget(self.plotter.widget, stretch=1)

            # Build a vertical control panel.
            control_panel = QtWidgets.QWidget()
            cp_layout = QtWidgets.QVBoxLayout(control_panel)

            cp_layout.addWidget(QtWidgets.QLabel("Select control point:"))
            cp_layout.addWidget(self.point_combo)
            cp_layout.addSpacing(10)

            cp_layout.addWidget(QtWidgets.QLabel("Adjust X:"))
            cp_layout.addWidget(self.slider_x)
            cp_layout.addWidget(self.textbox_x)
            cp_layout.addWidget(QtWidgets.QLabel("Adjust Y:"))
            cp_layout.addWidget(self.slider_y)
            cp_layout.addWidget(self.textbox_y)
            cp_layout.addWidget(QtWidgets.QLabel("Adjust Z:"))
            cp_layout.addWidget(self.slider_z)
            cp_layout.addWidget(self.textbox_z)
            if method == 'quadratic_from_poses' or method == 'cubic_from_poses':
                cp_layout.addSpacing(10)  # Add 10 pixels of space before the separator
                cp_layout.addWidget(create_separator())
                cp_layout.addWidget(QtWidgets.QLabel("Rotate X:"))
                cp_layout.addWidget(self.slider_roll)
                cp_layout.addWidget(self.textbox_roll)
                cp_layout.addWidget(QtWidgets.QLabel("Rotate Y:"))
                cp_layout.addWidget(self.slider_pitch)
                cp_layout.addWidget(self.textbox_pitch)
                cp_layout.addWidget(QtWidgets.QLabel("Rotate Z:"))
                cp_layout.addWidget(self.slider_yaw)
                cp_layout.addWidget(self.textbox_yaw)
            if method == 'cubic_from_poses':
                cp_layout.addSpacing(10)  # Add 10 pixels of space before the separator
                cp_layout.addWidget(create_separator())
                cp_layout.addSpacing(10)
                cp_layout.addWidget(self.swap_family_check_box)
                cp_layout.addWidget(QtWidgets.QLabel("Lambda:"))
                cp_layout.addWidget(self.slider_lambda)
                cp_layout.addWidget(self.textbox_lambda)

            cp_layout.addSpacing(20)
            cp_layout.addWidget(self.synthesize_button)

            cp_layout.addStretch(1)

            main_layout.addWidget(control_panel)
            self.setLayout(main_layout)
            self.setWindowTitle("Motion Designer")

        def _initialize_points(self, method, initial_pts):
            predefined_points = {
                'cubic_from_points': [
                    PointHomogeneous(),
                    PointHomogeneous([1, 1, 1, 0.3]),
                    PointHomogeneous([1, 3, -3, 0.5]),
                    PointHomogeneous([1, 0.5, -7, 1]),
                    PointHomogeneous([1, -3.2, -7, 4]),
                    PointHomogeneous([1, -7, -3, 2]),
                    PointHomogeneous([1, -8, 3, 0.5])
                ],
                'cubic_from_poses': [
                    DualQuaternion(),
                    DualQuaternion([0, 0, 0, 1, 1, 0, 1, 0]),
                    DualQuaternion([1, 2, 0, 0, -2, 1, 0, 0]),
                    DualQuaternion([3, 0, 1, 0, 1, 0, -3, 0])
                ],
                'quadratic_from_points': [
                    PointHomogeneous(),
                    PointHomogeneous([1, 1, 1, 2]),
                    PointHomogeneous([1, 3, -3, 1]),
                    PointHomogeneous([1, 2, -4, 1]),
                    PointHomogeneous([1, -2, -2, 2])
                ],
                'quadratic_from_poses': [
                    DualQuaternion(),
                    DualQuaternion(
                        TransfMatrix.from_vectors(
                            approach_z=[-0.0362862, 0.400074, 0.915764],
                            normal_x=[0.988751, -0.118680, 0.0910266],
                            origin=[0.33635718, 0.9436004, 0.3428654]).matrix2dq()),
                    DualQuaternion(
                        TransfMatrix.from_vectors(
                            approach_z=[-0.0463679, -0.445622, 0.894020],
                            normal_x=[0.985161, 0.127655, 0.114724],
                            origin=[-0.52857769, -0.4463076, -0.81766]).matrix2dq()),
                ]
            }

            required_points = {
                'cubic_from_points': 7,
                'cubic_from_poses': 4,
                'quadratic_from_points': 5,
                'quadratic_from_poses': 3
            }

            if method not in predefined_points:
                raise ValueError(f"Unknown method: {method}")

            if initial_pts is None:
                return predefined_points[method]

            if len(initial_pts) != required_points[method]:
                raise ValueError(
                    f"For a {method.replace('_', ' ')}, {required_points[method]} points are needed.")

            return initial_pts

        def set_sliders_for_point(self, index):
            """
            Set the slider positions to reflect the current coordinates of the
            control point with the given index.
            (Here we assume that coordinates are in the range roughly –10..10.)
            """
            index = index + 1  # skip the first point/pose
            sliders = [self.slider_x, self.slider_y, self.slider_z]
            text_boxes = [self.textbox_x, self.textbox_y, self.textbox_z]
            if self.method == 'quadratic_from_points' or self.method == 'cubic_from_points':
                pt = self.plotted_points[index]
                values = [int(pt[i] * 100) for i in range(3)]
            else:
                sliders.extend([self.slider_roll, self.slider_pitch, self.slider_yaw])
                text_boxes.extend([self.textbox_roll, self.textbox_pitch, self.textbox_yaw])
                pt = self.plotted_poses[index]
                rpy = self.previous_rpy_sliders_values[index]
                values = [
                    int(pt.tr.t[0] * 100),
                    int(pt.tr.t[1] * 100),
                    int(pt.tr.t[2] * 100),
                    int(rpy[0]),  # Roll
                    int(rpy[1]),  # Pitch
                    int(rpy[2])  # Yaw
                ]
                (self.slider_roll_prev, self.slider_pitch_prev,
                 self.slider_yaw_prev) = values[3:]
            #
            for slider, text_box, value in zip(sliders, text_boxes, values):
                slider.blockSignals(True)
                slider.setValue(value)
                text_box.setText(str(value / 100.0))
                slider.blockSignals(False)

        def on_synthesize_button_clicked(self):
            """
            Called when the "Synthesize mechanism" button is clicked. This method
            should be implemented to synthesize a mechanism based on the current
            control points.
            """
            if (self.method == 'quadratic_from_points'
                    or self.method == 'cubic_from_points'
                    or self.method == 'quadratic_from_poses'):
                c = MotionInterpolation.interpolate(self.points)
            else:
                p = MotionInterpolation.interpolate_cubic_numerically(
                    self.points,
                    lambda_val=self.lambda_val,
                    k_idx=self.motion_family_idx)
                c = RationalCurve.from_coeffs(p)
            self.mechanism_plotter.append(
                InteractivePlotterWidget(mechanism=RationalMechanism(c.factorize()),
                                         arrows_length=self.arrows_length,
                                         parent_app=self.plotter.app,
                                         white_background=self.white_background))
            self.mechanism_plotter[-1].show()


        def on_point_selection_changed(self, index):
            """
            When a different point is selected in the combo box, update the slider
            positions to match that point’s coordinates.
            """
            self.set_sliders_for_point(index)

        def on_slider_value_changed(self, value):
            """
            Called when any of the sliders change their value. Update the currently
            selected control point’s x, y, or z coordinate based on the slider values,
            update the control point markers, and then recalculate the motion curve.
            """
            index = self.point_combo.currentIndex() + 1
            # Convert slider values (integers) to floating‑point coordinates.
            new_x = self.slider_x.value() / 100.0
            new_y = self.slider_y.value() / 100.0
            new_z = self.slider_z.value() / 100.0

            self.textbox_x.setText(str(new_x))
            self.textbox_y.setText(str(new_y))
            self.textbox_z.setText(str(new_z))

            if self.method == 'quadratic_from_poses' or self.method == 'cubic_from_poses':
                if self.slider_roll.value() != self.slider_roll_prev:
                    new_roll = (self.slider_roll.value() - self.slider_roll_prev) / 100.0
                    new_mat = TransfMatrix.from_rotation('x', new_roll)
                    new_tr = self.plotted_poses[index].tr * new_mat
                    self.slider_roll_prev = self.slider_roll.value()
                    self.textbox_roll.setText(str(self.slider_roll.value() / 100.0))

                elif self.slider_pitch.value() != self.slider_pitch_prev:
                    new_pitch = (self.slider_pitch.value() - self.slider_pitch_prev) / 100.0
                    new_mat = TransfMatrix.from_rotation('y', new_pitch)
                    new_tr = self.plotted_poses[index].tr * new_mat
                    self.slider_pitch_prev = self.slider_pitch.value()
                    self.textbox_pitch.setText(str(self.slider_pitch.value() / 100.0))

                elif self.slider_yaw.value() != self.slider_yaw_prev:
                    new_yaw = (self.slider_yaw.value() - self.slider_yaw_prev) / 100.0
                    new_mat = TransfMatrix.from_rotation('z', new_yaw)
                    new_tr = self.plotted_poses[index].tr * new_mat
                    self.slider_yaw_prev = self.slider_yaw.value()
                    self.textbox_yaw.setText(str(self.slider_yaw.value() / 100.0))
                else:
                    new_tr = TransfMatrix.from_rpy_xyz(self.plotted_poses[index].tr.rpy(),
                                                       [new_x, new_y, new_z])

                self.previous_rpy_sliders_values[index][0] = self.slider_roll.value()
                self.previous_rpy_sliders_values[index][1] = self.slider_pitch.value()
                self.previous_rpy_sliders_values[index][2] = self.slider_yaw.value()

                new_dq = DualQuaternion(new_tr.matrix2dq())
                self.points[index] = new_dq
                self.plotted_poses[index].setData(new_tr)

            else:
                # update the selected control point
                self.points[index] = PointHomogeneous.from_3d_point([new_x, new_y, new_z])
                self.plotted_points[index] = np.array([new_x, new_y, new_z])
                # update the visual markers
                self.markers.setData(pos=self.plotted_points)

            # Recalculate and update the motion curve.
            self.update_curve_vis()

        def on_lambda_slider_value_changed(self, value):
            """
            Called when the lambda slider changes its value. Update the lambda value
            of the cubic curve, update the control point markers, and then recalculate
            the motion curve.
            """
            self.lambda_val = self.slider_lambda.value() / 100.0
            self.textbox_lambda.setText(str(self.lambda_val))
            self.update_curve_vis()

        def on_swap_family_check_box_changed(self, state):
            """
            Called when the swap family checkbox changes its state. Update the
            motion curve to reflect the new motion family.
            """
            if state == 2:
                self.motion_family_idx = 1
            else:
                self.motion_family_idx = 0

            self.update_curve_vis()

        def on_lambda_textbox_changed(self, text, slider):
            """
            Update the given slider with the value from the corresponding textbox.

            :param str text: The text input from the textbox. Should be a number.
            :param slider: The slider to update with the new value.
            """
            if text is not None:
                try:
                    value = float(text)
                    slider.blockSignals(True)
                    slider.setValue(int(value * 100))
                    slider.blockSignals(False)

                    if abs(value - 1.0) < 1e-10:
                        value = 1.00000001  # avoid numerical issues with 1.0
                        print("Warning: lambda value set to 1.0, using 1.00000001 instead.")
                    self.lambda_val = value
                    self.update_curve_vis()
                except ValueError:
                    raise ValueError(f"Invalid input for slider: {text}")

        def on_textbox_changed(self, text, slider):
            """
            Update the given slider with the value from the corresponding textbox.
            """
            if text is not None:
                try:
                    value = float(text)
                    slider.blockSignals(True)
                    slider.setValue(int(value * 100))
                    slider.blockSignals(False)

                    self.on_slider_value_changed(value)
                except ValueError:
                    raise ValueError(f"Invalid input for slider: {text}")

        def update_curve_vis(self):
            """
            Recalculate the motion curve using the current control points. The
            interpolation is performed by MotionInterpolation. Then update the curve
            line in the GLViewWidget.
            """

            # get the numeric coefficients from interpolation
            if self.method == 'cubic_from_points':
                coeffs = self.mi.interpolate_points_cubic(self.points,
                                                          return_numeric=True)
            elif self.method == 'quadratic_from_points':
                coeffs = self.mi.interpolate_points_quadratic(self.points,
                                                              return_numeric=True)
            elif self.method == 'quadratic_from_poses':
                coeffs = self.mi.interpolate_quadratic_numerically(self.points)
            elif self.method == 'cubic_from_poses':
                coeffs = self.mi.interpolate_cubic_numerically(self.points,
                                                               lambda_val=self.lambda_val,
                                                               k_idx=self.motion_family_idx)

            # create numpy polynomial objects
            curve = [np.polynomial.Polynomial(c[::-1]) for c in coeffs]

            # parameter values using a tangent substitution
            t_space = np.tan(np.linspace(-np.pi / 2, np.pi / 2, self.plotter.steps + 1))
            curve_points = []
            for t in t_space:
                dq = DualQuaternion([poly(t) for poly in curve])  # evaluate fot each t
                pt = dq.dq2point_via_matrix()
                curve_points.append(pt)
            curve_points = np.array(curve_points)

            t_space_frames = np.tan(np.linspace(-np.pi / 2, np.pi / 2, 51))
            curve_frames = []
            for t in t_space_frames:
                dq = DualQuaternion([poly(t) for poly in curve])
                curve_frames.append(TransfMatrix(dq.dq2matrix()))

            # if the curve line has not yet been created
            if self.curve_path_vis is None:
                self.curve_path_vis = gl.GLLinePlotItem(pos=curve_points,
                                                        color=(0.5, 0.5, 0.5, 1),
                                                        glOptions=self.render_mode,
                                                        width=2,
                                                        antialias=True)
                self.plotter.widget.addItem(self.curve_path_vis)

                self.curve_frames_vis = [FramePlotHelper(transform=tr,
                                                         length=self.plotter.arrows_length)
                                         for tr in curve_frames]
                for frame in self.curve_frames_vis:
                    frame.addToView(self.plotter.widget)
            else:  # update the existing curve visuals
                self.curve_path_vis.setData(pos=curve_points)
                for i, frame in enumerate(self.curve_frames_vis):
                    frame.setData(curve_frames[i])

            if self.preview_mechanism:
                self._preview_mechanism(coeffs)

        def _preview_mechanism(self, coefficients):
            """
            Compute and display the mechanism preview.

            :param np.ndarray coefficients: The coefficients of the rational curve,
                used to compute the mechanism configuration at t → ∞.
            """
            cr = RationalCurve.from_coeffs(coefficients)  # TODO avoid sympy
            me = RationalMechanism(cr.factorize())  # TODO avoid sympy
            me.smallest_polyline(update_design=True)

            t_val = 1 / np.finfo(np.float64).eps   # infinite t
            links = (me.factorizations[0].direct_kinematics(t_val) +
                     me.factorizations[1].direct_kinematics(t_val)[::-1])
            links.insert(0, links[-1])

            if self.lines is None:
                self.lines = []

                # base link
                line_item = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                              color=(1, 0.5, 0, 0.5),
                                              glOptions=self.render_mode,
                                              width=5,
                                              antialias=True)
                self.lines.append(line_item)
                self.plotter.widget.addItem(line_item)

                # other links
                for i in range(1, self.num_lines):
                    line_item = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                                  color=(1, 1, 0, 0.5),
                                                  glOptions=self.render_mode,
                                                  width=5,
                                                  antialias=True)
                    self.lines.append(line_item)
                    self.plotter.widget.addItem(line_item)
            else:
                for i, line in enumerate(self.lines):
                    pt1 = links[i]
                    pt2 = links[i + 1]
                    pts = np.array([pt1, pt2])
                    line.setData(pos=pts)

        def closeEvent(self, event):
            """
            Called when the window is closed. Ensure that the Qt application exits.
            """
            print("Closing the window... generated points for interpolation:")
            for pt in self.points:
                print(pt)
            if self.slider_lambda:
                print(f"Lambda: {self.slider_lambda.value() / 100.0}")
            if self.swap_family_check_box:
                print(f"Motion family index: {self.motion_family_idx}")
            self.clear_meshes()
            self.plotter.app.quit()

        def add_mesh(self,
                     vertices: np.ndarray,
                     faces: np.ndarray,
                     color: tuple = (0.7, 0.7, 0.7, 0.3),
                     name: str | None = None,
                     smooth: bool = False) -> object:
            """
            Add a CAD mesh to the 3D view. `vertices` should be (N, 3), `faces` (M, 3).
            No transform is applied — the mesh is assumed already positioned in the base frame.
            Returns the created GL item (can be stored or manipulated by caller).
            """
            if gl is None:
                raise RuntimeError("OpenGL / pyqtgraph not available")

            # lazy init container for CAD items
            if not hasattr(self, "cad_items"):
                self.cad_items = []

            # create meshdata and mesh item
            meshdata = gl.MeshData(vertexes=vertices, faces=faces)
            mesh_item = gl.GLMeshItem(meshdata=meshdata,
                                      smooth=bool(smooth),
                                      color=color,
                                      drawEdges=False,
                                      glOptions=self.render_mode)
            # store reference (name optional) and add to view
            self.cad_items.append((name, mesh_item))
            self.plotter.widget.addItem(mesh_item)
            return mesh_item

        def remove_mesh(self, name: str) -> bool:
            """
            Remove the first mesh with the given name. Returns True if removed.
            """
            if not hasattr(self, "cad_items"):
                return False
            for i, (n, item) in enumerate(self.cad_items):
                if n == name:
                    try:
                        self.plotter.widget.removeItem(item)
                    except Exception:
                        pass
                    del self.cad_items[i]
                    return True
            return False

        def clear_meshes(self) -> None:
            """
            Remove all added CAD meshes from the view.
            """
            if not hasattr(self, "cad_items"):
                return
            for _, item in self.cad_items:
                try:
                    self.plotter.widget.removeItem(item)
                except Exception:
                    pass
            self.cad_items = []

else:
    MotionDesignerWidget = None

