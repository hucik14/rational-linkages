from functools import wraps
import numpy as np
import sys

# PyQt and Pyqtgraph imports
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel
import pyqtgraph.opengl as gl

# Import your custom classes (adjust the import paths as needed)
from .DualQuaternion import DualQuaternion
from .MotionFactorization import MotionFactorization
from .MotionInterpolation import MotionInterpolation
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous, PointOrbit
from .RationalBezier import RationalBezier
from .RationalCurve import RationalCurve
from .RationalMechanism import RationalMechanism
from .TransfMatrix import TransfMatrix
from .MiniBall import MiniBall
from .Linkage import LineSegment
from .Quaternion import Quaternion


class PlotterPyqtgraph:
    def __init__(self,
                 discrete_step_space: int = 1000,
                 interval: tuple = (0, 1),
                 font_size_of_labels: int = 12):
        """
        Initialize the Pyqtgraph plotter. This version creates a GLViewWidget,
        sets a turntable‐like camera, adds a grid and coordinate axes.
        """
        # Create a Qt application if one is not already running.
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])

        # Create the GLViewWidget.
        self.widget = gl.GLViewWidget()
        self.widget.setWindowTitle('3D Plot')
        self.widget.opts['distance'] = 10
        self.widget.setCameraPosition(distance=10, azimuth=30, elevation=30)
        self.widget.show()

        # Add a grid.
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        self.widget.addItem(grid)

        # Add coordinate axes.
        self.plot(TransfMatrix())

        # Store parameters.
        self.t_space = np.linspace(interval[0], interval[1], discrete_step_space)
        self.steps = discrete_step_space
        self.label_font_size = font_size_of_labels

        # (The "interactive" flag is present for compatibility with your Vispy version.)
        self.interactive = False

    def _get_color(self, color, default):
        """
        Convert common color names to RGBA tuples.
        If color is already a tuple (or list) it is returned unchanged.
        """
        if isinstance(color, str):
            color_map = {
                'white': (1, 1, 1, 1),
                'red': (1, 0, 0, 1),
                'green': (0, 1, 0, 1),
                'blue': (0, 0, 1, 1),
                'yellow': (1, 1, 0, 1),
                'magenta': (1, 0, 1, 1),
                'cyan': (0, 1, 1, 1),
                'orange': (1, 0.5, 0, 1),
                'lime': (0, 1, 0, 1)
            }
            return color_map.get(color.lower(), default)
        return color

    def plot(self, objects_to_plot, **kwargs):
        """
        Plot one or several objects. If a list is provided, then (optionally)
        a list of labels may be provided.
        """
        if isinstance(objects_to_plot, list):
            label_list = kwargs.pop('label', None)
            for i, obj in enumerate(objects_to_plot):
                if label_list is not None:
                    kwargs['label'] = label_list[i]
                self._plot(obj, **kwargs)
        else:
            self._plot(objects_to_plot, **kwargs)
        self.widget.update()

    def _plot(self, object_to_plot, **kwargs):
        """
        Dispatch to the proper plotting method based on the object type.
        """
        type_to_plot = self.analyze_object(object_to_plot)
        if type_to_plot == "is_line":
            self._plot_line(object_to_plot, **kwargs)
        elif type_to_plot == "is_point":
            self._plot_point(object_to_plot, **kwargs)
        elif type_to_plot == "is_motion_factorization":
            self._plot_motion_factorization(object_to_plot, **kwargs)
        elif type_to_plot == "is_dq":
            self._plot_dual_quaternion(object_to_plot, **kwargs)
        elif type_to_plot == "is_transf_matrix":
            self._plot_transf_matrix(object_to_plot, **kwargs)
        elif type_to_plot == "is_rational_curve":
            self._plot_rational_curve(object_to_plot, **kwargs)
        elif type_to_plot == "is_rational_bezier":
            self._plot_rational_bezier(object_to_plot, **kwargs)
        elif type_to_plot == "is_rational_mechanism":
            self._plot_rational_mechanism(object_to_plot, **kwargs)
        elif type_to_plot == "is_interactive":
            self._plot_interactive(object_to_plot, **kwargs)
        elif type_to_plot == "is_miniball":
            self._plot_miniball(object_to_plot, **kwargs)
        elif type_to_plot == "is_line_segment":
            self._plot_line_segment(object_to_plot, **kwargs)
        elif type_to_plot == "is_point_orbit":
            self._plot_point_orbit(object_to_plot, **kwargs)
        else:
            raise TypeError("Unsupported type for plotting.")

    def analyze_object(self, object_to_plot):
        """
        Analyze the object type so that the proper plotting method is called.
        """
        if isinstance(object_to_plot, RationalMechanism):
            return "is_rational_mechanism"
        elif isinstance(object_to_plot, MotionFactorization) and not self.interactive:
            return "is_motion_factorization"
        elif isinstance(object_to_plot, NormalizedLine):
            return "is_line"
        elif isinstance(object_to_plot, PointHomogeneous):
            return "is_point"
        elif isinstance(object_to_plot, RationalBezier):
            return "is_rational_bezier"
        elif isinstance(object_to_plot, RationalCurve):
            return "is_rational_curve"
        elif isinstance(object_to_plot, DualQuaternion):
            return "is_dq"
        elif isinstance(object_to_plot, TransfMatrix):
            return "is_transf_matrix"
        elif isinstance(object_to_plot, MiniBall):
            return "is_miniball"
        elif isinstance(object_to_plot, LineSegment):
            return "is_line_segment"
        elif isinstance(object_to_plot, PointOrbit):
            return "is_point_orbit"
        else:
            raise TypeError("Unsupported type for plotting.")

    @staticmethod
    def _plotting_decorator(func):
        """
        A decorator to allow common post‑plotting updates.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result

        return wrapper

    @_plotting_decorator
    def plot_axis_between_two_points(self,
                                     p0: PointHomogeneous,
                                     p1: PointHomogeneous,
                                     **kwargs):
        """
        Plot an arrow (here as a simple line) from p0 to p1.
        """
        pos0 = np.array(p0.normalized_in_3d())
        pos1 = np.array(p1.normalized_in_3d())
        pts = np.array([pos0, pos1])
        color = self._get_color(kwargs.get('color', 'white'), (1, 1, 1, 1))
        line = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
        self.widget.addItem(line)
        scatter = gl.GLScatterPlotItem(pos=np.array([pos1]), color=color, size=5)
        self.widget.addItem(scatter)
        if 'label' in kwargs:
            mid = (pos0 + pos1) / 2
            self._add_text(kwargs['label'], mid, color, self.label_font_size)

    @_plotting_decorator
    def plot_line_segments_between_points(self,
                                          points: list,
                                          **kwargs):
        """
        Plot a connected line (polyline) through a list of points.
        """
        pts = np.array([p.normalized_in_3d() for p in points])
        color = self._get_color(kwargs.get('color', 'white'), (1, 1, 1, 1))
        line = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
        self.widget.addItem(line)

    @_plotting_decorator
    def plot_plane(self,
                   normal: np.ndarray,
                   point: np.ndarray,
                   xlim: tuple = (-1, 1),
                   ylim: tuple = (-1, 1),
                   **kwargs):
        """
        Plot a plane defined by a normal vector and a point on it.
        (Here we create a grid and then build a mesh.)
        """
        normal = np.asarray(normal)
        point = np.asarray(point)
        a, b, c = normal
        d = np.dot(normal, point)
        x = np.linspace(xlim[0], xlim[1], 20)
        y = np.linspace(ylim[0], ylim[1], 20)
        X, Y = np.meshgrid(x, y)
        Z = (d - a * X - b * Y) / c

        vertices, faces = self._create_mesh_from_grid(X, Y, Z)
        surface = gl.GLMeshItem(vertexes=vertices, faces=faces,
                                color=self._get_color(
                                    kwargs.get('color', (0.8, 0.2, 0.2, 0.4)),
                                    (0.8, 0.2, 0.2, 0.4)),
                                smooth=False, drawEdges=True,
                                edgeColor=(0.5, 0.5, 0.5, 1))
        self.widget.addItem(surface)

        if 'label' in kwargs:
            self._add_text(kwargs['label'], point, (1, 1, 1, 1), self.label_font_size)

    def _create_mesh_from_grid(self, X, Y, Z):
        """
        Create vertices and faces for a mesh given grid data.
        """
        m, n = X.shape
        vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        faces = []
        for i in range(m - 1):
            for j in range(n - 1):
                idx = i * n + j
                faces.append([idx, idx + 1, idx + n])
                faces.append([idx + 1, idx + n + 1, idx + n])
        faces = np.array(faces)
        return vertices, faces

    @_plotting_decorator
    def _plot_line(self, line: NormalizedLine, **kwargs):
        """
        Plot a line as an arrow (here a simple line). The method
        get_plot_data() is assumed to return a 6‑element array
        [x0, y0, z0, dx, dy, dz].
        """
        interval = kwargs.pop('interval', (-1, 1))
        data = line.get_plot_data(interval)

        start_pt = np.array(data[:3])
        direction = np.array(data[3:])
        end_pt = start_pt + direction
        pts = np.array([start_pt, end_pt])

        color = self._get_color(kwargs.get('color', 'white'), (1, 1, 1, 1))

        line_item = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
        self.widget.addItem(line_item)

        tip_point = gl.GLScatterPlotItem(pos=np.array([end_pt]), color=color, size=5)
        self.widget.addItem(tip_point)

        if 'label' in kwargs:
            mid = start_pt + direction / 2
            self._add_text(kwargs['label'], mid, color, self.label_font_size)

    @_plotting_decorator
    def _plot_point(self, point: PointHomogeneous, **kwargs):
        """
        Plot a point as a marker.
        """
        pos = np.array(point.get_plot_data())
        color = self._get_color(kwargs.get('color', 'red'), (1, 0, 0, 1))
        scatter = gl.GLScatterPlotItem(pos=np.array([pos]), color=color, size=10)
        self.widget.addItem(scatter)
        if 'label' in kwargs:
            self._add_text(kwargs['label'], pos, color, self.label_font_size)

    @_plotting_decorator
    def _plot_dual_quaternion(self, dq: DualQuaternion, **kwargs):
        """
        Plot a dual quaternion by converting it to a transformation matrix.
        """
        matrix = TransfMatrix(dq.dq2matrix())
        self._plot_transf_matrix(matrix, **kwargs)

    @_plotting_decorator
    def _plot_transf_matrix(self, matrix: TransfMatrix, **kwargs):
        """
        Plot a transformation matrix as three arrows (x, y, and z axes).
        """
        origin = np.array(matrix.t)
        x_axis = np.array([origin, origin + np.array(matrix.n)])
        y_axis = np.array([origin, origin + np.array(matrix.o)])
        z_axis = np.array([origin, origin + np.array(matrix.a)])

        x_line = gl.GLLinePlotItem(pos=x_axis, color=(1, 0, 0, 1), width=2,
                                   antialias=True)
        y_line = gl.GLLinePlotItem(pos=y_axis, color=(0, 1, 0, 1), width=2,
                                   antialias=True)
        z_line = gl.GLLinePlotItem(pos=z_axis, color=(0, 0, 1, 1), width=2,
                                   antialias=True)

        self.widget.addItem(x_line)
        self.widget.addItem(y_line)
        self.widget.addItem(z_line)

        if 'label' in kwargs:
            self._add_text(kwargs['label'], origin, (1, 1, 1, 1), self.label_font_size)

    @_plotting_decorator
    def _plot_rational_curve(self, curve: RationalCurve, **kwargs):
        """
        Plot a rational curve as a line. Optionally, plot poses along the curve.
        """
        interval = kwargs.pop('interval', (0, 1))
        if kwargs.pop('with_poses', False):
            if interval == 'closed':
                t_space = np.tan(np.linspace(-np.pi / 2, np.pi / 2, 51))
            else:
                t_space = np.linspace(interval[0], interval[1], 50)
            for t in t_space:
                pose_dq = DualQuaternion(curve.evaluate(t))
                self._plot_dual_quaternion(pose_dq)
        x, y, z = curve.get_plot_data(interval, self.steps)
        pts = np.column_stack((x, y, z))
        color = self._get_color(kwargs.get('color', 'yellow'), (1, 1, 0, 1))
        line_item = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
        self.widget.addItem(line_item)

    @_plotting_decorator
    def _plot_rational_bezier(self,
                              bezier: RationalBezier,
                              plot_control_points: bool = True,
                              **kwargs):
        """
        Plot a rational Bézier curve along with its control points.
        """
        interval = kwargs.pop('interval', (0, 1))
        x, y, z, x_cp, y_cp, z_cp = bezier.get_plot_data(interval, self.steps)
        pts = np.column_stack((x, y, z))
        color = self._get_color(kwargs.get('color', 'magenta'), (1, 0, 1, 1))
        line_item = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
        self.widget.addItem(line_item)
        if plot_control_points:
            cp = np.column_stack((x_cp, y_cp, z_cp))
            scatter = gl.GLScatterPlotItem(pos=cp, color=(1, 0, 0, 1), size=8)
            self.widget.addItem(scatter)
            cp_line = gl.GLLinePlotItem(pos=cp, color=(1, 0, 0, 1), width=1,
                                        antialias=True)
            self.widget.addItem(cp_line)

    @_plotting_decorator
    def _plot_motion_factorization(self, factorization: MotionFactorization, **kwargs):
        """
        Plot the motion factorization as a 3D line.
        """
        t = kwargs.pop('t', 0)
        points = factorization.direct_kinematics(t)
        pts = np.array(points)
        color = self._get_color(kwargs.get('color', 'orange'), (1, 0.5, 0, 1))
        line_item = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
        self.widget.addItem(line_item)

    @_plotting_decorator
    def _plot_rational_mechanism(self, mechanism: RationalMechanism, **kwargs):
        """
        Plot a rational mechanism by plotting its factorizations and the tool path.
        """
        t = kwargs.pop('t', 0)
        for factorization in mechanism.factorizations:
            self._plot_motion_factorization(factorization, t=t, **kwargs)
        pts0 = mechanism.factorizations[0].direct_kinematics_of_tool_with_link(
            t, mechanism.tool_frame.dq2point_via_matrix())
        pts1 = mechanism.factorizations[1].direct_kinematics_of_tool_with_link(
            t, mechanism.tool_frame.dq2point_via_matrix())[::-1]
        ee_points = np.concatenate((pts0, pts1))
        color = self._get_color(kwargs.get('color', 'cyan'), (0, 1, 1, 1))
        line_item = gl.GLLinePlotItem(pos=np.array(ee_points), color=color, width=2,
                                      antialias=True)
        self.widget.addItem(line_item)
        self._plot_tool_path(mechanism, **kwargs)

    @_plotting_decorator
    def _plot_tool_path(self, mechanism: RationalMechanism, **kwargs):
        """
        Plot the path of the tool.
        """
        t_lin = np.linspace(0, 2 * np.pi, self.steps)
        t_vals = [mechanism.factorizations[0].joint_angle_to_t_param(t_lin[i])
                  for i in range(self.steps)]
        ee_points = [mechanism.factorizations[0].direct_kinematics_of_tool(
            t_vals[i], mechanism.tool_frame.dq2point_via_matrix())
            for i in range(self.steps)]
        pts = np.array(ee_points)
        color = self._get_color(kwargs.get('color', 'lime'), (0, 1, 0, 1))
        line_item = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
        self.widget.addItem(line_item)

    @_plotting_decorator
    def _plot_miniball(self, ball: MiniBall, **kwargs):
        """
        Plot a MiniBall as a semi‑transparent mesh.
        """
        x, y, z = ball.get_plot_data()
        vertices, faces = self._create_mesh_from_grid(x, y, z)
        mesh = gl.GLMeshItem(vertexes=vertices, faces=faces,
                             color=self._get_color(
                                 kwargs.get('color', (0.2, 0.8, 0.2, 0.15)),
                                 (0.2, 0.8, 0.2, 0.15)),
                             smooth=False, drawEdges=True, edgeColor=(0, 0, 0, 1))
        self.widget.addItem(mesh)

    @_plotting_decorator
    def _plot_point_orbit(self, orbit: PointOrbit, **kwargs):
        """
        Plot a point orbit as a semi‑transparent mesh.
        """
        x, y, z = orbit.get_plot_data()
        vertices, faces = self._create_mesh_from_grid(x, y, z)
        mesh = gl.GLMeshItem(vertexes=vertices, faces=faces,
                             color=self._get_color(kwargs.get('color', (1, 0, 0, 0.15)),
                                                   (1, 0, 0, 0.15)),
                             smooth=False, drawEdges=True, edgeColor=(0, 0, 0, 1))
        self.widget.addItem(mesh)

    @_plotting_decorator
    def _plot_line_segment(self, segment: LineSegment, **kwargs):
        """
        Plot a line segment as a surface mesh.
        """
        x, y, z = segment.get_plot_data()
        vertices, faces = self._create_mesh_from_grid(x, y, z)
        mesh = gl.GLMeshItem(vertexes=vertices, faces=faces,
                             color=self._get_color(kwargs.get('color', (1, 1, 0, 0.2)),
                                                   (1, 1, 0, 0.2)),
                             smooth=False, drawEdges=True, edgeColor=(0, 0, 0, 1))
        self.widget.addItem(mesh)

    def _add_text(self, label_text: str, pos, color, font_size):
        """
        Add text labels using QLabel overlay.
        """
        label = QLabel(self.widget)
        label.setText(label_text)
        label.setStyleSheet(f"color: rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]}); font-size: {font_size}px;")
        label.move(int(pos[0]), int(pos[1]))
        label.show()

    def show(self):
        """Start the Qt event loop."""
        QApplication.exec_()


class MotionDesignerApp:
    """
    Main application class for the motion designer.

    Encapsulates the QApplication and the MotionDesigner widget.
    """
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = MotionDesigner()

    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())


class MotionDesigner(QtWidgets.QWidget):
    """
    Interactive plotting widget for designing motion curves with interpolated points.

    A widget that displays a 3D view of a motion curve and control points,
    plus a side panel with controls for selecting and modifying one of the
    control points (p0 to p6). Moving the sliders adjusts the x, y, and z
    coordinates of the selected control point, which then updates the curve.
    """
    def __init__(self,
                 parent=None,
                 steps=1000,
                 interval=(0, 1),
                 font_size_of_labels=12):
        super().__init__(parent)

        # an instance of Pyqtgraph-based plotter
        self.plotter = PlotterPyqtgraph(discrete_step_space=steps,
                                        interval=interval,
                                        font_size_of_labels=font_size_of_labels)

        # default points
        self.points = [
            PointHomogeneous(),
            PointHomogeneous([1, 1, 1, 0.3]),
            PointHomogeneous([1, 3, -3, 0.5]),
            PointHomogeneous([1, 0.5, -7, 1]),
            PointHomogeneous([1, -3.2, -7, 4]),
            PointHomogeneous([1, -7, -3, 2]),
            PointHomogeneous([1, -8, 3, 0.5])
        ]

        # array of control point coordinates (in 3D)
        self.plotted_points = np.array([pt.normalized_in_3d() for pt in self.points])

        self.mi = MotionInterpolation()

        self.curve_line = None  # path of motion curve
        self.frames = None  # poses of motion curve
        self.update_curve()  # initial curve update

        # interpolated points markers
        self.markers = gl.GLScatterPlotItem(pos=self.plotted_points,
                                            color=(0, 1, 0, 1),
                                            size=10)
        self.plotter.widget.addItem(self.markers)

        # --- build the Control Panel ---
        # combo box to select one of the points
        self.point_combo = QtWidgets.QComboBox()
        for i in range(len(self.points)):
            self.point_combo.addItem(f"Point {i}")
        self.point_combo.currentIndexChanged.connect(self.on_point_selection_changed)

        # sliders for adjusting x, y, and z
        self.slider_x = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # slider range
        for slider in (self.slider_x, self.slider_y, self.slider_z):
            slider.setMinimum(-1000)
            slider.setMaximum(1000)
            slider.setSingleStep(1)
            slider.valueChanged.connect(self.on_slider_value_changed)

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
        cp_layout.addWidget(QtWidgets.QLabel("Adjust X:"))
        cp_layout.addWidget(self.slider_x)
        cp_layout.addWidget(QtWidgets.QLabel("Adjust Y:"))
        cp_layout.addWidget(self.slider_y)
        cp_layout.addWidget(QtWidgets.QLabel("Adjust Z:"))
        cp_layout.addWidget(self.slider_z)
        cp_layout.addStretch(1)

        close_button = QtWidgets.QPushButton("Close")
        cp_layout.addWidget(close_button)
        close_button.clicked.connect(self.close)

        main_layout.addWidget(control_panel)
        self.setLayout(main_layout)
        self.setWindowTitle("Motion Designer With Sliders")

    def set_sliders_for_point(self, index):
        """
        Set the slider positions to reflect the current coordinates of the
        control point with the given index.
        (Here we assume that coordinates are in the range roughly –10..10.)
        """
        pt = self.plotted_points[index]
        # Block signals to avoid triggering update events while we set the sliders.
        self.slider_x.blockSignals(True)
        self.slider_y.blockSignals(True)
        self.slider_z.blockSignals(True)
        self.slider_x.setValue(int(pt[0] * 100))
        self.slider_y.setValue(int(pt[1] * 100))
        self.slider_z.setValue(int(pt[2] * 100))
        self.slider_x.blockSignals(False)
        self.slider_y.blockSignals(False)
        self.slider_z.blockSignals(False)

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
        index = self.point_combo.currentIndex()
        # Convert slider values (integers) to floating‑point coordinates.
        new_x = self.slider_x.value() / 100.0
        new_y = self.slider_y.value() / 100.0
        new_z = self.slider_z.value() / 100.0

        # Update the selected control point.
        # (Assuming PointHomogeneous.from_3d_point returns a new instance.)
        self.points[index] = PointHomogeneous.from_3d_point([new_x, new_y, new_z])

        # Update our stored array of control point coordinates.
        self.plotted_points[index] = np.array([new_x, new_y, new_z])

        # Update the markers in the 3D view.
        self.markers.setData(pos=self.plotted_points)

        # Recalculate and update the motion curve.
        self.update_curve()

    def update_curve(self):
        """
        Recalculate the motion curve using the current control points. The
        interpolation is performed by MotionInterpolation. Then update the curve
        line in the GLViewWidget.
        """
        # Get the numeric coefficients from cubic interpolation.
        coeffs = self.mi.interpolate_points_cubic(self.points, return_numeric=True).T
        # For each coordinate (x, y, z), create a 1D polynomial.
        curve = [np.polynomial.Polynomial(c[::-1]) for c in coeffs]

        # Generate parameter values using a tangent substitution.
        t_space = np.tan(np.linspace(-np.pi / 2, np.pi / 2, self.plotter.steps + 1))
        curve_points = []
        for t in t_space:
            # Evaluate each polynomial at t.
            dq = DualQuaternion([poly(t) for poly in curve])
            # Convert the dual quaternion to a 3D point.
            pt = dq.dq2point_via_matrix()
            curve_points.append(pt)
        curve_points = np.array(curve_points)

        t_space_frames = np.tan(np.linspace(-np.pi / 2, np.pi / 2, 51))

        frames_arrays = []
        for t in t_space_frames:
            dq = DualQuaternion([poly(t) for poly in curve])
            frames_arrays.append(TransfMatrix(dq.dq2matrix()))

        # If the curve line has not yet been created, create it.
        if self.curve_line is None:
            self.curve_line = gl.GLLinePlotItem(pos=curve_points,
                                                color=(0, 0, 1, 1),
                                                width=2,
                                                antialias=True)
            self.plotter.widget.addItem(self.curve_line)

            self.frames = [FramePlotHelper(transform=tr) for tr in frames_arrays]
            for frame in self.frames:
                frame.addToView(self.plotter.widget)
        else:
            self.curve_line.setData(pos=curve_points)
            for i, frame in enumerate(self.frames):
                frame.setData(frames_arrays[i])

    def closeEvent(self, event):
        """
        Called when the window is closed. Ensure that the Qt application exits.
        """
        print("Closing the window... generated points for interpolation:")
        for pt in self.points:
            print(pt)
        self.plotter.app.quit()

    def close(self):
        """
        Close the window.
        """
        self.closeEvent(None)


class FramePlotHelper:
    def __init__(self,
                 transform: TransfMatrix = TransfMatrix(),
                 width: float = 2,
                 antialias: bool = True):
        """
        Create a coordinate frame using three GLLinePlotItems.

        :param TransfMatrix transform: The initial transformation matrix.
        :param float width: The width of the lines.
        :param bool antialias: Whether to use antialiasing
        """
        # Create GLLinePlotItems for the three axes.
        # The initial positions are placeholders; they will be set properly in setData().
        self.x_axis = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                        color=(1, 0, 0, 1),
                                        width=width,
                                        antialias=antialias)
        self.y_axis = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                        color=(0, 1, 0, 1),
                                        width=width,
                                        antialias=antialias)
        self.z_axis = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                        color=(0, 0, 1, 1),
                                        width=width,
                                        antialias=antialias)

        # Set the initial transformation
        self.setData(transform)

    def setData(self, transform: TransfMatrix):
        """
        Update the coordinate frame using a new 4x4 transformation matrix.

        :param TransfMatrix transform: The new transformation matrix.
        """
        # Update the positions for each axis.
        self.x_axis.setData(pos=np.array([transform.t, transform.t + transform.n]))
        self.y_axis.setData(pos=np.array([transform.t, transform.t + transform.o]))
        self.z_axis.setData(pos=np.array([transform.t, transform.t + transform.a]))

    def addToView(self, view: gl.GLViewWidget):
        """
        Add all three axes to a GLViewWidget.

        :param gl.GLViewWidget view: The view to add the axes to.
        """
        view.addItem(self.x_axis)
        view.addItem(self.y_axis)
        view.addItem(self.z_axis)