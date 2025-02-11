from functools import wraps
import numpy as np

# PyQt and Pyqtgraph imports

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui
import pyqtgraph.opengl as gl

# Import your custom classes (adjust the import paths as needed)
from .DualQuaternion import DualQuaternion
from .MotionFactorization import MotionFactorization
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous, PointOrbit
from .RationalBezier import RationalBezier
from .RationalCurve import RationalCurve
from .RationalMechanism import RationalMechanism
from .TransfMatrix import TransfMatrix
from .MiniBall import MiniBall
from .Linkage import LineSegment


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
        self.widget = CustomGLViewWidget()
        self.widget.setWindowTitle('3D Plot')
        self.widget.opts['distance'] = 10
        self.widget.setCameraPosition(distance=10, azimuth=30, elevation=30)
        self.widget.show()

        # Add a grid.
        grid = gl.GLGridItem()
        grid.setSize(20, 20)
        grid.setSpacing(1, 1)
        self.widget.addItem(grid)

        # Add coordinate axes.
        self.plot(TransfMatrix())

        # Store parameters.
        self.t_space = np.linspace(interval[0], interval[1], discrete_step_space)
        self.steps = discrete_step_space

        self.labels = []
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
            self.add_label(kwargs['label'], mid, color, self.label_font_size)

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
            self.add_label(kwargs['label'], point, (1, 1, 1, 1), self.label_font_size)

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
            self.add_label(kwargs['label'], mid, color, self.label_font_size)

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
            self.widget.add_label(scatter, kwargs['label'])

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
            self.widget.add_label(origin, kwargs['label'])

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
        color = self._get_color(kwargs.get('color', 'blue'), (1, 1, 0, 1))
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

    def paintEvent(self, event):
        # Draw the usual 3D scene first.
        super(PlotterPyqtgraph, self).paintEvent(event)

        # Set up a QPainter to draw overlay text.
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.white)
        painter.setFont(QtGui.QFont("Arial", 12))

        # Get the Model-View-Projection (MVP) matrix.
        projection_matrix = self.projectionMatrix()
        view_matrix = self.viewMatrix()
        mvp = projection_matrix * view_matrix

        # Iterate over each label and project its 3D point to 2D screen coordinates.
        for entry in self.labels:
            point = entry['point']
            text = entry['text']
            # Convert the 3D point to homogeneous coordinates.
            point_vec = QtGui.QVector4D(point.pos[0][0], point.pos[0][1], point.pos[0][2], 1.0)
            projected = mvp.map(point_vec)
            # Perform perspective division
            if projected.w() != 0:
                ndc_x = projected.x() / projected.w()
                ndc_y = projected.y() / projected.w()
            else:
                ndc_x, ndc_y = 0, 0
            # Convert normalized device coordinates to screen coordinates.
            x = int((ndc_x * 0.5 + 0.5) * self.width())
            y = int((1 - (ndc_y * 0.5 + 0.5)) * self.height())
            painter.drawText(x, y, text)

        painter.end()

    def show(self):
        """Start the Qt event loop."""
        QApplication.exec_()

    def closeEvent(self, event):
        """
        Called when the window is closed. Ensure that the Qt application exits.
        """
        self.app.quit()
        event.accept()


class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super(CustomGLViewWidget, self).__init__(*args, **kwargs)
        # List to hold labels: each entry is a dict with keys 'point' and 'text'
        self.labels = []

    def add_label(self, point, text):
        """
        Adds a label for a 3D point.
        """
        self.labels.append({'point': point, 'text': text})

    def paintEvent(self, event):
        # Draw the usual 3D scene first.
        super(CustomGLViewWidget, self).paintEvent(event)

        # Set up a QPainter to draw overlay text.
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.white)
        painter.setFont(QtGui.QFont("Arial", 12))

        # Get the Model-View-Projection (MVP) matrix.
        projection_matrix = self.projectionMatrix()
        view_matrix = self.viewMatrix()
        mvp = projection_matrix * view_matrix

        # Iterate over each label and project its 3D point to 2D screen coordinates.
        for entry in self.labels:
            point = entry['point']
            text = entry['text']

            projected = mvp.map(self._obtain_label_vec(point))
            # Perform perspective division
            if projected.w() != 0:
                ndc_x = projected.x() / projected.w()
                ndc_y = projected.y() / projected.w()
            else:
                ndc_x, ndc_y = 0, 0
            # Convert normalized device coordinates to screen coordinates.
            x = int((ndc_x * 0.5 + 0.5) * self.width())
            y = int((1 - (ndc_y * 0.5 + 0.5)) * self.height())
            painter.drawText(x, y, text)

        painter.end()

    @staticmethod
    def _obtain_label_vec(pt):
        """
        Obtain the label vector.
        """
        # Convert the 3D point to homogeneous coordinates.

        if isinstance(pt, np.ndarray):
            point_vec = pt

        elif isinstance(pt, PointHomogeneous):
            point_vec = [pt.coordinates_normalized[1],
                         pt.coordinates_normalized[2],
                         pt.coordinates_normalized[3]]

        elif isinstance(pt, TransfMatrix):
            point_vec = [pt.t[0], pt.t[1], pt.t[2]]

        elif isinstance(pt, FramePlotHelper):
            point_vec = [pt.tr.t[0], pt.tr.t[1], pt.tr.t[2]]

        else:  # is pyqtgraph marker (scatter)
            point_vec = [pt.pos[0][0], pt.pos[0][1], pt.pos[0][2]]

        return QtGui.QVector4D(point_vec[0], point_vec[1], point_vec[2], 1.0)


class FramePlotHelper:
    def __init__(self,
                 transform: TransfMatrix = TransfMatrix(),
                 width: float = 2.,
                 length: float = 1.,
                 antialias: bool = True):
        """
        Create a coordinate frame using three GLLinePlotItems.

        :param TransfMatrix transform: The initial transformation matrix.
        :param float width: The width of the lines.
        :param float length: The length of the axes.
        :param bool antialias: Whether to use antialiasing
        """
        # Create GLLinePlotItems for the three axes.
        # The initial positions are placeholders; they will be set properly in setData().
        self.x_axis = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                        color=(1, 0, 0, 0.5),
                                        width=width,
                                        antialias=antialias)
        self.y_axis = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                        color=(0, 1, 0, 0.5),
                                        width=width,
                                        antialias=antialias)
        self.z_axis = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                        color=(0, 0, 1, 0.5),
                                        width=width,
                                        antialias=antialias)

        # Set the initial transformation
        self.tr = transform
        self.length = length
        self.setData(transform)

    def setData(self, transform: TransfMatrix):
        """
        Update the coordinate frame using a new 4x4 transformation matrix.

        :param TransfMatrix transform: The new transformation matrix.
        """
        self.tr = transform

        # Update the positions for each axis.
        self.x_axis.setData(pos=np.array([transform.t, transform.t + self.length * transform.n]))
        self.y_axis.setData(pos=np.array([transform.t, transform.t + self.length * transform.o]))
        self.z_axis.setData(pos=np.array([transform.t, transform.t + self.length * transform.a]))

    def addToView(self, view: gl.GLViewWidget):
        """
        Add all three axes to a GLViewWidget.

        :param gl.GLViewWidget view: The view to add the axes to.
        """
        view.addItem(self.x_axis)
        view.addItem(self.y_axis)
        view.addItem(self.z_axis)