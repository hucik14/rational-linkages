import sys
import numpy as np

from PyQt6.QtWidgets import QApplication
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from .DualQuaternion import DualQuaternion
from .MotionFactorization import MotionFactorization
from .NormalizedLine import NormalizedLine
from .NormalizedPlane import NormalizedPlane
from .PointHomogeneous import PointHomogeneous, PointOrbit
from .RationalBezier import RationalBezier
from .RationalCurve import RationalCurve
from .RationalMechanism import RationalMechanism
from .TransfMatrix import TransfMatrix
from .MiniBall import MiniBall
from .Linkage import LineSegment


class PlotterPyqtgraph:
    """
    PyQtGraph plotter for 3D visualization of geometric objects.
    """
    def __init__(self,
                 discrete_step_space: int = 1000,
                 interval: tuple = (0, 1),
                 arrows_length: float = 1.0,
                 white_background: bool = False
                 ):
        """
        Initialize the Pyqtgraph plotter.

        This version creates a GLViewWidget, sets a turntable‐like camera,
        adds a grid and coordinate axes.
        """
        # Create a Qt application if one is not already running.
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        self.white_background = white_background

        # Create the GLViewWidget.
        self.widget = CustomGLViewWidget(white_background=self.white_background)  # gl.GLViewWidget()
        self.widget.setWindowTitle('Rational Linkages')
        self.widget.opts['distance'] = 10
        self.widget.setCameraPosition(distance=10, azimuth=30, elevation=30)

        if self.white_background:
            self.widget.setBackgroundColor(255, 255, 255, 255)
            self.render_mode = 'opaque'
        else:
            self.render_mode = 'additive'

        self.widget.show()

        # add a grid
        grid = gl.GLGridItem()
        grid.setSize(20, 20)
        grid.setSpacing(1, 1)
        if self.white_background:
            grid.setColor(QtGui.QColor(QtCore.Qt.GlobalColor.lightGray))
        self.widget.addItem(grid)

        # store parameters
        self.t_space = np.linspace(interval[0], interval[1], discrete_step_space)
        self.steps = discrete_step_space
        self.arrows_length = arrows_length

        # add origin coordinates
        self.plot(TransfMatrix())

        self.labels = []

    @staticmethod
    def _get_color(color, default):
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
        elif type_to_plot == "is_miniball":
            self._plot_miniball(object_to_plot, **kwargs)
        elif type_to_plot == "is_line_segment":
            self._plot_line_segment(object_to_plot, **kwargs)
        elif type_to_plot == "is_point_orbit":
            self._plot_point_orbit(object_to_plot, **kwargs)
        elif type_to_plot == "is_plane":
            self._plot_plane(object_to_plot, **kwargs)
        else:
            raise TypeError("Unsupported type for plotting.")

    def analyze_object(self, object_to_plot):
        """
        Analyze the object type so that the proper plotting method is called.
        """
        if isinstance(object_to_plot, RationalMechanism):
            return "is_rational_mechanism"
        elif isinstance(object_to_plot, MotionFactorization):
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
        elif isinstance(object_to_plot, NormalizedPlane):
            return "is_plane"
        else:
            raise TypeError("Unsupported type for plotting.")

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
        color = self._get_color(kwargs.get('color', 'magenta'), (1, 1, 1, 1))
        line = gl.GLLinePlotItem(pos=pts,
                                 color=color,
                                 glOptions=self.render_mode,
                                 width=2,
                                 antialias=True)
        self.widget.addItem(line)
        scatter = gl.GLScatterPlotItem(pos=np.array([pos1]),
                                       color=color,
                                       glOptions=self.render_mode,
                                       size=5)
        self.widget.addItem(scatter)
        if 'label' in kwargs:
            mid = (pos0 + pos1) / 2
            self.widget.add_label(mid, kwargs['label'])

    def plot_line_segments_between_points(self,
                                          points: list,
                                          **kwargs):
        """
        Plot a connected line (polyline) through a list of points.
        """
        pts = np.array([p.normalized_in_3d() for p in points])
        color = self._get_color(kwargs.get('color', 'green'), (1, 1, 1, 1))
        line = gl.GLLinePlotItem(pos=pts,
                                 color=color,
                                 glOptions=self.render_mode,
                                 width=2,
                                 antialias=True)
        self.widget.addItem(line)

    def _plot_plane(self,
                    plane: NormalizedPlane,
                    xlim: tuple = (-1, 1),
                    ylim: tuple = (-1, 1),
                    **kwargs):
        """
        Plot a plane as a semi‑transparent mesh.

        :param NormalizedPlane plane: The plane to plot.
        :param tuple xlim: The x‐axis limits.
        :param tuple ylim: The y‐axis limits.
        """
        grid_points = plane.data_to_plot(xlim, ylim)

        vertices, faces = self._create_mesh_from_grid(grid_points)
        surface = gl.GLMeshItem(vertexes=vertices,
                                faces=faces,
                                color=self._get_color(
                                    kwargs.get('color', (0.8, 0.2, 0.2, 0.2)),
                                    (0.8, 0.2, 0.2, 0.2)),
                                smooth=False,
                                drawEdges=True,
                                edgeColor=(0.5, 0.5, 0.5, 1))
        self.widget.addItem(surface)

    @staticmethod
    def _create_mesh_from_grid(grid_points: tuple):
        """
        Create vertices and faces for a mesh given grid data.
        """
        x, y, z = grid_points
        m, n = x.shape
        vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        faces = []
        for i in range(m - 1):
            for j in range(n - 1):
                idx = i * n + j
                faces.append([idx, idx + 1, idx + n])
                faces.append([idx + 1, idx + n + 1, idx + n])
        faces = np.array(faces)
        return vertices, faces

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

        color = self._get_color(kwargs.get('color', 'magenta'), (1, 1, 1, 1))

        line_item = gl.GLLinePlotItem(pos=pts,
                                      color=color,
                                      glOptions=self.render_mode,
                                      width=2,
                                      antialias=True)
        self.widget.addItem(line_item)

        tip_point = gl.GLScatterPlotItem(pos=np.array([end_pt]),
                                         color=color,
                                         glOptions=self.render_mode,
                                         size=5)
        self.widget.addItem(tip_point)

        if 'label' in kwargs:
            mid = start_pt + direction / 2
            self.widget.add_label(mid, kwargs['label'])

    def _plot_point(self, point: PointHomogeneous, **kwargs):
        """
        Plot a point as a marker.
        """
        pos = np.array(point.get_plot_data())
        color = self._get_color(kwargs.get('color', 'red'), (1, 0, 0, 1))
        scatter = gl.GLScatterPlotItem(pos=np.array([pos]),
                                       color=color,
                                       glOptions=self.render_mode,
                                       size=10)
        self.widget.addItem(scatter)

        if 'label' in kwargs:
            self.widget.add_label(scatter, kwargs['label'])

    def _plot_dual_quaternion(self, dq: DualQuaternion, **kwargs):
        """
        Plot a dual quaternion by converting it to a transformation matrix.
        """
        matrix = TransfMatrix(dq.dq2matrix())
        self._plot_transf_matrix(matrix, **kwargs)

    def _plot_transf_matrix(self, matrix: TransfMatrix, **kwargs):
        """
        Plot a transformation matrix as three arrows (x, y, and z axes).
        """
        origin = np.array(matrix.t)
        x_axis = np.array([origin, origin + self.arrows_length * np.array(matrix.n)])
        y_axis = np.array([origin, origin + self.arrows_length * np.array(matrix.o)])
        z_axis = np.array([origin, origin + self.arrows_length * np.array(matrix.a)])

        x_line = gl.GLLinePlotItem(pos=x_axis,
                                   color=(1, 0, 0, 1),
                                   glOptions=self.render_mode,
                                   width=2,
                                   antialias=True)
        y_line = gl.GLLinePlotItem(pos=y_axis,
                                   color=(0, 1, 0, 1),
                                   glOptions=self.render_mode,
                                   width=2,
                                   antialias=True)
        z_line = gl.GLLinePlotItem(pos=z_axis,
                                   color=(0, 0, 1, 1),
                                   glOptions=self.render_mode,
                                   width=2,
                                   antialias=True)

        self.widget.addItem(x_line)
        self.widget.addItem(y_line)
        self.widget.addItem(z_line)

        if 'label' in kwargs:
            self.widget.add_label(origin, kwargs['label'])

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
        color = self._get_color(kwargs.get('color', 'orange'), (1, 1, 0, 1))
        line_item = gl.GLLinePlotItem(pos=pts,
                                      color=color,
                                      glOptions=self.render_mode,
                                      width=2,
                                      antialias=True)
        self.widget.addItem(line_item)

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
        color = self._get_color(kwargs.get('color', 'yellow'), (1, 0, 1, 1))
        line_item = gl.GLLinePlotItem(pos=pts,
                                      color=color,
                                      glOptions=self.render_mode,
                                      width=2,
                                      antialias=True)
        self.widget.addItem(line_item)
        if plot_control_points:
            cp = np.column_stack((x_cp, y_cp, z_cp))
            scatter = gl.GLScatterPlotItem(pos=cp,
                                           color=(1, 0, 0, 1),
                                           glOptions=self.render_mode,
                                           size=8)
            self.widget.addItem(scatter)
            cp_line = gl.GLLinePlotItem(pos=cp,
                                        color=(1, 0, 0, 1),
                                        glOptions=self.render_mode,
                                        width=1,
                                        antialias=True)
            self.widget.addItem(cp_line)

    def _plot_motion_factorization(self, factorization: MotionFactorization, **kwargs):
        """
        Plot the motion factorization as a 3D line.
        """
        t = kwargs.pop('t', 0)
        points = factorization.direct_kinematics(t)
        pts = np.array(points)
        color = self._get_color(kwargs.get('color', 'orange'), (1, 0.5, 0, 1))
        line_item = gl.GLLinePlotItem(pos=pts,
                                      color=color,
                                      glOptions=self.render_mode,
                                      width=2,
                                      antialias=True)
        self.widget.addItem(line_item)

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
        line_item = gl.GLLinePlotItem(pos=np.array(ee_points),
                                      color=color,
                                      glOptions=self.render_mode,
                                      width=2,
                                      antialias=True)
        self.widget.addItem(line_item)
        self._plot_tool_path(mechanism, **kwargs)

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
        line_item = gl.GLLinePlotItem(pos=pts,
                                      color=(1, 0, 1, 1),
                                      glOptions=self.render_mode,
                                      width=2,
                                      antialias=True)
        self.widget.addItem(line_item)

    def _plot_miniball(self, ball: MiniBall, **kwargs):
        """
        Plot a MiniBall as a semi‑transparent mesh.
        """
        raise NotImplementedError("TODO, make as point orbit")

    def _plot_point_orbit(self, orbit: PointOrbit, **kwargs):
        """
        Plot a point orbit as a semi‑transparent mesh.
        """
        if 'color' in kwargs:
            coloring = kwargs.pop('color')
        else:
            coloring = (1, 0.5, 0, 0.15)

        center, radius = orbit.get_plot_data()
        mesh = gl.MeshData.sphere(rows=8, cols=8, radius=radius)
        sphere = gl.GLMeshItem(meshdata=mesh,
                               color=coloring,
                               glOptions='translucent',
                               drawFaces=True,
                               drawEdges=False)
        sphere.translate(*center)
        self.widget.addItem(sphere)

    def _plot_line_segment(self, segment: LineSegment, **kwargs):
        """
        Plot a line segment as a surface mesh.
        """
        raise NotImplementedError("TODO, see matplotlib version")

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
        self.app.exec()

    def closeEvent(self, event):
        """
        Called when the window is closed. Ensure that the Qt application exits.
        """
        self.app.quit()
        event.accept()


class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, white_background: bool = False, *args, **kwargs):
        super(CustomGLViewWidget, self).__init__(*args, **kwargs)
        # List to hold labels: each entry is a dict with keys 'point' and 'text'
        self.labels = []

        # get white_background from kwargs
        self.white_background = white_background

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

        if self.white_background:
            painter.setPen(QtGui.QColor(QtCore.Qt.GlobalColor.black))
        else:
            painter.setPen(QtGui.QColor(QtCore.Qt.GlobalColor.white))

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
                                        glOptions='translucent',
                                        width=width,
                                        antialias=antialias)
        self.y_axis = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                        color=(0, 1, 0, 0.5),
                                        glOptions='translucent',
                                        width=width,
                                        antialias=antialias)
        self.z_axis = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                        color=(0, 0, 1, 0.5),
                                        glOptions='translucent',
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


class InteractivePlotterWidget(QtWidgets.QWidget):
    """
    A QWidget that contains a PlotterPyqtgraph 3D view and interactive controls
    (sliders and text boxes) for plotting and manipulating a mechanism.
    """
    def __init__(self,
                 mechanism: RationalMechanism,
                 show_tool: bool = True,
                 steps: int = 1000,
                 joint_sliders_lim: float = 1.0,
                 arrows_length: float = 1.0,
                 white_background: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)

        self.mechanism = mechanism
        self.show_tool = show_tool
        self.steps = steps
        self.joint_sliders_lim = joint_sliders_lim
        self.arrows_length = arrows_length

        self.white_background = white_background
        if self.white_background:
            self.render_mode = 'opaque'
        else:
            self.render_mode = 'additive'

        # Create the PlotterPyqtgraph instance.
        self.plotter = PlotterPyqtgraph(discrete_step_space=self.steps,
                                        arrows_length=self.arrows_length,
                                        white_background=self.white_background)
        # Optionally adjust the camera.
        self.plotter.widget.setCameraPosition(distance=10, azimuth=30, elevation=30)

        # Main layout: split between the 3D view and a control panel.
        main_layout = QtWidgets.QHBoxLayout(self)

        # Add the 3D view (PlotterPyqtgraph’s widget) to the layout.
        main_layout.addWidget(self.plotter.widget, stretch=5)

        # Create the control panel (on the right).
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)

        # --- Driving joint angle slider ---
        control_layout.addWidget(QtWidgets.QLabel("Joint angle [rad]:"))
        self.move_slider = self.create_float_slider(0.0, 2 * np.pi, 0.0,
                                                    orientation=QtCore.Qt.Orientation.Horizontal)
        control_layout.addWidget(self.move_slider)

        # --- Text boxes ---
        self.text_box_angle = QtWidgets.QLineEdit()
        self.text_box_angle.setPlaceholderText("Set angle [rad]:")
        control_layout.addWidget(self.text_box_angle)

        self.text_box_param = QtWidgets.QLineEdit()
        self.text_box_param.setPlaceholderText("Set parameter t [-]:")
        control_layout.addWidget(self.text_box_param)

        self.save_mech_pkl = QtWidgets.QLineEdit()
        self.save_mech_pkl.setPlaceholderText("Save mechanism PKL, filename:")
        control_layout.addWidget(self.save_mech_pkl)

        self.save_figure_box = QtWidgets.QLineEdit()
        self.save_figure_box.setPlaceholderText("Save figure PNG, filename:")
        control_layout.addWidget(self.save_figure_box)

        # --- Joint connection sliders ---
        joint_sliders_layout = QtWidgets.QHBoxLayout()
        self.joint_sliders = []

        # Initialize sliders for each joint
        for i in range(self.mechanism.num_joints):
            slider0, slider1 = self._init_joint_sliders(i, self.joint_sliders_lim)
            self.joint_sliders.append(slider0)
            self.joint_sliders.append(slider1)

            # Arrange sliders vertically for each joint
            joint_layout = QtWidgets.QVBoxLayout()

            joint_layout.addWidget(QtWidgets.QLabel(f"j{i}cp0"))
            joint_layout.addWidget(slider0)
            joint_layout.addWidget(QtWidgets.QLabel(f"j{i}cp1"))
            joint_layout.addWidget(slider1)

            joint_sliders_layout.addLayout(joint_layout)

        control_layout.addLayout(joint_sliders_layout)

        # Set default values for the first factorization
        for i in range(self.mechanism.factorizations[0].number_of_factors):
            default_val0 = self.mechanism.factorizations[0].linkage[i].points_params[0]
            default_val1 = self.mechanism.factorizations[0].linkage[i].points_params[1]
            self.joint_sliders[2 * i].setValue(int(default_val0 * 100))
            self.joint_sliders[2 * i + 1].setValue(int(default_val1 * 100))

        # Set default values for the second factorization
        offset = 2 * self.mechanism.factorizations[0].number_of_factors
        for i in range(self.mechanism.factorizations[1].number_of_factors):
            default_val0 = self.mechanism.factorizations[1].linkage[i].points_params[0]
            default_val1 = self.mechanism.factorizations[1].linkage[i].points_params[1]
            self.joint_sliders[offset + 2 * i].setValue(int(default_val0 * 100))
            self.joint_sliders[offset + 2 * i + 1].setValue(int(default_val1 * 100))

        main_layout.addWidget(control_panel, stretch=1)

        # --- Initialize plot items for the mechanism links ---
        self.lines = []
        num_lines = self.mechanism.num_joints * 2
        for i in range(num_lines):
            # if i is even, make the link color green, and joints red
            if i % 2 == 0:
                line_item = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                              color=(0, 1, 0, 1),
                                              glOptions=self.render_mode,
                                              width=5,
                                              antialias=True)
            else:
                line_item = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                              color=(1, 0, 0, 1),
                                              glOptions=self.render_mode,
                                              width=5,
                                              antialias=True)
            self.lines.append(line_item)
            self.plotter.widget.addItem(line_item)

        # --- If desired, initialize tool plot and tool frame ---
        if self.show_tool:
            self.tool_link = gl.GLLinePlotItem(pos=np.zeros((3, 3)),
                                               color=(0, 1, 0, 0.5),
                                               glOptions=self.render_mode,
                                               width=5,
                                               antialias=True)
            self.plotter.widget.addItem(self.tool_link)
            self.tool_frame = FramePlotHelper(
                transform=TransfMatrix(self.mechanism.tool_frame.dq2matrix()),
                length=self.arrows_length)
            self.tool_frame.addToView(self.plotter.widget)

        # --- Plot the tool path ---
        self._plot_tool_path()

        # --- Connect signals to slots ---
        self.move_slider.valueChanged.connect(self.on_move_slider_changed)
        self.text_box_angle.returnPressed.connect(self.on_angle_text_entered)
        self.text_box_param.returnPressed.connect(self.on_param_text_entered)
        self.save_mech_pkl.returnPressed.connect(self.on_save_save_mech_pkl)
        self.save_figure_box.returnPressed.connect(self.on_save_figure_box)
        for slider in self.joint_sliders:
            slider.valueChanged.connect(self.on_joint_slider_changed)

        # Set initial configuration (home position)
        self.move_slider.setValue(self.move_slider.minimum())
        self.plot_slider_update(self.move_slider.value() / 100.0)

        self.setWindowTitle('Rational Linkages')

    # --- Helper to create a “float slider” (using integer scaling) ---
    def create_float_slider(self, min_val, max_val, init_val,
                            orientation=QtCore.Qt.Orientation.Horizontal):
        slider = QtWidgets.QSlider(orientation)
        slider.setMinimum(int(min_val * 100))
        slider.setMaximum(int(max_val * 100))
        slider.setValue(int(init_val * 100))
        slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(10)
        return slider

    def _init_joint_sliders(self, idx, slider_limit):
        """
        Create a pair of vertical sliders for joint connection parameters.
        (The slider values are scaled by 100.)
        """
        slider0 = self.create_float_slider(-slider_limit,
                                           slider_limit,
                                           0.0,
                                           orientation=QtCore.Qt.Orientation.Vertical)
        slider1 = self.create_float_slider(-slider_limit,
                                           slider_limit,
                                           0.0,
                                           orientation=QtCore.Qt.Orientation.Vertical)
        return slider0, slider1

    def _plot_tool_path(self):
        """
        Plot the tool path (as a continuous line) using a set of computed points.
        """
        t_lin = np.linspace(0, 2 * np.pi, self.steps)
        t_vals = [self.mechanism.factorizations[0].joint_angle_to_t_param(t)
                  for t in t_lin]
        ee_points = [self.mechanism.factorizations[0].direct_kinematics_of_tool(
                        t, self.mechanism.tool_frame.dq2point_via_matrix())
                     for t in t_vals]
        pts = np.array(ee_points)
        tool_path = gl.GLLinePlotItem(pos=pts,
                                      color=(0.5, 0.5, 0.5, 1),
                                      glOptions=self.render_mode,
                                      width=2,
                                      antialias=True)
        self.plotter.widget.addItem(tool_path)

    # --- Slots for interactive control events ---
    def on_move_slider_changed(self, value):
        """
        Called when the driving joint angle slider is moved.
        """
        angle = value / 100.0  # Convert back to a float value.
        self.plot_slider_update(angle)

    def on_angle_text_entered(self):
        """
        Called when the angle text box is submitted.
        """
        try:
            val = float(self.text_box_angle.text())
            # Normalize angle to [0, 2*pi]
            if val >= 0:
                val = val % (2 * np.pi)
            else:
                val = (val % (2 * np.pi)) - np.pi
            self.move_slider.setValue(int(val * 100))
        except ValueError:
            pass

    def on_param_text_entered(self):
        """
        Called when the t-parameter text box is submitted.
        """
        try:
            val = float(self.text_box_param.text())
            self.plot_slider_update(val, t_param=val)
            joint_angle = self.mechanism.factorizations[0].t_param_to_joint_angle(val)
            self.move_slider.setValue(int(joint_angle * 100))
        except ValueError:
            pass

    def on_save_save_mech_pkl(self):
        """
        Called when the save text box is submitted.
        """
        filename = self.save_mech_pkl.text()
        self.mechanism.save(filename=filename)

        QtWidgets.QMessageBox.information(self,
                                          "Success",
                                          f"Mechanism saved as {filename}.pkl")

    def on_save_figure_box(self):
        """
        Called when the filesave text box is submitted.

        Saves the current figure in the specified format.
        """
        filename = self.save_figure_box.text()

        self.plotter.widget.readQImage().save(filename + ".png")
        # self.plotter.widget.readQImage().save(filename + ".png", quality=100)
        QtWidgets.QMessageBox.information(self,
                                          "Success",
                                          f"Figure saved as {filename}.png")

    def on_joint_slider_changed(self, value):
        """
        Called when any joint slider is changed.
        Updates the joint connection parameters and refreshes the plot.
        """
        num_of_factors = self.mechanism.factorizations[0].number_of_factors
        # Update first factorization's linkage parameters.
        for i in range(num_of_factors):
            self.mechanism.factorizations[0].linkage[i].set_point_by_param(
                0, self.joint_sliders[2 * i].value() / 100.0)
            self.mechanism.factorizations[0].linkage[i].set_point_by_param(
                1, self.joint_sliders[2 * i + 1].value() / 100.0)
        # Update second factorization's linkage parameters.
        for i in range(num_of_factors):
            self.mechanism.factorizations[1].linkage[i].set_point_by_param(
                0, self.joint_sliders[2 * num_of_factors + 2 * i].value() / 100.0)
            self.mechanism.factorizations[1].linkage[i].set_point_by_param(
                1, self.joint_sliders[2 * num_of_factors + 1 + 2 * i].value() / 100.0)
        self.plot_slider_update(self.move_slider.value() / 100.0)

    def plot_slider_update(self, angle, t_param=None):
        """
        Update the mechanism plot based on the current joint angle or t parameter.
        """
        if t_param is not None:
            t = t_param
        else:
            t = self.mechanism.factorizations[0].joint_angle_to_t_param(angle)

        # Compute link positions.
        links = (self.mechanism.factorizations[0].direct_kinematics(t) +
                 self.mechanism.factorizations[1].direct_kinematics(t)[::-1])
        links.insert(0, links[-1])  # as in your original code

        # Update each line segment.
        for i, line in enumerate(self.lines):
            pt1 = links[i]
            pt2 = links[i+1]
            pts = np.array([pt1, pt2])
            line.setData(pos=pts)

        if self.show_tool:
            pts0 = self.mechanism.factorizations[0].direct_kinematics(t)[-1]
            pts1 = self.mechanism.factorizations[0].direct_kinematics_of_tool(t, self.mechanism.tool_frame.dq2point_via_matrix())
            pts2 = self.mechanism.factorizations[1].direct_kinematics(t)[-1]

            tool_triangle = [pts0, pts1, pts2]

            self.tool_link.setData(pos=np.array(tool_triangle))

            # Update tool frame (pose) arrows.
            pose_dq = DualQuaternion(self.mechanism.evaluate(t))
            # Compute the pose matrix by composing the mechanism’s pose and tool frame.
            pose_matrix = TransfMatrix(pose_dq.dq2matrix()) * TransfMatrix(self.mechanism.tool_frame.dq2matrix())
            self.tool_frame.setData(pose_matrix)

        self.plotter.widget.update()


class InteractivePlotter:
    """
    Main application class for the interactive plotting of mechanisms.

    Encapsulates the QApplication and the InteractivePlotter widget.

    :examples:

    .. testcode:: [interactiveplotter_example1]

        from rational_linkages import InteractivePlotter
        from rational_linkages.models import collisions_free_6r


        p = InteractivePlotter(mechanism=collisions_free_6r())
        p.show()

    .. testcleanup:: [motiondesigner_example1]

        del p


    """
    def __init__(self,
                 mechanism: RationalMechanism,
                 show_tool=True,
                 steps=1000,
                 joint_sliders_lim=1.0,
                 arrows_length=1.0,
                 white_background: bool = False):
        """
        Initialize the application.

        :param RationalMechanism mechanism: The mechanism to be plotted
        :param bool show_tool: whether to show the tool (end-effector) frame
        :param int steps: the number of discrete steps for the curve path
        :param float joint_sliders_lim: the limit for the joint sliders
        :param float arrows_length: the length of the arrows of plotted frames
        """
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        self.window = InteractivePlotterWidget(mechanism=mechanism,
                                               show_tool=show_tool,
                                               steps=steps,
                                               joint_sliders_lim=joint_sliders_lim,
                                               arrows_length=arrows_length,
                                               white_background=white_background)

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
        self.app.exec()

    def closeEvent(self, event):
        self.app.closeAllWindows()
        self.app.quit()
        event.accept()
