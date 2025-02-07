from functools import wraps
from itertools import cycle
import numpy as np

from PyQt5 import QtWidgets, QtCore

from vispy import scene, app
from vispy.scene import visuals

# Import your classes as before
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
from .Quaternion import Quaternion



class PlotterVispy:
    def __init__(self,
                 discrete_step_space: int = 1000,
                 interval: tuple = (0, 1),
                 font_size_of_labels: int = 100,
                 ):
        """
        Initialize the Vispy plotter.

        Note: This version uses a SceneCanvas with a TurntableCamera.
        """

        # Create the Vispy canvas and set up a 3D view.
        self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45,
                                                         azimuth=30,
                                                         elevation=30,
                                                         distance=10)



        origin = visuals.XYZAxis(parent=self.view.scene)
        self.view.add(visuals.GridLines())


        # Store parameters (some parameters from the matplotlib version are not used)
        self.t_space = np.linspace(interval[0], interval[1], discrete_step_space)
        self.steps = discrete_step_space
        self.label_font_size = font_size_of_labels

    def plot(self, objects_to_plot, **kwargs):
        """
        Plot one or several objects using Vispy.

        :param objects_to_plot: a single object or a list of objects
        :param kwargs: plotting options following matplotlib standards and syntax; optional kwargs:
            - with_poses=True: rational curve with poses
            - interval='closed': rational curve will be closed in the interval (tangent half-angle substitution)
            - show_tool=True: mechanism with tool frame
        """
        if isinstance(objects_to_plot, list):
            label_list = kwargs.pop('label', None)
            for i, obj in enumerate(objects_to_plot):
                if label_list is not None:
                    kwargs['label'] = label_list[i]
                self._plot(obj, **kwargs)
        else:
            self._plot(objects_to_plot, **kwargs)
        self.canvas.update()

    def _plot(self, object_to_plot, **kwargs):
        """
        Dispatch the plotting based on object type.
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
        Analyze the object type for dispatching the correct plotting method.
        """
        if isinstance(object_to_plot, RationalMechanism):# and not self.interactive:
            return "is_rational_mechanism"
        # elif isinstance(object_to_plot, RationalMechanism) and self.interactive:
        #     return "is_interactive"
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
        A decorator to allow common post-plotting updates.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            # self.update_camera_range()
            return result

        return wrapper

    @_plotting_decorator
    def plot_axis_between_two_points(self,
                                     p0: PointHomogeneous,
                                     p1: PointHomogeneous,
                                     **kwargs):
        """
        Plot an arrow from p0 to p1.
        """
        pos = np.vstack([p0.normalized_in_3d(), p1.normalized_in_3d()])
        arrow_vis = np.array([p0.normalized_in_3d(),
                              p1.normalized_in_3d()]).reshape((1, 6))
        arrow = visuals.Arrow(pos=pos,
                              color=kwargs.get('color', 'white'),
                              arrow_size=10,
                              width=2,
                              arrows=arrow_vis,
                              arrow_color=kwargs.get('color', 'white'))
        self.view.add(arrow)
        # Optionally add a text label at the midpoint.
        if 'label' in kwargs:
            mid = (p0.normalized_in_3d() + p1.normalized_in_3d()) / 2
            text = visuals.Text(text=kwargs['label'],
                                color='white',
                                pos=mid,
                                font_size=self.label_font_size)
            self.view.add(text)

    @_plotting_decorator
    def plot_line_segments_between_points(self,
                                          points: list[PointHomogeneous],
                                          **kwargs):
        """
        Plot a connected line (polyline) through a list of points.
        """
        pts = np.array([p.normalized_in_3d() for p in points])
        line = visuals.Line(pos=pts,
                            color=kwargs.get('color', 'white'),
                            width=2,
                            method='gl')
        self.view.add(line)

    @_plotting_decorator
    def plot_plane(self,
                   normal: np.ndarray,
                   point: np.ndarray,
                   xlim: tuple[float, float] = (-1, 1),
                   ylim: tuple[float, float] = (-1, 1),
                   **kwargs):
        """
        Plot a plane defined by a normal vector and a point on it.
        """
        normal = np.asarray(normal)
        point = np.asarray(point)
        a, b, c = normal
        d = np.dot(normal, point)
        x = np.linspace(xlim[0], xlim[1], 20)
        y = np.linspace(ylim[0], ylim[1], 20)
        X, Y = np.meshgrid(x, y)
        Z = (d - a * X - b * Y) / c

        # Create a SurfacePlot instead of a Mesh
        surface = visuals.SurfacePlot(x=X, y=Y, z=Z,
                                      color=kwargs.get('color', (0.8, 0.2, 0.2, 0.4)))
        self.view.add(surface)

        if 'label' in kwargs:
            text = visuals.Text(text=kwargs['label'], color='white', pos=point)
            self.view.add(text)

    @_plotting_decorator
    def _plot_line(self, line: NormalizedLine, **kwargs):
        """
        Plot a line as an arrow. Assumes the method get_plot_data() returns a
        6‑element array [x0, y0, z0, dx, dy, dz].
        """
        interval = kwargs.pop('interval', (-1, 1))
        data = line.get_plot_data(interval)
        start = data[:3]
        direction = data[3:]
        end = start + direction
        pos = np.vstack([start, end])
        arrows_vis = np.array([start, end]).reshape((1, 6))
        arrow = visuals.Arrow(pos=pos,
                              color=kwargs.get('color', 'white'),
                              arrow_size=10,
                              width=2,
                              arrows=arrows_vis,
                              arrow_color=kwargs.get('color', 'white'))
        self.view.add(arrow)
        if 'label' in kwargs:
            mid = start + direction / 2
            text = visuals.Text(text=kwargs['label'], color='white', pos=mid)
            self.view.add(text)

    @_plotting_decorator
    def _plot_point(self, point: PointHomogeneous, **kwargs):
        """
        Plot a point as a marker.
        """
        pos = point.get_plot_data()  # Expecting a 3D coordinate.
        marker = visuals.Markers()
        marker.set_data(np.array([pos]),
                        face_color=kwargs.get('color', 'red'),
                        size=10)
        self.view.add(marker)
        if 'label' in kwargs:
            text = visuals.Text(text=kwargs['label'],
                                color=kwargs.get('color', 'red'),
                                pos=pos,
                                font_size=self.label_font_size)
            self.view.add(text)

    @_plotting_decorator
    def _plot_dual_quaternion(self, dq: DualQuaternion, **kwargs):
        """
        Plot a dual quaternion as a transformation by converting it to a matrix.
        """
        matrix = TransfMatrix(dq.dq2matrix())
        self._plot_transf_matrix(matrix, **kwargs)

    @_plotting_decorator
    def _plot_transf_matrix(self, matrix: TransfMatrix, **kwargs):
        """
        Plot a transformation matrix as three arrows representing the x, y, and z axes.
        """
        # Expect get_plot_data() to return tuples (origin, end) for each axis.

        pos = np.array([matrix.t,
                        matrix.t + matrix.n,
                        matrix.t,
                        matrix.t + matrix.o,
                        matrix.t,
                        matrix.t + matrix.a])

        frame = visuals.XYZAxis(pos=pos)
        self.view.add(frame)
        if 'label' in kwargs:
            text = visuals.Text(text=kwargs['label'],
                                color='white',
                                font_size=self.label_font_size,
                                pos=matrix.t)
            self.view.add(text)

    @_plotting_decorator
    def _plot_rational_curve(self, curve: RationalCurve, **kwargs):
        """
        Plot a rational curve as a line. Optionally plot poses along the curve.
        """
        interval = kwargs.pop('interval', (0, 1))

        if 'with_poses' in kwargs and kwargs['with_poses'] is True:
            kwargs.pop('with_poses')

            if interval == 'closed':
                # tangent half-angle substitution for closed curves
                t_space = np.tan(np.linspace(-np.pi / 2, np.pi / 2, 51))
            else:
                t_space = np.linspace(interval[0], interval[1], 50)

            for t in t_space:
                pose_dq = DualQuaternion(curve.evaluate(t))
                self._plot_dual_quaternion(pose_dq)

        x, y, z = curve.get_plot_data(interval, self.steps)
        pos = np.column_stack((x, y, z))
        line = visuals.Line(pos=pos,
                            color=kwargs.get('color', 'yellow'),
                            width=2,
                            method='gl')
        self.view.add(line)

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
        pos = np.column_stack((x, y, z))
        line = visuals.Line(pos=pos,
                            color=kwargs.get('color', 'magenta'),
                            width=2,
                            method='gl')
        self.view.add(line)
        if plot_control_points:
            cp = np.column_stack((x_cp, y_cp, z_cp))
            markers = visuals.Markers()
            markers.set_data(cp, face_color='red', size=8)
            self.view.add(markers)
            self.view.add(visuals.Line(pos=cp,
                                       color='red',
                                       width=1,
                                       method='gl'))

    @_plotting_decorator
    def _plot_motion_factorization(self, factorization: MotionFactorization, **kwargs):
        """
        Plot the motion factorization as a 3D line.
        """
        t = kwargs.pop('t', 0)
        points = factorization.direct_kinematics(t)
        pos = np.array(points)
        line = visuals.Line(pos=pos,
                            color=kwargs.get('color', 'orange'),
                            width=2,
                            method='gl')
        self.view.add(line)

    @_plotting_decorator
    def _plot_rational_mechanism(self, mechanism: RationalMechanism, **kwargs):
        """
        Plot a rational mechanism by plotting its factorizations and the tool path.
        """
        # self.plotted['mechanism'] = mechanism
        t = kwargs.pop('t', 0)
        for factorization in mechanism.factorizations:
            self._plot_motion_factorization(factorization, t=t, **kwargs)
        pts0 = mechanism.factorizations[0].direct_kinematics_of_tool_with_link(
            t, mechanism.tool_frame.dq2point_via_matrix())
        pts1 = mechanism.factorizations[1].direct_kinematics_of_tool_with_link(
            t, mechanism.tool_frame.dq2point_via_matrix())[::-1]
        ee_points = np.concatenate((pts0, pts1))
        pos = np.array(ee_points)
        line = visuals.Line(pos=pos,
                            color=kwargs.get('color', 'cyan'),
                            width=2,
                            method='gl')
        self.view.add(line)
        self._plot_tool_path(mechanism, **kwargs)

    @_plotting_decorator
    def _plot_tool_path(self, mechanism: RationalMechanism, **kwargs):
        """
        Plot the path of the tool.
        """
        t_lin = np.linspace(0, 2 * np.pi, self.steps)
        t_vals = [mechanism.factorizations[0].joint_angle_to_t_param(t_lin[i])
                  for i in range(self.steps)]
        ee_points = [
            mechanism.factorizations[0].direct_kinematics_of_tool(
                t_vals[i], mechanism.tool_frame.dq2point_via_matrix())
            for i in range(self.steps)
        ]
        pos = np.array(ee_points)
        line = visuals.Line(pos=pos,
                            color=kwargs.get('color', 'lime'),
                            width=2,
                            method='gl')
        self.view.add(line)

    @_plotting_decorator
    def _plot_miniball(self, ball: MiniBall, **kwargs):
        """
        Plot a MiniBall as a semi‑transparent mesh.
        """
        x, y, z = ball.get_plot_data()
        # Here we assume that x, y, z are 2D arrays from a grid.
        vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        # In a structured grid one would normally generate faces; here a simple example:
        mesh = visuals.Mesh(vertices=vertices,
                            color=kwargs.get('color', (0.2, 0.8, 0.2, 0.15)),
                            shading='flat',
                            mode='triangle')
        self.view.add(mesh)

    @_plotting_decorator
    def _plot_point_orbit(self, orbit: PointOrbit, **kwargs):
        """
        Plot a point orbit as a semi‑transparent mesh.
        """
        x, y, z = orbit.get_plot_data()
        vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        mesh = visuals.Mesh(vertices=vertices,
                            color=kwargs.get('color', (1, 0, 0, 0.15)),
                            shading='flat',
                            mode='triangle')
        self.view.add(mesh)

    @_plotting_decorator
    def _plot_line_segment(self, segment: LineSegment, **kwargs):
        """
        Plot a line segment as a surface mesh.
        """
        x, y, z = segment.get_plot_data()
        vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        mesh = visuals.Mesh(vertices=vertices,
                            color=kwargs.get('color', (1, 1, 0, 0.2)),
                            shading='flat',
                            mode='triangle')
        self.view.add(mesh)

    def show(self):
        app.run()


class PlotterVispyInteractive(PlotterVispy):
    def __init__(self,
                 interactive: bool = True,
                 j_sliders_limit: float = 1.0,
                 **kwargs):
        """
        Initialize the Vispy plotter for interactive plotting.
        """
        super().__init__(**kwargs)
        self.interactive = interactive
        self.j_sliders_limit = j_sliders_limit

    def _init_control_panel(self):
        """
        Create a control panel with sliders and text boxes.
        """
        self.control_panel = QtWidgets.QWidget()
        cp_layout = QtWidgets.QGridLayout(self.control_panel)
        self.control_panel.setLayout(cp_layout)

        # Driving joint angle slider (horizontal)
        self.move_slider = self._init_slider()  # No index means driving joint slider
        cp_layout.addWidget(QtWidgets.QLabel("Joint angle [rad]:"), 0, 0)
        cp_layout.addWidget(self.move_slider, 0, 1)

        # Text boxes for manual input
        self.text_box_angle = QtWidgets.QLineEdit()
        self.text_box_angle.setPlaceholderText("Set angle [rad]")
        cp_layout.addWidget(self.text_box_angle, 1, 0)

        self.text_box_param = QtWidgets.QLineEdit()
        self.text_box_param.setPlaceholderText("Set param t")
        cp_layout.addWidget(self.text_box_param, 1, 1)

        self.text_box_save = QtWidgets.QLineEdit()
        self.text_box_save.setPlaceholderText("Save with filename")
        cp_layout.addWidget(self.text_box_save, 1, 2)

        # Container for joint sliders (vertical)
        self.joint_sliders = []
        self.joint_sliders_widget = QtWidgets.QWidget()
        js_layout = QtWidgets.QHBoxLayout(self.joint_sliders_widget)
        self.joint_sliders_widget.setLayout(js_layout)
        cp_layout.addWidget(self.joint_sliders_widget, 2, 0, 1, 3)

        # Connect signals for text boxes.
        self.text_box_angle.returnPressed.connect(self.submit_angle)
        self.text_box_param.returnPressed.connect(self.submit_parameter)
        self.text_box_save.returnPressed.connect(self.submit_save)

        # Connect the main slider
        self.move_slider.valueChanged.connect(self.plot_slider_update)

        # # Add the control panel to the main layout.
        # self.view.main_layout.addWidget(self.control_panel)

    @staticmethod
    def _init_slider(idx: int = None, j_sliders=None, slider_limit: float = 1.0):
        """
        Initialize a slider widget.

        For the driving joint angle slider, a horizontal QSlider is returned.
        For joint connection sliders, a pair of vertical QSliders is returned.
        """
        if idx is None:
            # Driving joint angle slider: use integer range scaled by 100.
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(628)  # 0 to 2*pi, scaled (i.e. 6.28*100)
            slider.setValue(0)
            slider.setSingleStep(1)
            return slider
        else:
            # Joint connection sliders.
            slider0 = QtWidgets.QSlider(QtCore.Qt.Vertical)
            slider1 = QtWidgets.QSlider(QtCore.Qt.Vertical)
            # Scale: multiply slider_limit by 100 to work in integer range.
            slider0.setMinimum(int(-slider_limit * 100))
            slider0.setMaximum(int(slider_limit * 100))
            slider0.setValue(0)
            slider1.setMinimum(int(-slider_limit * 100))
            slider1.setMaximum(int(slider_limit * 100))
            slider1.setValue(0)
            return slider0, slider1

    def _plot_interactive(self, mechanism: RationalMechanism,
                          show_tool: bool = True, **kwargs):
        """
        Plot a mechanism in interactive mode using Qt widgets.

        This method sets up the tool path, creates the interactive sliders and
        text boxes, and connects the event handlers.
        """
        self.plotted['mechanism'] = mechanism
        self.show_tool = show_tool

        # Plot the tool path (using your pre‐existing method).
        self._plot_tool_path(mechanism, **kwargs)

        # Create joint connection sliders for each joint.
        self.joint_sliders = []
        num_joints = mechanism.num_joints
        # Here we assume that each joint has two sliders.
        # Add these sliders to the joint sliders widget layout.
        js_layout = self.joint_sliders_widget.layout()
        # Clear any old sliders.
        for i in reversed(range(js_layout.count())):
            js_layout.itemAt(i).widget().deleteLater()

        for i in range(num_joints):
            slider0, slider1 = self._init_slider(idx=i,
                                                 slider_limit=self.j_sliders_limit)
            self.joint_sliders.append(slider0)
            self.joint_sliders.append(slider1)
            # Add to layout (each joint gets its two sliders side by side)
            joint_container = QtWidgets.QWidget()
            joint_layout = QtWidgets.QHBoxLayout(joint_container)
            joint_layout.addWidget(slider0)
            joint_layout.addWidget(slider1)
            js_layout.addWidget(joint_container)

            # Connect the sliders to update the connecting points.
            slider0.valueChanged.connect(self.plot_connecting_points_update)
            slider1.valueChanged.connect(self.plot_connecting_points_update)

        # Initialize the joint slider positions to the home configuration.
        # (Assuming that each factorization has a 'linkage' list with a
        #  'points_params' attribute.)
        for i in range(mechanism.factorizations[0].number_of_factors):
            self.joint_sliders[2 * i].setValue(
                int(mechanism.factorizations[0].linkage[i].points_params[0] * 100))
            self.joint_sliders[2 * i + 1].setValue(
                int(mechanism.factorizations[0].linkage[i].points_params[1] * 100))
        for i in range(mechanism.factorizations[1].number_of_factors):
            base = 2 * mechanism.factorizations[0].number_of_factors
            self.joint_sliders[base + 2 * i].setValue(
                int(mechanism.factorizations[1].linkage[i].points_params[0] * 100))
            self.joint_sliders[base + 2 * i + 1].setValue(
                int(mechanism.factorizations[1].linkage[i].points_params[1] * 100))

        # Initialize linkages plot.
        linestyles = cycle(['solid', 'dashdot'])
        self.lines = []
        # In Vispy we use Line visuals. Create one line per connecting segment.
        # (Store them so that later updates call set_data() on them.)
        num_lines = mechanism.num_joints * 2  # adjust as needed
        for i in range(num_lines):
            line = scene.visuals.Line(color='black', width=2)
            self.view.add(line)
            self.lines.append(line)

        if self.show_tool:
            # Create a visual for the tool connection (dashed purple line)
            self.tool_plot = scene.visuals.Line(color='purple',
                                                width=2,
                                                method='gl',
                                                # dash_pattern=[5, 5]
                                                )
            self.view.add(self.tool_plot)
            # For the tool frame (three arrows), store them in a list.
            self.pose_frame = []
            for color in ["red", "green", "blue"]:
                arrow = scene.visuals.Arrow(color=color, width=3, arrow_size=10)
                self.view.add(arrow)
                self.pose_frame.append(arrow)

        # (Re)connect the text box signals (already connected in _init_control_panel).
        # Finally, initialize the plot at home configuration by setting the main slider to 0.
        self.move_slider.setValue(0)

    def submit_angle(self):
        """Event handler for the angle text box."""
        try:
            val = float(self.text_box_angle.text())
        except ValueError:
            return
        # Normalize angle to [0, 2*pi]
        if val >= 0:
            val = val % (2 * np.pi)
        else:
            val = (val % (2 * np.pi)) - np.pi
        # Scale to slider integer range (0 to 628)
        self.move_slider.setValue(int(val * 100))
        self.text_box_angle.clear()

    def submit_parameter(self):
        """Event handler for the t-parameter text box."""
        try:
            val = float(self.text_box_param.text())
        except ValueError:
            return
        # Update the plot with the new t parameter.
        self.plot_slider_update(val, t_param=val)
        # Also update the driving slider accordingly.
        mech = self.plotted['mechanism']
        new_angle = mech.factorizations[0].t_param_to_joint_angle(val)
        self.move_slider.setValue(int(new_angle * 100))
        self.text_box_param.clear()

    def submit_save(self):
        """Event handler for the save filename text box."""
        filename = self.text_box_save.text()
        mech = self.plotted['mechanism']
        mech.save(filename=filename)
        self.text_box_save.clear()

    def plot_connecting_points_update(self, val):
        """
        Event handler for the joint connection points sliders.
        Update the linkage point parameters based on the slider values.
        """
        mechanism = self.plotted['mechanism']
        num_of_factors = mechanism.factorizations[0].number_of_factors

        # Update first factorization linkage points.
        for i in range(num_of_factors):
            # Divide by 100 to convert from slider integer to float.
            val0 = self.joint_sliders[2 * i].value() / 100.0
            val1 = self.joint_sliders[2 * i + 1].value() / 100.0
            mechanism.factorizations[0].linkage[i].set_point_by_param(0, val0)
            mechanism.factorizations[0].linkage[i].set_point_by_param(1, val1)

        # Update second factorization linkage points.
        for i in range(num_of_factors):
            base = 2 * num_of_factors
            val0 = self.joint_sliders[base + 2 * i].value() / 100.0
            val1 = self.joint_sliders[base + 2 * i + 1].value() / 100.0
            mechanism.factorizations[1].linkage[i].set_point_by_param(0, val0)
            mechanism.factorizations[1].linkage[i].set_point_by_param(1, val1)

        # Update the plot using the current value of the main slider.
        current_angle = self.move_slider.value() / 100.0
        self.plot_slider_update(current_angle)

    def plot_slider_update(self, val, t_param: float = None):
        """
        Event handler for the driving joint angle slider.
        Updates the link positions and (if enabled) the tool and tool frame.
        """
        mechanism = self.plotted['mechanism']
        if t_param is not None:
            t = t_param
        else:
            # Convert the joint angle (from the slider) to the t parameter.
            angle = val if isinstance(val, float) else (val / 100.0)
            t = mechanism.factorizations[0].joint_angle_to_t_param(angle)

        # Update the link connections.
        links = (mechanism.factorizations[0].direct_kinematics(t)
                 + mechanism.factorizations[1].direct_kinematics(t)[::-1])
        # Insert the last point again for closure.
        links.insert(0, links[-1])
        # Assume links is a list of (x,y,z) points.
        xs, ys, zs = zip(*links)
        # Update each line segment.
        for i, line in enumerate(self.lines):
            # Note: the line visual’s set_data() expects an (N,3) array.
            pts = np.array(
                [[xs[i], ys[i], zs[i]], [xs[i + 1], ys[i + 1], zs[i + 1]]])
            line.set_data(pos=pts)

        if self.show_tool:
            # Plot tool: get the tool triangle from the last points of each factorization.
            pt0 = mechanism.factorizations[0].direct_kinematics(t)[-1]
            pt1 = mechanism.factorizations[1].direct_kinematics(t)[-1]
            tool = mechanism.factorizations[0].direct_kinematics_of_tool(t,
                                                                         mechanism.tool_frame.dq2point_via_matrix())
            # Build the triangle: [pt0, tool, pt1]
            tool_triangle = np.array([pt0, tool, pt1])
            self.tool_plot.set_data(pos=tool_triangle)

            # Plot tool frame.
            # Evaluate the mechanism at t to get the pose.
            pose_dq = DualQuaternion(mechanism.evaluate(t))
            pose_matrix = TransfMatrix(pose_dq.dq2matrix()) * TransfMatrix(
                mechanism.tool_frame.dq2matrix())
            # Assume get_plot_data() returns three tuples for x, y, and z axes.
            x_vec, y_vec, z_vec = pose_matrix.get_plot_data()
            # Update the three arrow visuals.
            for arrow, vec in zip(self.pose_frame, [x_vec, y_vec, z_vec]):
                # Here we set the arrow’s data: starting at the mechanism’s translation and pointing in vec direction.
                origin = np.array(pose_matrix.t)
                end = origin + np.array(vec) * self.arrows_length
                arrow.set_data(pos=np.vstack([origin, end]))
        # Force an update of the canvas.
        self.canvas.update()






