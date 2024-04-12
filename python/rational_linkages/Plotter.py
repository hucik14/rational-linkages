from functools import wraps
from itertools import cycle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

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


class Plotter:
    def __init__(self,
                 interactive: bool = False,
                 jupyter_notebook: bool = False,
                 interval=(-1, 1),
                 steps=50,
                 arrows_length: float = 1.0,
                 joint_range_lim: float = 1.0):
        """
        Initialize the plotter

        :param interactive: activate interactive mode
        :param jupyter_notebook: activate jupyter notebook mode
        :param interval: interval for plotting
        :param steps: number of steps for plotting
        :param arrows_length: length of quiver arrows for poses and frames
        :param joint_range_lim: limit for joint sliders, will be +/- value
        """
        # use Qt5Agg backend for interactive plotting
        if interactive and not jupyter_notebook:
            matplotlib.use("Qt5Agg")

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")

        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_zlabel("Z-axis")
        self.ax.set_aspect("equal")

        # Initialize min/max variables
        self.min_x, self.max_x = float('inf'), float('-inf')
        self.min_y, self.max_y = float('inf'), float('-inf')
        self.min_z, self.max_z = float('inf'), float('-inf')

        if interactive:
            plt.subplots_adjust(
                top=1.0,
                bottom=0.16,
                left=0.32,
                right=0.935,
                hspace=0.2,
                wspace=0.2
            )
        else:
            plt.subplots_adjust(
                top=1.0,
                bottom=0.1,
                left=0.0,
                right=1.0,
                hspace=0.2,
                wspace=0.2
            )

        self.t_space = np.linspace(interval[0], interval[1], steps)
        self.steps = steps
        self.interactive = interactive
        self.jupyter_notebook = jupyter_notebook
        self.j_sliders_limit = joint_range_lim

        # length of quiver arrows for poses and frames
        self.arrows_length = arrows_length

        self.plotted = {}

    def plot(self, objects_to_plot, **kwargs):
        """
        Plot the object

        :param objects_to_plot: NormalizedLine, PointHomogeneous, RationalMechanism,
            MotionFactorization, DualQuaternion, TransfMatrix, RationalCurve,
            RationalBezier, MiniBall, or list of those
        :param kwargs: plotting options following matplotlib standards and syntax
        """
        # if list of objects, plot each object separately
        if isinstance(objects_to_plot, list):
            # check for label list
            label_list = kwargs.pop('label', None)

            for i, obj in enumerate(objects_to_plot):
                if label_list is not None:
                    kwargs['label'] = label_list[i]
                self._plot(obj, **kwargs)

        # if single object, plot it
        else:
            self._plot(objects_to_plot, **kwargs)

    def _plot(self, object_to_plot, **kwargs):
        """
        Plot the object

        :param object_to_plot: NormalizedLine, PointHomogeneous, RationalMechanism,
            MotionFactorization, DualQuaternion, TransfMatrix, RationalCurve, MiniBall,
            or RationalBezier
        :param kwargs: plotting options following matplotlib standards and syntax
        """
        type_to_plot = self.analyze_object(object_to_plot)

        match type_to_plot:
            case "is_line":
                self._plot_line(object_to_plot, **kwargs)
            case "is_point":
                self._plot_point(object_to_plot, **kwargs)
            case "is_motion_factorization":
                self._plot_motion_factorization(object_to_plot, **kwargs)
            case "is_dq":
                self._plot_dual_quaternion(object_to_plot, **kwargs)
            case "is_transf_matrix":
                self._plot_transf_matrix(object_to_plot, **kwargs)
            case "is_rational_curve":
                self._plot_rational_curve(object_to_plot, **kwargs)
            case "is_rational_bezier":
                self._plot_rational_bezier(object_to_plot, **kwargs)
            case "is_rational_mechanism":
                self._plot_rational_mechanism(object_to_plot, **kwargs)
            case "is_interactive":
                self._plot_interactive(object_to_plot, **kwargs)
            case "is_miniball":
                self._plot_miniball(object_to_plot, **kwargs)
            case "is_line_segment":
                self._plot_line_segment(object_to_plot, **kwargs)
            case "is_point_orbit":
                self._plot_point_orbit(object_to_plot, **kwargs)

    def analyze_object(self, object_to_plot):
        """
        Analyze the object to plot

        :param object_to_plot: NormalizedLine, PointHomogeneous, RationalMechanism,
            MotionFactorization, DualQuaternion, TransfMatrix, RationalCurve
            or RationalBezier

        :return: str - 'is_line', 'is_point', 'is_motion_factorization', 'is_dq' or
            'is_rational_mechanism'
        """
        if isinstance(object_to_plot, RationalMechanism) and not self.interactive:
            return "is_rational_mechanism"
        elif isinstance(object_to_plot, RationalMechanism) and self.interactive:
            return "is_interactive"
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
            raise TypeError(
                "Other types than NormalizedLine, PointHomogeneous, RationalMechanism, "
                "MotionFactorization or DualQuaternion not yet implemented"
            )

    @staticmethod
    def _plotting_decorator(func):
        """
        Decorator for plotting functions

        :param func: plotting function
        :return: decorated plotting function
        """
        @wraps(func)
        def _wrapper(self, *args, **kwargs):
            # use the plotting function
            func(self, *args, **kwargs)

            # decorate the plot - set aspect ratio and update legend
            self.ax.set_aspect("equal")
            if not self.jupyter_notebook:
                self.ax.legend()
        return _wrapper

    @_plotting_decorator
    def _plot_line(self, line: NormalizedLine, **kwargs):
        """
        Plot a line

        :param line: NormalizedLine
        :param kwargs: matplotlib options
        """
        if 'interval' in kwargs:
            interval = kwargs['interval']
            kwargs.pop('interval')
        else:
            interval = (-1, 1)

        line = line.get_plot_data(interval)

        if 'label' not in kwargs:
            kwargs['label'] = "line"
        else:
            mid_of_line = (line[:3] + line[3:]/2)
            self.ax.text(*mid_of_line, ' ' + kwargs['label'])

        self.ax.quiver(*line, **kwargs)

    @_plotting_decorator
    def _plot_point(self, point: PointHomogeneous, **kwargs):
        """
        Plot a point

        :param point: PointHomogeneous
        :param kwargs: matplotlib options
        """
        point = point.get_plot_data()

        if 'label' not in kwargs:
            kwargs['label'] = "point"
        else:
            self.ax.text(*point, ' ' + kwargs['label'])

        self.ax.scatter(*point, **kwargs)

    @_plotting_decorator
    def _plot_dual_quaternion(self, dq: DualQuaternion, **kwargs):
        """
        Plot a dual quaternion as a transformation

        :param dq: DualQuaternion
        :param kwargs: not used
        """
        matrix = TransfMatrix(dq.dq2matrix())
        self._plot_transf_matrix(matrix, **kwargs)

    @_plotting_decorator
    def _plot_transf_matrix(self, matrix: TransfMatrix, **kwargs):
        """
        Plot a transformation matrix

        :param transf_matrix: TransfMatrix
        :param kwargs: not used
        """
        x_vec, y_vec, z_vec = matrix.get_plot_data()

        if 'label' not in kwargs:
            kwargs['label'] = 'Tf'
        else:
            self.ax.text(*matrix.t, ' ' + kwargs['label'])

        self.ax.quiver(*x_vec, color="red", length=self.arrows_length)
        self.ax.quiver(*y_vec, color="green", length=self.arrows_length)
        self.ax.quiver(*z_vec, color="blue", length=self.arrows_length)

    @_plotting_decorator
    def _plot_rational_curve(self, curve: RationalCurve, **kwargs):
        """
        Plot a rational curve

        :param RationalCurve curve: RationalCurve
        :param kwargs: interval and matplotlib options
        """
        if 'interval' in kwargs:
            interval = kwargs['interval']
            kwargs.pop('interval')
        else:
            interval = (0, 1)

        x, y, z = curve.get_plot_data(interval, self.steps)

        if 'label' not in kwargs:
            kwargs['label'] = 'curve'

        self.ax.plot(x, y, z, **kwargs)

    @_plotting_decorator
    def _plot_rational_bezier(self,
                              bezier: RationalBezier,
                              plot_control_points: bool = True,
                              **kwargs):
        """
        Plot a rational Bezier curve

        :param bezier: RationalBezier
        :param plot_control_points: plot control points
        :param kwargs: interval and matplotlib options
        """
        if 'interval' in kwargs:
            interval = kwargs['interval']
            kwargs.pop('interval')
        else:
            interval = (0, 1)

        x, y, z, x_cp, y_cp, z_cp = bezier.get_plot_data(interval, self.steps)

        if 'label' not in kwargs:
            kwargs['label'] = "bezier curve"

        self.ax.plot(x, y, z, **kwargs)

        if plot_control_points:
            self.ax.plot(x_cp, y_cp, z_cp, "ro:")

    @_plotting_decorator
    def _plot_motion_factorization(self, factorization: MotionFactorization, **kwargs):
        """
        Plot a motion factorization

        :param factorization: MotionFactorization
        :param kwargs: t-curve parameter of driving joint axis and matplotlib options
        """
        if 't' in kwargs:
            t = kwargs['t']
            kwargs.pop('t')
        else:
            t = 0

        points = factorization.direct_kinematics(t)
        x, y, z = zip(*points)

        if 'label' not in kwargs:
            kwargs['label'] = "factorization"

        self.ax.plot(x, y, z, **kwargs)

    @_plotting_decorator
    def _plot_rational_mechanism(self, mechanism: RationalMechanism, **kwargs):
        """
        Plot a mechanism

        :param mechanism: RationalMechanism
        :param kwargs: t-curve parameter of driving joint axis and matplotlib options
        """
        self.plotted['mechanism'] = mechanism
        if 't' in kwargs:
            t = kwargs['t']
            kwargs.pop('t')
        else:
            t = 0

        # plot factorizations
        for factorization in mechanism.factorizations:
            self._plot_motion_factorization(factorization, t=t, **kwargs)

        # plot end effector triangle
        pts0 = mechanism.factorizations[0].direct_kinematics_of_tool_with_link(t, mechanism.tool_frame.dq2point_via_matrix())
        pts1 = mechanism.factorizations[1].direct_kinematics_of_tool_with_link(t, mechanism.tool_frame.dq2point_via_matrix())[::-1]
        ee_points = np.concatenate((pts0, pts1))

        if 'label' not in kwargs:
            kwargs['label'] = "end effector"

        x, y, z = zip(*ee_points)
        self.ax.plot(x, y, z, **kwargs)

        self._plot_tool_path(mechanism, **kwargs)

    @_plotting_decorator
    def _plot_tool_path(self, mechanism: RationalMechanism, **kwargs):
        # plot end effector path
        t_lin = np.linspace(0, 2 * np.pi, self.steps)
        t = [mechanism.factorizations[0].joint_angle_to_t_param(t_lin[i])
             for i in range(self.steps)]

        ee_points = [mechanism.factorizations[0].direct_kinematics_of_tool(
            t[i], mechanism.tool_frame.dq2point_via_matrix()) for i in range(self.steps)]

        kwargs['label'] = "tool path"

        x, y, z = zip(*ee_points)
        self.ax.plot(x, y, z, **kwargs)

    @_plotting_decorator
    def _plot_miniball(self, ball: MiniBall, **kwargs):
        """
        Plot a ball
        """
        if 'label' not in kwargs:
            kwargs['label'] = "Miniball of Bezier curve"
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.15

        x, y, z = ball.get_plot_data()

        self.ax.plot_surface(x, y, z, **kwargs)

    @_plotting_decorator
    def _plot_point_orbit(self, orbit: PointOrbit, **kwargs):
        """
        Plot a sphere of given point orbit
        """
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.15

        x, y, z = orbit.get_plot_data()

        self.ax.plot_surface(x, y, z, **kwargs)

    @_plotting_decorator
    def _plot_line_segment(self, segment: LineSegment, **kwargs):
        """
        Plot a line segment
        """
        x, y, z = segment.get_plot_data()

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.2

        self.ax.plot_surface(x, y, z, **kwargs)

    @_plotting_decorator
    def _plot_interactive(self,
                          mechanism: RationalMechanism,
                          show_tool: bool = True,
                          **kwargs):
        """
        Plot a mechanism in interactive mode

        :param RationalMechanism mechanism: RationalMechanism
        :param bool show_tool: show tool linkage and frame
        :param kwargs: matplotlib options
        """
        self.plotted['mechanism'] = mechanism
        self.show_tool = show_tool

        # plot the curve (tool path)
        self._plot_tool_path(mechanism, **kwargs)

        # append first slider that is the driving joint angle slider
        self.move_slider = self._init_slider()
        # set a text box that can be used to set the angle manually
        self.text_box_angle = TextBox(self.fig.add_axes([0.3, 0.055, 0.15, 0.05]),
                                      "Set angle [rad]: ", textalignment="right")
        # set a text box that can be used to set the t param manually
        self.text_box_param = TextBox(self.fig.add_axes([0.3, 0.11, 0.15, 0.05]),
                                      "Set param t [-]: ", textalignment="right")
        # text box to save files
        self.text_box_save = TextBox(self.fig.add_axes([0.3, 0.165, 0.15, 0.05]),
                                     "Save with filename: ", textalignment="right")

        # vertical sliders to control physical linkage position (connecting points)
        self.joint_sliders = []
        for i in range(mechanism.num_joints):
            slider0, slider1 = self._init_slider(idx=i,
                                                 j_sliders=self.joint_sliders,
                                                 slider_limit=self.j_sliders_limit)
            self.joint_sliders.append(slider0)
            self.joint_sliders.append(slider1)

        # set joint parameters to home configuration
        for i in range(mechanism.factorizations[0].number_of_factors):
            self.joint_sliders[2 * i].set_val(
                mechanism.factorizations[0].linkage[i].points_params[0])
            self.joint_sliders[1 + 2 * i].set_val(
                mechanism.factorizations[0].linkage[i].points_params[1])

        for i in range(mechanism.factorizations[1].number_of_factors):
            self.joint_sliders[
                2 * mechanism.factorizations[0].number_of_factors + 2 * i].set_val(
                mechanism.factorizations[1].linkage[i].points_params[0])
            self.joint_sliders[2 * mechanism.factorizations[
                0].number_of_factors + 1 + 2 * i].set_val(
                mechanism.factorizations[1].linkage[i].points_params[1])

        # initialize the linkages plot
        linestyles = cycle(['solid', 'dashdot'])
        self.lines = []
        for i in range(mechanism.num_joints * 2):
            # alter between solid (links) and dashdot (joints)
            linestyle = next(linestyles)
            line, = self.ax.plot([], [], [], linestyle=linestyle,
                                 color='black', marker='.')
            self.lines.append(line)

        if self.show_tool:
            # initialize the tool point interactive plot
            self.tool_plot, = self.ax.plot([], [], [], color="purple",
                                           linestyle="dashed", label="tool connection")
            # initialize the tool frame
            self.pose_frame = [self.ax.quiver([], [], [], [], [], [], color="red",
                                              length=self.arrows_length),
                               self.ax.quiver([], [], [], [], [], [], color="green",
                                              length=self.arrows_length),
                               self.ax.quiver([], [], [], [], [], [], color="blue",
                                              length=self.arrows_length)]

        def submit_angle(text):
            """Event handler for the text box"""
            val = float(text)
            val = val % (2 * np.pi)
            self.move_slider.set_val(val)

        def submit_parameter(text):
            """Event handler for the text box"""
            val = float(text)
            self.plot_slider_update(val, t_param=val)
            self.move_slider.set_val(mechanism.factorizations[0].t_param_to_joint_angle(val))

        def submit_save(text):
            """Event handler for the text box"""
            val = text
            mechanism.save(filename=val)

        # connect the slider and text box to the event handlers
        self.move_slider.on_changed(self.plot_slider_update)
        self.text_box_angle.on_submit(submit_angle)
        self.text_box_param.on_submit(submit_parameter)
        self.text_box_save.on_submit(submit_save)

        # joint physical placement sliders
        for i in range(4 * mechanism.factorizations[0].number_of_factors):
            self.joint_sliders[i].on_changed(self.plot_connecting_points_update)

        # initialize the plot in home configuration
        self.move_slider.set_val(0.0)

    @staticmethod
    def _init_slider(idx: int = None, j_sliders=None, slider_limit: float = 1.0):
        """
        Initialize the slider for interactive plotting

        :param int idx: index of the slider, first one is added automatically as joint
            angle slider
        :param list j_sliders: list of joint sliders
        :param float slider_limit: limit for joint sliders, will be +/- value

        :return: matplotlib slider
        """
        if idx is None:  # driving joint angle slider
            slider = Slider(
                ax=plt.axes([0.3, 0.01, 0.5, 0.05]),
                label="Joint angle [rad]: ",
                valmin=0.0,
                valmax=np.pi * 2,
                valinit=0.0,
                valstep=0.01,
            )
            return slider

        else:  # joint connection points sliders
            i = int(len(j_sliders) / 2)
            slider0 = Slider(
                ax=plt.axes([0.03 + i * 0.04, 0.25, 0.0225, 0.63]),
                #label="j{}.0".format(i),
                label="j{}".format(i),
                valmin=-slider_limit,
                valmax=slider_limit,
                valinit=0.0,
                orientation="vertical",
            )
            slider1 = Slider(
                ax=plt.axes([0.045 + i * 0.04, 0.25, 0.0225, 0.63]),
                #label="j{}.1".format(i),
                label="",
                valmin=-slider_limit,
                valmax=slider_limit,
                valinit=0.0,
                orientation="vertical",
            )
            return slider0, slider1

    def plot_connecting_points_update(self, val: tuple):
        """Event handler for the joint connection points sliders"""
        num_of_factors = self.plotted['mechanism'].factorizations[0].number_of_factors

        for i in range(num_of_factors):
            self.plotted['mechanism'].factorizations[0].linkage[i].set_point_by_param(0, self.joint_sliders[2 * i].val)
            self.plotted['mechanism'].factorizations[0].linkage[i].set_point_by_param(1, self.joint_sliders[1 + 2 * i].val)

        for i in range(num_of_factors):
            self.plotted['mechanism'].factorizations[1].linkage[i].set_point_by_param(0, self.joint_sliders[2 * num_of_factors + 2 * i].val)
            self.plotted['mechanism'].factorizations[1].linkage[i].set_point_by_param(1, self.joint_sliders[2 * num_of_factors + 1 + 2 * i].val)

        # update the plot
        self.plot_slider_update(self.move_slider.val)

    def plot_slider_update(self, val: float, t_param: float = None):
        """Event handler for the joint angle slider"""
        if t_param is not None:
            t = t_param
        else:
            # t parametrization for the driving joint
            t = self.plotted['mechanism'].factorizations[0].joint_angle_to_t_param(val)

        # plot links
        links = (self.plotted['mechanism'].factorizations[0].direct_kinematics(t)
                 + self.plotted['mechanism'].factorizations[1].direct_kinematics(t)[::-1])
        links.insert(0, links[-1])

        x, y, z = zip(*[links[j] for j in range(len(links))])

        for i, line in enumerate(self.lines):
            line.set_data_3d([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]])

        """
        xyz_coordinates = [(100 * xi, 100 * yi, 100 * zi) for xi, yi, zi in zip(x, y, z)]

        # Save XYZ coordinates to a CSV file
        pts_file_path = "xyz.pts"

        # Write the XYZ coordinates to the .pts file
        with open(pts_file_path, 'w') as pts_file:
            for point in xyz_coordinates:
                x, y, z = point
                pts_file.write(f"{x} {y} {z}\n")
        """

        if self.show_tool:
            # plot tool
            # use last point of each factorization
            tool_triangle = ([self.plotted['mechanism'].factorizations[0].direct_kinematics(t)[-1]]
                             + [self.plotted['mechanism'].factorizations[1].direct_kinematics(t)[-1]])
            # get tool point
            tool = self.plotted['mechanism'].factorizations[0].direct_kinematics_of_tool(
                t, self.plotted['mechanism'].tool_frame.dq2point_via_matrix())
            # add tool point to tool triangle
            tool_triangle.insert(1, tool)

            x, y, z = zip(*[tool_triangle[j] for j in range(len(tool_triangle))])
            self.tool_plot.set_data_3d(x, y, z)

            # plot tool frame
            pose_dq = DualQuaternion(self.plotted['mechanism'].evaluate(t))
            pose_matrix = TransfMatrix(pose_dq.dq2matrix()) * TransfMatrix(self.plotted['mechanism'].tool_frame.dq2matrix())

            x_vec, y_vec, z_vec = pose_matrix.get_plot_data()

            # remove old frame (quiver has no update method)
            for pose_arrow in self.pose_frame:
                pose_arrow.remove()
            # plot new frame
            self.pose_frame = [self.ax.quiver(*vec, color=color,
                                              length=self.arrows_length)
                               for vec, color in zip([x_vec, y_vec, z_vec],
                                                     ["red", "green", "blue"])]

        self.update_limits(self.ax)

        # update the plot
        if not self.jupyter_notebook:
            self.fig.canvas.draw_idle()
            self.fig.canvas.update()
            self.fig.canvas.flush_events()

    def show(self):
        """
        Show the plot
        """
        self.update_limits(self.ax)
        plt.show()

    def update_limits(self, ax):
        """
        Update the limits of the plot

        :param ax: matplotlib axes
        """
        # Inner function to update the minimum and maximum values
        def update_min_max(data):
            # Update min and max for x and y axes
            self.min_x, self.max_x = (min(self.min_x, np.min(data[:, 0])),
                                      max(self.max_x, np.max(data[:, 0])))
            self.min_y, self.max_y = (min(self.min_y, np.min(data[:, 1])),
                                      max(self.max_y, np.max(data[:, 1])))
            # Update min and max for z-axis if present
            if data.shape[1] > 2:
                self.min_z, self.max_z = (min(self.min_z, np.min(data[:, 2])),
                                          max(self.max_z, np.max(data[:, 2])))

        # Iterate over all artists in the Axes3D object
        for artist in ax.get_children():
            # Handle 3D scatter plots
            if isinstance(artist, matplotlib.collections.PathCollection):
                update_min_max(np.array(artist._offsets3d).T)
            # Handle 3D line plots
            elif hasattr(artist, '_verts3d'):
                update_min_max(np.array(artist._verts3d).T)
            # Handle 3D quiver plots
            elif hasattr(artist, '_segments3d'):
                for segment in artist._segments3d:
                    update_min_max(np.array(segment))
            # Handle other collection types (like PolyCollection for polygons)
            elif isinstance(artist, matplotlib.collections.Collection):
                for path in artist.get_paths():
                    for polygon in path.to_polygons():
                        update_min_max(np.array(polygon))

        # Set the updated limits to the axes
        ax.set_xlim3d(float(self.min_x), float(self.max_x))
        ax.set_ylim3d(float(self.min_y), float(self.max_y))
        ax.set_zlim3d(float(self.min_z), float(self.max_z))
        ax.set_aspect("equal")

