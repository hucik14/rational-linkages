import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

from functools import wraps

from DualQuaternion import DualQuaternion
from NormalizedLine import NormalizedLine
from PointHomogeneous import PointHomogeneous
from MotionFactorization import MotionFactorization
from RationalMechanism import RationalMechanism
from TransfMatrix import TransfMatrix
from RationalCurve import RationalCurve
from RationalBezier import RationalBezier


class Plotter:
    def __init__(self, interval=(-1, 1), steps=50, interactive: bool = False):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")

        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_zlabel("Z-axis")
        self.ax.set_aspect("equal")

        self.t_space = np.linspace(interval[0], interval[1], steps)
        self.steps = steps
        self.interactive = interactive


    def plot(self, object_to_plot, **kwargs):
        """
        Plot the object

        :param object_to_plot: NormalizedLine, PointHomogeneous, RationalMechanism,
        MotionFactorization, DualQuaternion, TransfMatrix, RationalCurve
        or RationalBezier
        :param kwargs: plotting options following matplotlib standards and syntax

        :return: matplotlib axis
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

    def _plot_dual_quaternion(self, dq: DualQuaternion, **kwargs):
        """
        Plot a dual quaternion as a transformation

        :param dq: DualQuaternion
        :param kwargs: not used
        """
        from TransfMatrix import TransfMatrix
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

        self.ax.quiver(*x_vec, color="red")
        self.ax.quiver(*y_vec, color="green")
        self.ax.quiver(*z_vec, color="blue")

    @_plotting_decorator
    def _plot_rational_curve(self, curve: RationalCurve, **kwargs):
        """
        Plot a rational curve

        :param curve: RationalCurve
        :param kwargs: interval and matplotlib options
        """
        if 'interval' in kwargs:
            interval = kwargs['interval']
            kwargs.pop('interval')
        else:
            interval = (0, 1)

        x, y, z = curve.get_plot_data(interval, self.steps)

        if 'label' not in kwargs:
            kwargs['label'] = "curve"

        self.ax.plot(x, y, z, **kwargs)

    @_plotting_decorator
    def _plot_rational_bezier(self, bezier: RationalBezier, **kwargs):
        """
        Plot a rational Bezier curve

        :param bezier: RationalBezier
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
        if 't' in kwargs:
            t = kwargs['t']
            kwargs.pop('t')
        else:
            t = 0

        # plot factorizations
        for factorization in mechanism.factorizations:
            self._plot_motion_factorization(factorization, t=t, **kwargs)

        # plot end effector triangle
        pts0 = mechanism.factorizations[0].direct_kinematics_of_end_effector_with_link(t, mechanism.end_effector.array())
        pts1 = mechanism.factorizations[1].direct_kinematics_of_end_effector_with_link(t, mechanism.end_effector.array())[::-1]
        ee_points = np.concatenate((pts0, pts1))

        if 'label' not in kwargs:
            kwargs['label'] = "end effector"

        x, y, z = zip(*ee_points)
        self.ax.plot(x, y, z, **kwargs)

        self._plot_tool_path(mechanism, **kwargs)

    def _plot_tool_path(self, mechanism: RationalMechanism, **kwargs):
        # plot end effector path
        t_lin = np.linspace(0, 2 * np.pi, self.steps)
        t = [mechanism.factorizations[0].joint_angle_to_t_param(t_lin[i])
             for i in range(self.steps)]

        ee_points = [mechanism.factorizations[0].direct_kinematics_of_end_effector(
            t[i], mechanism.end_effector.array()) for i in range(self.steps)]

        kwargs['label'] = "motion curve"

        x, y, z = zip(*ee_points)
        self.ax.plot(x, y, z, **kwargs)

    @_plotting_decorator
    def _plot_interactive(self, mechanism: RationalMechanism, **kwargs):
        """
        Plot a mechanism in interactive mode

        :param RationalMechanism mechanism: RationalMechanism
        :param kwargs: matplotlib options
        """
        # plot the curve (tool path)
        self._plot_tool_path(mechanism, **kwargs)

        # initialize the sliders
        self.sliders = []
        # append first slider that is the driving joint angle slider
        self.sliders.append(self._init_slider())
        # set a text box that can be used to set the angle manually
        self.text_box = TextBox(self.fig.add_axes([0.2, 0.06, 0.15, 0.05]),
                                "Set angle [rad]: ", textalignment="center")

        # initialize the linkages plot
        self.link_plot, = self.ax.plot([], [], [], color="black")
        # initialize the tool point interactive plot
        self.tool_plot, = self.ax.plot([], [], [], color="red")
        # initialize the tool frame
        self.pose_frame = [self.ax.quiver([], [], [], [], [], [], color="red"),
                           self.ax.quiver([], [], [], [], [], [], color="green"),
                           self.ax.quiver([], [], [], [], [], [], color="blue")]

        def plot_slider_update(val):
            """Event handler for the joint angle slider"""
            # t parametrization for the driving joint
            t = mechanism.factorizations[0].joint_angle_to_t_param(val)

            # plot links
            links = (mechanism.factorizations[0].direct_kinematics(t)
                     + mechanism.factorizations[1].direct_kinematics(t)[::-1])

            x, y, z = zip(*[links[j] for j in range(len(links))])
            self.link_plot.set_data_3d(x, y, z)

            # plot tool
            # use last point of each factorization
            tool_triangle = ([mechanism.factorizations[0].direct_kinematics(t)[-1]]
                             + [mechanism.factorizations[1].direct_kinematics(t)[-1]])
            # get tool point
            tool = mechanism.factorizations[0].direct_kinematics_of_end_effector(
                t, mechanism.end_effector.dq2point_homogeneous())
            # add tool point to tool triangle
            tool_triangle.insert(1, tool)

            x, y, z = zip(*[tool_triangle[j] for j in range(len(tool_triangle))])
            self.tool_plot.set_data_3d(x, y, z)

            # plot tool frame
            pose_dq = DualQuaternion(mechanism.evaluate(t))
            pose_matrix = TransfMatrix(pose_dq.dq2matrix())

            x_vec, y_vec, z_vec = pose_matrix.get_plot_data()

            # remove old frame (quiver has no update method)
            for pose_arrow in self.pose_frame:
                pose_arrow.remove()
            # plot new frame
            self.pose_frame = [self.ax.quiver(*vec, color=color) for vec, color in
                               zip([x_vec, y_vec, z_vec], ["red", "green", "blue"])]

            # update the plot
            self.fig.canvas.draw_idle()
            self.fig.canvas.update()
            self.fig.canvas.flush_events()

        def submit_angle(text):
            """Event handler for the text box"""
            val = float(text)
            val = val % (2 * np.pi)
            self.sliders[0].set_val(val)

        # connect the slider and text box to the event handlers
        self.sliders[0].on_changed(plot_slider_update)
        self.text_box.on_submit(submit_angle)

        # initialize the plot in home configuration
        self.sliders[0].set_val(0.0)

    @staticmethod
    def _init_slider(idx: int = None):
        """
        Initialize the slider for interactive plotting

        :param int idx: index of the slider, first one is added automatically as joint
        angle slider

        :return: matplotlib slider
        """
        if idx is None:  # driving joint angle slider
            slider = Slider(
                ax=plt.axes([0.2, 0.01, 0.5, 0.05]),
                label="Joint angle [rad]: ",
                valmin=0.0,
                valmax=np.pi * 2,
                valinit=0.0,
                valstep=0.01,
            )
        else:  # joint connection points sliders
            slider = Slider(
                ax=plt.axes([0.3, 0.2, 0.5, 0.05]),
                label="Driving joint angle in rad",
                valmin=-2.0,
                valmax=2.0,
                valinit=0.0,
                valstep=0.1,
            )
        return slider
