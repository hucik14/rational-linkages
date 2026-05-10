from functools import wraps
from itertools import cycle
from os import makedirs
from os.path import isdir, join
from warnings import warn

import numpy

from .DualQuaternion import DualQuaternion
from .Linkage import LineSegment
from .MiniBall import MiniBall
from .MotionFactorization import MotionFactorization
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous, PointOrbit
from .RationalBezier import RationalBezier
from .RationalCurve import RationalCurve
from .RationalMechanism import RationalMechanism
from .TransfMatrix import TransfMatrix

# Try importing GUI components
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox

except (ImportError, OSError):
    warn("Failed to import Matplotlib. Check the package installation.")

    matplotlib = None
    plt = None
    Slider = None
    TextBox = None


class PlotterMatplotlib:
    def __init__(self,
                 interactive: bool = False,
                 base=None,
                 jupyter_notebook: bool = False,
                 show_legend: bool = False,
                 show_controls: bool = True,
                 paper_visual: bool = False,
                 ticks_step: float = None,
                 interval: tuple = (-1, 1),
                 steps: int = 50,
                 arrows_length: float = 1.0,
                 joint_sliders_lim: float = 1.0):
        """
        Initialize the PlotterMatplotlib instance.

        Parameters
        ----------
        interactive : bool, optional
            Activate interactive mode.
        base : TransfMatrix or DualQuaternion, optional
            Base transformation for plotting. Must be a TransfMatrix or DualQuaternion instance.
        jupyter_notebook : bool, optional
            Activate Jupyter notebook mode.
        show_legend : bool, optional
            Show the legend in the plot.
        show_controls : bool, optional
            Show or hide the controls for interactive plotting.
        paper_visual : bool, optional
            Make the visual output suitable for a scientific paper.
        ticks_step : float, optional
            Step for ticks on axes. If None, automatic ticks will be used.
        interval : tuple, optional
            Interval for plotting. For curves, can be specified as 'closed' for full parametrization.
        steps : int, optional
            Number of steps for plotting.
        arrows_length : float, optional
            Length of quiver arrows for poses and frames.
        joint_sliders_lim : float, optional
            Limit for joint sliders; will be +/- this value.

        Notes
        -----
        Use the `with_poses` keyword argument in plotting methods to plot the poses
        along the curve.
        """

        # use interactive backend for interactive plotting
        if interactive and not jupyter_notebook:
            try:
                matplotlib.use("macosx")
            except:
                try:
                    matplotlib.use("QtAgg")
                except:
                    try:
                        matplotlib.use("qtagg")
                    except:
                        raise RuntimeError(
                            "Matplotlib backend error. Use Pyqtgraph backend instead."
                        )
        self.paper_visual = paper_visual
        self.ticks_step = ticks_step

        if self.paper_visual:
            font_size = 8
            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                "font.size": font_size,
                "axes.labelsize": font_size,
                "legend.fontsize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
            })

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_aspect("equal")

        if self.paper_visual:
            self.ax.set_xlabel("X [m]")
            self.ax.set_ylabel("Y [m]")
            self.ax.set_zlabel("Z [m]")
            self.fig.patch.set_facecolor('white')
            self.ax.set_facecolor('white')
            self.ax.set_proj_type('ortho')

            # transparent grid
            for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                axis.pane.set_facecolor((1, 1, 1, 1))
                axis.pane.set_edgecolor((1, 1, 1, 1))

            if self.ticks_step:
                from matplotlib.ticker import MultipleLocator
                self.ax.xaxis.set_major_locator(MultipleLocator(ticks_step))
                self.ax.yaxis.set_major_locator(MultipleLocator(ticks_step))
                self.ax.zaxis.set_major_locator(MultipleLocator(ticks_step))

        # Initialize min/max variables
        self.min_x, self.max_x = float('inf'), float('-inf')
        self.min_y, self.max_y = float('inf'), float('-inf')
        self.min_z, self.max_z = float('inf'), float('-inf')

        if base is not None:
            if isinstance(base, TransfMatrix):
                if not base.is_rotation():
                    raise ValueError("Given matrix is not proper rotation.")
                self.base = base
                self.base_arr = self.base.array()
            elif isinstance(base, DualQuaternion):
                self.base = TransfMatrix(base.dq2matrix())
                self.base_arr = self.base.array()
            else:
                raise TypeError("Base must be a TransfMatrix or DualQuaternion instance.")
        else:
            self.base = None
            self.base_arr = None

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

        self.t_space = numpy.linspace(interval[0], interval[1], steps)
        self.steps = steps
        self.legend = show_legend
        self.interactive = interactive
        self.jupyter_notebook = jupyter_notebook
        self.joint_sliders_lim = joint_sliders_lim
        self.show_controls = show_controls

        # length of quiver arrows for poses and frames
        self.arrows_length = arrows_length

        self.plotted = {}

    def plot(self, objects_to_plot, **kwargs):
        """
        Plot one or more objects.

        Parameters
        ----------
        objects_to_plot : NormalizedLine, PointHomogeneous, RationalMechanism, MotionFactorization, DualQuaternion, TransfMatrix, RationalCurve, RationalBezier, MiniBall, or list
            The object(s) to plot.

        Other Parameters
        ----------------
        with_poses : bool, optional
            If True, plot poses along a rational curve.
        interval : str or tuple, optional
            If 'closed', rational curve will be closed in the interval (tangent half-angle substitution).
        show_tool : bool, optional
            If True, plot the mechanism with tool frame.
        **kwargs
            Additional plotting options following matplotlib standards and syntax.
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
        Plot a single object.

        Parameters
        ----------
        object_to_plot : NormalizedLine, PointHomogeneous, RationalMechanism, MotionFactorization, DualQuaternion, TransfMatrix, RationalCurve, MiniBall, or RationalBezier
            The object to plot.
        **kwargs
            Additional plotting options following matplotlib standards and syntax.
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
        Analyze the object to determine its type for plotting.

        Parameters
        ----------
        object_to_plot : NormalizedLine, PointHomogeneous, RationalMechanism, MotionFactorization, DualQuaternion, TransfMatrix, RationalCurve, RationalBezier
            The object to analyze.

        Returns
        -------
        str
            One of: 'is_line', 'is_point', 'is_motion_factorization', 'is_dq', 'is_rational_mechanism', etc.
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
        Decorator for plotting functions.

        Parameters
        ----------
        func : callable
            The plotting function to decorate.

        Returns
        -------
        callable
            The decorated plotting function.
        """
        @wraps(func)
        def _wrapper(self, *args, **kwargs):
            # use the plotting function
            func(self, *args, **kwargs)

            # decorate the plot - set aspect ratio and update legend
            self.ax.set_aspect("equal")

            # show legend
            if self.legend:
                self.ax.legend()
        return _wrapper

    @_plotting_decorator
    def plot_axis_between_two_points(self,
                                     p0: PointHomogeneous,
                                     p1: PointHomogeneous,
                                     **kwargs):
        """
        Plot a line between two points.

        Parameters
        ----------
        p0 : PointHomogeneous
            First point.
        p1 : PointHomogeneous
            Second point.
        **kwargs
            Matplotlib options.
        """
        line = numpy.concatenate((p0.normalized_euclidean(),
                               p1.normalized_euclidean() - p0.normalized_euclidean()))

        if 'label' not in kwargs:
            kwargs['label'] = "line"
        else:
            mid_of_line = (line[:3] + line[3:]/2)
            self.ax.text(*mid_of_line, ' ' + kwargs['label'])

        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = '-.'

        self.ax.quiver(*line, **kwargs)

    @_plotting_decorator
    def plot_line_segments_between_points(self,
                                          points: list[PointHomogeneous],
                                          **kwargs):
        """
        Plot line segments between a list of points.

        Parameters
        ----------
        points : list of PointHomogeneous
            List of points.
        **kwargs
            Matplotlib options.
        """
        pts = [p.normalized_euclidean() for p in points]

        x_coords = [pt[0] for pt in pts]
        y_coords = [pt[1] for pt in pts]
        z_coords = [pt[2] for pt in pts]

        if 'label' not in kwargs:
            kwargs['label'] = "segment"

        self.ax.plot(x_coords, y_coords, z_coords, **kwargs)

    @_plotting_decorator
    def plot_plane(self,
                   normal: numpy.ndarray,
                   point: numpy.ndarray,
                   xlim: tuple[float, float] = (-1, 1),
                   ylim: tuple[float, float] = (-1, 1),
                   **kwargs):
        """
        Plot a plane in 3D given a normal vector and a point on the plane.

        Parameters
        ----------
        normal : numpy.ndarray
            Normal vector of the plane.
        point : numpy.ndarray
            Point on the plane.
        xlim : tuple of float, optional
            X-axis limits.
        ylim : tuple of float, optional
            Y-axis limits.
        **kwargs
            Matplotlib options.
        """
        normal = numpy.asarray(normal)
        point = numpy.asarray(point)

        # Extract the normal vector components
        a, b, c = normal

        # Calculate d in the plane equation ax + by + cz = d
        d = numpy.dot(normal, point)

        # Create a grid of x and y values
        x = numpy.linspace(*xlim, 20)
        y = numpy.linspace(*ylim, 20)
        x, y = numpy.meshgrid(x, y)

        # Solve for z in the plane equation
        z = (d - a * x - b * y) / c

        if 'label' not in kwargs:
            kwargs['label'] = "plane"
        else:
            self.ax.text(*point, ' ' + kwargs['label'])

        self.ax.plot_surface(x, y, z, alpha=0.2, rstride=100, cstride=100)

    @_plotting_decorator
    def _plot_line(self, line: NormalizedLine, **kwargs):
        """
        Plot a line.

        Parameters
        ----------
        line : NormalizedLine
            The line to plot.
        **kwargs
            Matplotlib options.
        """
        if 'interval' in kwargs:
            interval = kwargs['interval']
            kwargs.pop('interval')
        else:
            interval = (-1, 1)

        line = line.get_plot_data(interval)

        if 'label' not in kwargs:
            kwargs['label'] = "axis"
        else:
            mid_of_line = (line[:3] + line[3:]/2)
            self.ax.text(*mid_of_line, ' ' + kwargs['label'])

        self.ax.quiver(*line, **kwargs)

    @_plotting_decorator
    def _plot_point(self, point: PointHomogeneous, **kwargs):
        """
        Plot a point.

        Parameters
        ----------
        point : PointHomogeneous
            The point to plot.
        **kwargs
            Matplotlib options.
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
        Plot a dual quaternion as a transformation.

        Parameters
        ----------
        dq : DualQuaternion
            The dual quaternion to plot.
        **kwargs
            Not used.
        """
        matrix = TransfMatrix(dq.dq2matrix())
        self._plot_transf_matrix(matrix, **kwargs)

    @_plotting_decorator
    def _plot_transf_matrix(self, matrix: TransfMatrix, **kwargs):
        """
        Plot a transformation matrix.

        Parameters
        ----------
        matrix : TransfMatrix
            The transformation matrix to plot.
        **kwargs
            Not used.
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
        Plot a rational curve.

        Parameters
        ----------
        curve : RationalCurve
            The rational curve to plot.
        **kwargs
            Interval and matplotlib options.
        """
        if 'interval' in kwargs:
            interval = kwargs['interval']
            kwargs.pop('interval')
        else:
            interval = (0, 1)

        if 'with_poses' in kwargs and kwargs['with_poses'] is True:
            kwargs.pop('with_poses')

            if interval == 'closed':
                # tangent half-angle substitution for closed curves
                t_space = numpy.tan(numpy.linspace(-numpy.pi / 2, numpy.pi / 2, 51))
            else:
                t_space = numpy.linspace(interval[0], interval[1], 50)

            for t in t_space:
                pose_dq = DualQuaternion(curve.evaluate(t))
                self._plot_dual_quaternion(pose_dq)

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
        Plot a rational Bezier curve.

        Parameters
        ----------
        bezier : RationalBezier
            The rational Bezier curve to plot.
        plot_control_points : bool, optional
            If True, plot control points.
        **kwargs
            Interval and matplotlib options.
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
        Plot a motion factorization.

        Parameters
        ----------
        factorization : MotionFactorization
            The motion factorization to plot.
        **kwargs
            t-curve parameter of driving joint axis and matplotlib options.
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
        Plot a mechanism.

        Parameters
        ----------
        mechanism : RationalMechanism
            The mechanism to plot.
        **kwargs
            t-curve parameter of driving joint axis and matplotlib options.
        """
        self.plotted['mechanism'] = mechanism

        show_tool = kwargs.pop('show_tool', False)
        t = kwargs.pop('t', 0)

        self._plot_tool_path(mechanism, **kwargs)

        # plot factorizations
        for factorization in mechanism.factorizations:
            self._plot_motion_factorization(factorization, t=t, **kwargs, color='black')

        if show_tool:
            # plot end effector triangle
            pts0 = mechanism.factorizations[0].direct_kinematics_of_tool_with_link(
                t, mechanism.tool_frame.dq2point_via_matrix())
            pts1 = mechanism.factorizations[1].direct_kinematics_of_tool_with_link(
                t, mechanism.tool_frame.dq2point_via_matrix())[::-1]
            ee_points = numpy.concatenate((pts0, pts1))

        if 'label' not in kwargs:
            kwargs['label'] = "end effector"

        x, y, z = zip(*ee_points)
        self.ax.plot(x, y, z, color='black', **kwargs)

    @_plotting_decorator
    def _plot_tool_path(self, mechanism: RationalMechanism, **kwargs):
        """
        Plot the end effector path for a mechanism.

        Parameters
        ----------
        mechanism : RationalMechanism
            The mechanism whose tool path is to be plotted.
        **kwargs
            Matplotlib options.
        """
        # plot end effector path
        t_lin = numpy.linspace(0, 2 * numpy.pi, self.steps)
        t = [mechanism.factorizations[0].joint_angle_to_t_param(t_lin[i])
             for i in range(self.steps)]

        ee_points = [mechanism.factorizations[0].direct_kinematics_of_tool(
            t[i], mechanism.tool_frame.dq2point_via_matrix()) for i in range(self.steps)]

        if self.base_arr is not None:
            # transform points to base frame
            ee_points = [self.base_arr @ numpy.insert(p, 0, 1)
                         for p in ee_points]
            # normalize
            ee_points = [p[1:4]/p[0] for p in ee_points]

        kwargs['label'] = "tool path"

        x, y, z = zip(*ee_points)
        self.ax.plot(x, y, z, **kwargs, color='lightgray', lw=2)

    @_plotting_decorator
    def _plot_miniball(self, ball: MiniBall, **kwargs):
        """
        Plot a ball (miniball).

        Parameters
        ----------
        ball : MiniBall
            The miniball to plot.
        **kwargs
            Matplotlib options.
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
        Plot a sphere of a given point orbit.

        Parameters
        ----------
        orbit : PointOrbit
            The point orbit to plot.
        **kwargs
            Matplotlib options.
        """
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.15

        x, y, z = orbit.get_plot_data_mpl()

        self.ax.plot_surface(x, y, z, **kwargs)

    @_plotting_decorator
    def _plot_line_segment(self, segment: LineSegment, **kwargs):
        """
        Plot a line segment.

        Parameters
        ----------
        segment : LineSegment
            The line segment to plot.
        **kwargs
            Matplotlib options.
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
        Plot a mechanism in interactive mode.

        Parameters
        ----------
        mechanism : RationalMechanism
            The mechanism to plot.
        show_tool : bool, optional
            If True, show tool linkage and frame.
        **kwargs
            Matplotlib options.
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
                                                 slider_limit=self.joint_sliders_lim)
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

            # normalize angle to [0, 2*pi]
            if val >= 0:
                val = val % (2 * numpy.pi)
            else:
                val = (val % (2 * numpy.pi)) - numpy.pi
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
        Initialize the slider for interactive plotting.

        Parameters
        ----------
        idx : int, optional
            Index of the slider. The first one is added automatically as the joint angle slider.
        j_sliders : list, optional
            List of joint sliders.
        slider_limit : float, optional
            Limit for joint sliders; will be +/- this value.

        Returns
        -------
        Slider or tuple of Slider
            Matplotlib slider(s).
        """
        if idx is None:  # driving joint angle slider
            slider = Slider(
                ax=plt.axes([0.3, 0.01, 0.5, 0.05]),
                label="Joint angle [rad]: ",
                valmin=0.0,
                valmax=numpy.pi * 2,
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
        """
        Event handler for the joint connection points sliders.

        Parameters
        ----------
        val : tuple
            The value(s) from the joint sliders.
        """
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
        """
        Event handler for the joint angle slider.

        Parameters
        ----------
        val : float
            The value from the joint angle slider.
        t_param : float, optional
            The t parameter for the driving joint, if provided.
        """
        if t_param is not None:
            t = t_param
        else:
            # t parametrization for the driving joint
            t = self.plotted['mechanism'].factorizations[0].joint_angle_to_t_param(val)

        # plot links
        links = (self.plotted['mechanism'].factorizations[0].direct_kinematics(t)
                 + self.plotted['mechanism'].factorizations[1].direct_kinematics(t)[::-1])
        links.insert(0, links[-1])

        if self.base_arr is not None:
            # transform points to base frame
            links = [self.base_arr @ numpy.insert(p, 0, 1) for p in links]
            # normalize
            links = [p[1:4]/p[0] for p in links]

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

            if self.base_arr is not None:
                # transform points to base frame
                tool_triangle = [self.base_arr @ numpy.insert(p, 0, 1)
                                 for p in tool_triangle]
                # normalize
                tool_triangle = [p[1:4]/p[0] for p in tool_triangle]

            x, y, z = zip(*[tool_triangle[j] for j in range(len(tool_triangle))])
            self.tool_plot.set_data_3d(x, y, z)

            # plot tool frame
            pose_dq = DualQuaternion(self.plotted['mechanism'].evaluate(t))
            pose_matrix = TransfMatrix(pose_dq.dq2matrix()) * TransfMatrix(
                self.plotted['mechanism'].tool_frame.dq2matrix())

            if self.base_arr is not None:
                # transform pose in respect to base frame
                pose_matrix = self.base * pose_matrix

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
        Show the plot.
        """
        self.update_limits(self.ax)
        plt.show()

    def update_limits(self, ax):
        """
        Update the limits of the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to update limits for.
        """
        # Inner function to update the minimum and maximum values
        def update_min_max(data):
            # Update min and max for x and y axes
            self.min_x, self.max_x = (min(self.min_x, numpy.min(data[:, 0])),
                                      max(self.max_x, numpy.max(data[:, 0])))
            self.min_y, self.max_y = (min(self.min_y, numpy.min(data[:, 1])),
                                      max(self.max_y, numpy.max(data[:, 1])))
            # Update min and max for z-axis if present
            if data.shape[1] > 2:
                self.min_z, self.max_z = (min(self.min_z, numpy.min(data[:, 2])),
                                          max(self.max_z, numpy.max(data[:, 2])))

        # Iterate over all artists in the Axes3D object
        for artist in ax.get_children():
            # Handle 3D scatter plots
            if isinstance(artist, matplotlib.collections.PathCollection):
                update_min_max(numpy.array(artist._offsets3d).T)
            # Handle 3D line plots
            elif hasattr(artist, '_verts3d'):
                update_min_max(numpy.array(artist._verts3d).T)
            # Handle 3D quiver plots
            elif hasattr(artist, '_segments3d'):
                for segment in artist._segments3d:
                    update_min_max(numpy.array(segment))
            # Handle other collection types (like PolyCollection for polygons)
            elif isinstance(artist, matplotlib.collections.Collection):
                for path in artist.get_paths():
                    for polygon in path.to_polygons():
                        update_min_max(numpy.array(polygon))

        # Set the updated limits to the axes
        ax.set_xlim3d(float(self.min_x), float(self.max_x))
        ax.set_ylim3d(float(self.min_y), float(self.max_y))
        ax.set_zlim3d(float(self.min_z), float(self.max_z))
        ax.set_aspect("equal")

    def animate(self,
                number_of_frames: int = 10,
                file_type: str = "png",
                filename_prefix: str = "frame_",
                output_dir: str = "animation_frames"):
        """
        Animate the mechanism and save frames in a folder.

        PNG is the default file type, PDF is also supported.

        Parameters
        ----------
        number_of_frames : int, optional
            Number of time steps (frames).
        file_type : str, optional
            File type to save the frames ('pdf', 'png').
        filename_prefix : str, optional
            Prefix for the output filenames.
        output_dir : str, optional
            Directory where the frames should be saved.
        """
        # check if the file_type is supported
        if file_type not in plt.gcf().canvas.get_supported_filetypes():
            raise ValueError(f"Unsupported file type {file_type}")

        # if the output directory does not exist, create it
        if not isdir(output_dir):
            makedirs(output_dir)

        t_angle = numpy.linspace(0, 2 * numpy.pi, number_of_frames)

        # perform the animation once to scale the plot for equal axes limits
        for i, val in enumerate(t_angle):
            self.plot_slider_update(val)

        # save the frames
        for i, val in enumerate(t_angle):
            self.plot_slider_update(val)
            self.fig.savefig(
                join(output_dir, f"{filename_prefix}{i}.{file_type}"))

        print("Animation frames saved successfully in folder: ", output_dir)

    def animate_angles(self, list_of_angles: list, sleep_time: float = 1.0):
        """
        Animate the mechanism passing through a list of joint angles.

        Parameters
        ----------
        list_of_angles : list
            List of joint angles.
        sleep_time : float, optional
            Time to wait between each frame (in seconds).
        """
        from time import sleep  # lazy import

        t_angle = list_of_angles

        for i, val in enumerate(t_angle):
            self.plot_slider_update(val)
            sleep(sleep_time)

    def save_image(self, filename: str, file_type: str = "png"):
        """
        Save the current canvas to a file.

        Parameters
        ----------
        filename : str
            Name of the file (without extension).
        file_type : str, optional
            File type to save the frames ('pdf', 'png').
        """
        # check if the file_type is supported
        if file_type not in plt.gcf().canvas.get_supported_filetypes():
            raise ValueError(f"Unsupported file type {file_type}")

        self.fig.savefig(filename + "." + file_type)
        print("Canvas saved successfully as: ", filename + "." + file_type)

    def trigger_controls_visibility(self):
        """
        Toggle the visibility of controls for interactive plotting.

        Returns
        -------
        None
        """
        self.show_controls = not self.show_controls

        for control in [self.move_slider, self.text_box_angle, self.text_box_param,
                        self.text_box_save] + self.joint_sliders:
            control.ax.set_visible(self.show_controls)

        return None