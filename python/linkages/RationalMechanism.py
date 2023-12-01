import numpy as np
from copy import deepcopy

from RationalCurve import RationalCurve
from DualQuaternion import DualQuaternion
from matplotlib import pyplot as plt
from TransfMatrix import TransfMatrix
from matplotlib.widgets import Slider
from MotionFactorization import MotionFactorization


class RationalMechanism(RationalCurve):
    """
    Class representing rational mechanisms in dual quaternion space.
    """

    def __init__(self, factorizations: list[MotionFactorization],
                 end_effector: DualQuaternion = None):
        """
        Initializes a RationalMechanism object
        """
        super().__init__(factorizations[0].set_of_polynomials)
        self.factorizations = factorizations

        self.is_linkage = True if len(self.factorizations) == 2 else False

        self.end_effector = (
            DualQuaternion(self.evaluate(0, inverted_part=True))
            if end_effector is None else end_effector)

    def get_dh_params(self, alpha_form: str = "cos_alpha") -> tuple:
        """
        Get the Denavit-Hartenberg parameters of the linkage.

        :param alpha_form: str - form of the returned alpha parameter, can be
            "cos_alpha", "deg", or "rad"

        :return: tuple (d, a, alpha)
        """
        from NormalizedLine import NormalizedLine
        # Combine factorizations to get the linkage
        linkage = self.factorizations[0] + self.factorizations[1]
        # Create NormalizedLine objects for each axis rotation
        lines = [NormalizedLine(linkage.dq_axes[i].dq2screw())
                 for i in range(len(linkage.dq_axes))]

        d = []
        a = []
        alpha = []

        # Calculate Denavit-Hartenberg parameters for each pair of axis rotations
        pts_prev = lines[-1].common_perpendicular_to_other_line(lines[0])[0]
        for i in range(len(lines) - 1):
            pts, a_i, al_i = lines[i].common_perpendicular_to_other_line(lines[i+1])
            d_i = lines[i].get_point_param(pts_prev[1]) - lines[i].get_point_param(pts[0])
            pts_prev = deepcopy(pts)

            d.append(d_i)
            a.append(a_i)
            alpha.append(al_i)

        # Calculate Denavit-Hartenberg parameters for the last pair
        pts, a_i, al_i = lines[-1].common_perpendicular_to_other_line(lines[0])
        d_i = lines[-1].get_point_param(pts_prev[1]) - lines[-1].get_point_param(pts[0])

        d.append(d_i)
        a.append(a_i)
        alpha.append(al_i)

        # Convert alpha to the specified form
        match alpha_form:
            case "cos_alpha":
                pass
            case "deg":
                alpha = [np.degrees(np.arccos(alpha_i)) for alpha_i in alpha]
            case "rad":
                alpha = [np.arccos(alpha_i) for alpha_i in alpha]
            case _:
                raise ValueError("alpha_form must be cos_alpha, deg or rad")

        return d, a, alpha

    def plot(self, interval=(-1, 1), steps=50, ax=None, line_style=None) -> plt.axes:
        """
        Plot the curve in 2D or 3D, based on the dimension of the curve

        :param interval: tuple - interval of the parameter t
        :param steps: int - number of steps in the interval
        :param ax: existing matplotlib axis
        :param line_style: str - line style of the plot

        :return: matplotlib axis
        """
        # t = sp.Symbol('t')
        t_space = np.linspace(interval[0], interval[1], steps)

        # obtain the poses of the curve in the interval for given steps
        curve_poses = [DualQuaternion(self.evaluate(t_space[i])) for i in range(steps)]
        curve_poses_inv = [
            DualQuaternion(self.evaluate(t_space[i], inverted_part=True))
            for i in range(steps)
        ]

        curve_poses = curve_poses + curve_poses_inv[::-1]

        for i, pose in enumerate(curve_poses):
            curve_poses[i] = TransfMatrix((pose * self.end_effector).dq2matrix())

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        else:
            ax = ax
            fig = ax.figure

        # plot only path of curve with no orientation
        x, y, z = zip(*[curve_poses[i].t for i in range(len(curve_poses))])
        ax.plot(x, y, z, linestyle=":", color="red")

        # plot axes in home position
        for i, factorization in enumerate(self.factorizations):
            linestyle = "-." if i == 0 else ":"
            for j in range(factorization.number_of_factors):
                ax = factorization.dq_axes[j].plot_as_line(
                    (-0.5, 0.5), ax=ax, line_style=linestyle
                )

        if steps < 20:
            poses = [curve_poses[i].plot() for i in range(len(curve_poses))]
        else:
            poses = [
                curve_poses[i].plot()
                for i in range(0, len(curve_poses), int(steps / 5))
            ]

        arrow_length = 0.2
        for i in range(len(poses)):
            x, y, z, u, v, w = poses[i][0]
            ax.quiver(
                x,
                y,
                z,
                u,
                v,
                w,
                length=arrow_length,
                normalize=False,
                color="red",
                alpha=0.5,
            )
            x, y, z, u, v, w = poses[i][1]
            ax.quiver(
                x,
                y,
                z,
                u,
                v,
                w,
                length=arrow_length,
                normalize=False,
                color="green",
                alpha=0.5,
            )
            x, y, z, u, v, w = poses[i][2]
            ax.quiver(
                x,
                y,
                z,
                u,
                v,
                w,
                length=arrow_length,
                normalize=False,
                color="blue",
                alpha=0.5,
            )

        # Plotting widgets
        self.t_slider = Slider(
            ax=plt.axes([0.3, 0.1, 0.5, 0.05]),
            label="Timestep parameter t",
            #valmin=-1.0,
            #valmax=1.0,
            valmin=0.0,
            valmax=6.28,
            valinit=0.0,
            valstep=0.01,
        )

        self.t_inf_slider = Slider(
            ax=plt.axes([0.3, 0.05, 0.5, 0.05]),
            label="Part of t via infinity",
            valmin=-1.0,
            valmax=1.0,
            valinit=0.0,
            valstep=0.01,
        )

        (link_plot,) = ax.plot([], [], [], color="black")
        (end_effector_plot,) = ax.plot([], [], [], color="red", marker="o")

        def plot_slider_update(val):
            # t parametrization for the driving joint
            t = self.factorizations[0].joint_angle_to_t_param(val)

            if self.is_linkage:
                # joint positions, the other factorization is in reversed order
                _link = (
                    self.factorizations[0].direct_kinematics(t)
                    + self.factorizations[1].direct_kinematics(t)[::-1]
                )
                _ee_triangle = ([self.factorizations[0].direct_kinematics(t)[-1]]
                                + [self.factorizations[1].direct_kinematics(t)[-1]])
            else:
                _link = self.factorizations[0].direct_kinematics(t)
                _ee_triangle = [_link[-1]]

            _x, _y, _z = zip(*[_link[j] for j in range(len(_link))])
            link_plot.set_data_3d(_x, _y, _z)

            _ee = self.factorizations[0].direct_kinematics_of_tool(
                t, self.end_effector.dq2point_homogeneous()
            )
            _ee_triangle.insert(1, _ee)
            _x, _y, _z = zip(*[_ee_triangle[j] for j in range(len(_ee_triangle))])
            end_effector_plot.set_data_3d(_x, _y, _z)

            fig.canvas.draw_idle()
            fig.canvas.update()
            fig.canvas.flush_events()

        def plot_slider_update_infinity(val):
            val = 1e-10 if val == 0 else val
            if self.is_linkage:
                # joint positions, the other factorization is in reversed order
                _link = (
                    self.factorizations[0].direct_kinematics(val, inverted_part=True)
                    + self.factorizations[1].direct_kinematics(val, inverted_part=True)[
                        ::-1
                    ]
                )
                _ee_triangle = [
                    self.factorizations[0].direct_kinematics(val, inverted_part=True)[
                        -1
                    ]
                ] + [
                    self.factorizations[1].direct_kinematics(val, inverted_part=True)[
                        -1
                    ]
                ]
            else:
                _link = self.factorizations[0].direct_kinematics(
                    val, inverted_part=True
                )
                _ee_triangle = [_link[-1]]

            _x, _y, _z = zip(*[_link[j] for j in range(len(_link))])
            link_plot.set_data_3d(_x, _y, _z)

            _ee = self.factorizations[0].direct_kinematics_of_tool(
                val, self.end_effector.dq2point_homogeneous(), inverted_part=True
            )
            _ee_triangle.insert(1, _ee)
            _x, _y, _z = zip(*[_ee_triangle[j] for j in range(len(_ee_triangle))])
            end_effector_plot.set_data_3d(_x, _y, _z)

            fig.canvas.draw_idle()
            fig.canvas.update()
            fig.canvas.flush_events()

        self.t_slider.on_changed(plot_slider_update)
        self.t_inf_slider.on_changed(plot_slider_update_infinity)

        return ax
