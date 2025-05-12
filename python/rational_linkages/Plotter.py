from .RationalMechanism import RationalMechanism
from .PlotterPyqtgraph import PlotterPyqtgraph, InteractivePlotter


class Plotter:
    def __init__(self,
                 mechanism: RationalMechanism = None,
                 show_tool: bool = True,
                 backend: str = 'pyqtgraph',
                 jupyter_notebook: bool = False,
                 show_legend: bool = False,
                 show_controls: bool = True,
                 interval: tuple = (-1, 1),
                 steps: int = None,
                 arrows_length: float = 1.0,
                 joint_sliders_lim: float = 1.0,
                 white_background: bool = False,
                 parent_app=None, ):
        """
        Initialize the plotter super class.

        Default backend is 'pyqtgraph', but can be changed to 'matplotlib'.
        """

        if mechanism is not None and not jupyter_notebook:
            interactive = True
        else:
            interactive = False

        if backend == 'pyqtgraph' and not jupyter_notebook:
            if show_legend:
                print('Warning: The legend is supported only in Matplotlib backend. ')
            elif show_controls:
                print('Warning: Hiding controls is supported only in Matplotlib backend.')

            if steps is None:
                steps = 2000

            if interactive:
                self.plotter = InteractivePlotter(mechanism=mechanism,
                                                  show_tool=show_tool,
                                                  steps=steps,
                                                  arrows_length=arrows_length,
                                                  joint_sliders_lim=joint_sliders_lim,
                                                  white_background=white_background)
            else:
                self.plotter = PlotterPyqtgraph(parent_app=parent_app,
                                                interval=interval,
                                                steps=steps,
                                                arrows_length=arrows_length,
                                                white_background=white_background)

        else:
            if steps is None:
                steps = 200

            from .PlotterMatplotlib import PlotterMatplotlib
            self.plotter = PlotterMatplotlib(interactive=interactive,
                                             jupyter_notebook=jupyter_notebook,
                                             show_legend=show_legend,
                                             show_controls=show_controls,
                                             interval=interval,
                                             steps=steps,
                                             arrows_length=arrows_length,
                                             joint_sliders_lim=joint_sliders_lim)
            if mechanism is not None:
                self.plotter.plot(mechanism, show_tool=show_tool)

    def plot(self, *args, **kwargs):
        """
        Plot the given arguments.

        :param args: The arguments to plot.
        :param kwargs: The keyword arguments to plot.
        """
        self.plotter.plot(*args, **kwargs)

    def show(self):
        """
        Show the plot.
        """
        self.plotter.show()


