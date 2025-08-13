from .PlotterPyqtgraph import InteractivePlotter, PlotterPyqtgraph
from .RationalMechanism import RationalMechanism


class Plotter:
    def __new__(cls,
               mechanism: RationalMechanism = None,
               base = None,
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
               parent_app=None):
        """
        Create and return an appropriate plotter instance based on parameters.

        Default backend is 'pyqtgraph', but can be changed to 'matplotlib'.
        """
        # delegate to the create method
        return cls.create(
            mechanism=mechanism,
            base=base,
            show_tool=show_tool,
            backend=backend,
            jupyter_notebook=jupyter_notebook,
            show_legend=show_legend,
            show_controls=show_controls,
            interval=interval,
            steps=steps,
            arrows_length=arrows_length,
            joint_sliders_lim=joint_sliders_lim,
            white_background=white_background,
            parent_app=parent_app
        )
    @classmethod
    def create(cls,
               mechanism: RationalMechanism = None,
               base = None,
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
               parent_app=None):
        """
        Create and return an appropriate plotter instance based on parameters.

        Default backend is 'pyqtgraph', but can be changed to 'matplotlib'.
        """
        interactive = mechanism is not None and not jupyter_notebook

        if backend == 'pyqtgraph' and not jupyter_notebook:
            if show_legend:
                print('Warning: The legend is supported only in Matplotlib backend. ')
            elif not show_controls:
                print(
                    'Warning: Hiding controls is supported only in Matplotlib backend.')

            if steps is None:
                steps = 2000

            if interactive:
                return InteractivePlotter(mechanism=mechanism,
                                          base=base,
                                          show_tool=show_tool,
                                          steps=steps,
                                          arrows_length=arrows_length,
                                          joint_sliders_lim=joint_sliders_lim,
                                          white_background=white_background)
            else:
                return PlotterPyqtgraph(parent_app=parent_app,
                                        base=base,
                                        interval=interval,
                                        steps=steps,
                                        arrows_length=arrows_length,
                                        white_background=white_background)
        else:
            if steps is None:
                steps = 200

            from .PlotterMatplotlib import PlotterMatplotlib
            plotter = PlotterMatplotlib(interactive=interactive,
                                        base=base,
                                        jupyter_notebook=jupyter_notebook,
                                        show_legend=show_legend,
                                        show_controls=show_controls,
                                        interval=interval,
                                        steps=steps,
                                        arrows_length=arrows_length,
                                        joint_sliders_lim=joint_sliders_lim)
            if mechanism is not None:
                plotter.plot(mechanism, show_tool=show_tool)

            return plotter