from .PlotterPyqtgraph import InteractivePlotter, PlotterPyqtgraph
from .RationalMechanism import RationalMechanism


class Plotter:
    """Create and return an appropriate plotter instance based on parameters.

    Default backend is ``'pyqtgraph'`` but can be changed to
    ``'matplotlib'``.

    Parameters
    ----------
    mechanism
        Optional :class:`RationalMechanism` instance used by an interactive
        plotter. If provided and ``jupyter_notebook`` is False an
        interactive widget is created.
    base
        Optional base transform or object passed to backends.
    show_tool
        Whether to draw the end-effector/tool in the plots.
    backend
        Backend name to use (``'pyqtgraph'`` or ``'matplotlib'``).
    jupyter_notebook
        If True, create plotters suited for Jupyter notebooks.
    show_legend
        Whether to display a legend (only supported by Matplotlib
        backend).
    show_controls
        Whether to show interactive controls (only supported by
        Matplotlib backend when running non-interactively).
    paper_visual
        If True, use a visualization style suitable for paper figures.
        (Matplotlib backend only).
    ticks_step
        Tick step size (Matplotlib backend only).
    interval
        Parameter sampling interval passed to some plotter backends.
    steps
        Number of sampling steps used when plotting curves.
    arrows_length
        Visual length of pose arrows in the plot.
    joint_sliders_lim
        Limit for joint sliders in interactive widgets.
    white_background
        If True, use a white background for plotting.
    parent_app
        Optional parent application instance for GUI backends.

    Returns
    -------
    object
        An instance of a backend-specific plotter class.

    Notes
    -----
    The returned backend instance exposes ``.plot(obj, **kwargs)`` and accepts
    additional runtime plotting keyword arguments. The most used kwargs are:

    Common kwargs across backends
        ``label`` : str (or list of str when plotting a list)
            Label for plotted object(s). For a list of objects, pass a list of
            labels of matching length.
        ``interval`` : tuple or ``'closed'``
            Parameter interval used for :class:`RationalCurve` and
            :class:`RationalBezier`. If set to ``'closed'``, the curve will
            be plotted using tangent function in full scale, including t -> inf.
        ``with_poses`` : bool
            Show poses for :class:`RationalCurve`.
        ``t`` : float
            Mechanism/factorization configuration parameter.

    PyQtGraph-oriented kwargs
        ``color`` : str or RGBA tuple/list
            Color name (``white``, ``red``, ``green``, ``blue``, ``yellow``,
            ``magenta``, ``cyan``, ``orange``, ``lime``) or RGBA tuple.
        ``size`` : float
            Marker size for point plotting.

    Matplotlib-oriented kwargs
        Standard Matplotlib style kwargs are forwarded to backend primitives
        (``ax.plot``, ``ax.scatter``, ``ax.quiver``, ``ax.plot_surface``), e.g.
        ``color``, ``linewidth``/``lw``, ``linestyle``, ``marker``, ``alpha``.

    Custom Matplotlib artists
        When ``backend='matplotlib'``, the returned object exposes
        ``plotter.ax`` and ``plotter.fig``. You can add your own artists
        directly, for example: ``plotter.ax.scatter(x, y, z, c='k', s=20)``.
        This is backend-specific and is not available in the PyQtGraph backend.

    Examples
        ``plotter.plot(curve, color='orange', interval=(0, 1), with_poses=True)``

        ``plotter.plot(mechanism, t=0.3, show_tool=True)``

        ``plotter = Plotter(backend='matplotlib'); plotter.ax.scatter([0], [0], [0])``
    """
    def __new__(cls,
                mechanism: RationalMechanism = None,
                base = None,
                show_tool: bool = True,
                backend: str = 'pyqtgraph',
                jupyter_notebook: bool = False,
                show_legend: bool = False,
                show_controls: bool = True,
                paper_visual: bool = False,
                ticks_step: float = None,
                interval: tuple = (-1, 1),
                steps: int = None,
                arrows_length: float = 1.0,
                joint_sliders_lim: float = 1.0,
                white_background: bool = False,
                parent_app=None):
        # delegate to the create method
        return cls.create(
            mechanism=mechanism,
            base=base,
            show_tool=show_tool,
            backend=backend,
            jupyter_notebook=jupyter_notebook,
            show_legend=show_legend,
            show_controls=show_controls,
            paper_visual=paper_visual,
            ticks_step=ticks_step,
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
               paper_visual: bool = False,
               ticks_step: float = None,
               interval: tuple = (-1, 1),
               steps: int = None,
               arrows_length: float = 1.0,
               joint_sliders_lim: float = 1.0,
               white_background: bool = False,
               parent_app=None):
        """Create and return an appropriate plotter instance based on parameters.

        Returns
        -------
        object
            Backend-specific plotter instance.
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
                                        paper_visual=paper_visual,
                                        ticks_step=ticks_step,
                                        interval=interval,
                                        steps=steps,
                                        arrows_length=arrows_length,
                                        joint_sliders_lim=joint_sliders_lim)
            if mechanism is not None:
                plotter.plot(mechanism, show_tool=show_tool)

            return plotter