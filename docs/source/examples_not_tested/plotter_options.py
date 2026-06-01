"""Manual plotting options exerciser for Plotter backends.

This script creates representative geometry and mechanism objects, then calls
`Plotter(...).plot(..., **kwargs)` with the options documented in `Plotter`.
Use it for manual GUI verification and quick smoke checks.
"""

from __future__ import annotations

import argparse
import traceback

from rational_linkages import NormalizedLine, Plotter, PointHomogeneous
from rational_linkages.models import bennett_ark24


def build_demo_objects():
    """Create a small set of objects used by both backends."""
    mechanism = bennett_ark24()
    curve = mechanism.curve()

    p0 = PointHomogeneous([1.0, 0.0, 0.0, 0.0])
    p1 = PointHomogeneous([1.0, 0.35, 0.15, 0.25])
    p2 = PointHomogeneous([1.0, -0.15, 0.30, 0.10])

    axis_line = NormalizedLine.from_two_points(p0, p1)
    return mechanism, curve, p0, p1, p2, axis_line


def exercise_matplotlib(show: bool):
    """Exercise documented factory args and plot kwargs on matplotlib backend."""
    mechanism, curve, p0, p1, p2, axis_line = build_demo_objects()

    # Factory options from Plotter.__new__/create docs.
    plotter = Plotter(
        backend="matplotlib",
        mechanism=None,
        base=None,
        jupyter_notebook=True,
        show_legend=True,
        show_controls=True,
        paper_visual=False,
        ticks_step=0.2,
        interval=(-1, 1),
        steps=250,
        arrows_length=0.12,
        joint_sliders_lim=1.2,
        show_tool=True,
        white_background=False,
    )

    # Common kwargs.
    plotter.plot(curve, label="curve tuple interval", interval=(0, 1), with_poses=True, color="tab:orange", lw=2)
    plotter.plot(curve, label="curve closed interval", interval="closed", color="tab:blue", linestyle="--")

    # Label list handling.
    plotter.plot([p0, p1, p2], label=["p0", "p1", "p2"], color="crimson", marker="o")

    # Line + interval kwargs.
    plotter.plot(axis_line, label="axis", interval=(-0.75, 0.75), color="purple", linestyle="-.")

    # Direct matplotlib customization via exposed axes.
    plotter.ax.scatter([1.15], [0.15], [0.30], c="k", s=40, marker="x")
    plotter.ax.plot([0, 0.2], [0, 0.1], [0, 0.25], color="black", lw=1, linestyle="--")

    if show:
        plotter.show()


def exercise_pyqtgraph(show: bool):
    """Exercise documented factory args and plot kwargs on pyqtgraph backend."""
    mechanism, curve, p0, p1, p2, axis_line = build_demo_objects()

    # Factory options from Plotter.__new__/create docs.
    plotter = Plotter(
        backend="pyqtgraph",
        mechanism=None,
        base=None,
        interval=(-1, 1),
        steps=500,
        arrows_length=0.15,
        white_background=True,
        show_tool=True,
    )

    # Common kwargs.
    plotter.plot(curve, interval=(0, 1), with_poses=True, color="orange", label="curve with poses")
    plotter.plot(curve, interval="closed", color="yellow", label="closed curve")

    # PyQtGraph-specific kwargs.
    plotter.plot(p0, label="p0", color="red", size=9)
    plotter.plot(p1, label="p1", color=(0.2, 0.9, 0.2, 1.0), size=11)
    plotter.plot(axis_line, label="axis", interval=(-0.6, 0.6), color="magenta")

    # Label list handling.
    plotter.plot([p0, p1, p2], label=["list p0", "list p1", "list p2"], color="lime", size=7)

    # Additional public helper methods with color/label kwargs.
    plotter.plot_axis_between_two_points(p0, p1, color="blue", label="axis p0->p1")
    plotter.plot_line_segments_between_points([p0, p1, p2], color="green")

    if show:
        plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Manual Plotter option exerciser")
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "pyqtgraph", "both"],
        default="both",
        help="Which backend(s) to run.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open windows; run plotting calls only.",
    )
    args = parser.parse_args()

    run_matplotlib = args.backend in {"matplotlib", "both"}
    run_pyqtgraph = args.backend in {"pyqtgraph", "both"}
    show = not args.no_show

    failures = []

    if run_matplotlib:
        try:
            print("[manual_plotter_options] Running matplotlib scenario...")
            exercise_matplotlib(show=show)
            print("[manual_plotter_options] matplotlib scenario completed.")
        except Exception as exc:  # pragma: no cover - manual script
            failures.append(("matplotlib", exc, traceback.format_exc()))

    if run_pyqtgraph:
        try:
            print("[manual_plotter_options] Running pyqtgraph scenario...")
            exercise_pyqtgraph(show=show)
            print("[manual_plotter_options] pyqtgraph scenario completed.")
        except Exception as exc:  # pragma: no cover - manual script
            failures.append(("pyqtgraph", exc, traceback.format_exc()))

    if failures:
        print("\n[manual_plotter_options] Failures:")
        for backend, exc, tb in failures:
            print(f"--- {backend}: {exc}")
            print(tb)
        raise SystemExit(1)

    print("\n[manual_plotter_options] All selected scenarios completed successfully.")


if __name__ == "__main__":
    main()


