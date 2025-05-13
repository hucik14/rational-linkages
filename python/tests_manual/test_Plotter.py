import unittest
import os
from unittest.mock import MagicMock
from matplotlib.testing.compare import compare_images

from rational_linkages.Plotter import Plotter
from rational_linkages.DualQuaternion import DualQuaternion
from rational_linkages.TransfMatrix import TransfMatrix
from rational_linkages.NormalizedLine import NormalizedLine
from rational_linkages.PointHomogeneous import PointHomogeneous
from rational_linkages.models import bennett_ark24


class TestPlotter(unittest.TestCase):

    def setUp(self):
        self.plt = Plotter(backend='matplotlib')

    def test_plot_line(self):
        line = NormalizedLine()
        self.plt.plotter._plot_line = MagicMock()
        self.plt.plot(line)
        self.plt.plotter._plot_line.assert_called_once_with(line)

    def test_plot_point(self):
        point = PointHomogeneous()
        self.plt.plotter._plot_point = MagicMock()
        self.plt.plot(point)
        self.plt.plotter._plot_point.assert_called_once_with(point)

    def test_plot_dual_quaternion(self):
        dq = DualQuaternion()
        self.plt.plotter._plot_dual_quaternion = MagicMock()
        self.plt.plot(dq)
        self.plt.plotter._plot_dual_quaternion.assert_called_once_with(dq)

    def test_plot_transf_matrix(self):
        matrix = TransfMatrix()
        self.plt.plotter._plot_transf_matrix = MagicMock()
        self.plt.plot(matrix)
        self.plt.plotter._plot_transf_matrix.assert_called_once_with(matrix)

    def test_plot_rational_curve(self):
        mechanism = bennett_ark24()
        curve = mechanism.get_motion_curve()
        self.plt.plotter._plot_rational_curve = MagicMock()
        self.plt.plot(curve)
        self.plt.plotter._plot_rational_curve.assert_called_once_with(curve)

    def test_plot_interactive(self):
        mechanism = bennett_ark24()
        plot = Plotter(mechanism=mechanism, backend='matplotlib')
        t = 0.5
        plot.plotter.plot_slider_update(t, t_param=t)

    def test_plot_line_image(self):
        line = NormalizedLine()
        self.plt.plot(line)
        self.plt.plotter.update_limits(self.plt.plotter.ax)
        self.plt.plotter.fig.savefig('test_plot_line.png')
        self.assertIsNone(compare_images('plot_line_baseline.png',
                                       'test_plot_line.png',
                                       0.01))
        os.remove('test_plot_line.png')

    def test_plot_list(self):
        dq1 = DualQuaternion()
        dq2 = TransfMatrix()
        self.plt.plotter._plot_dual_quaternion = MagicMock()
        self.plt.plotter._plot_transf_matrix = MagicMock()
        self.plt.plotter.plot([dq1, dq2])
        self.plt.plotter._plot_dual_quaternion.assert_called_once_with(dq1)
        self.plt.plotter._plot_transf_matrix.assert_called_once_with(dq2)


if __name__ == '__main__':
    unittest.main()
