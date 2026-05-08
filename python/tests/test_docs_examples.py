import runpy
from pathlib import Path
import pytest
from unittest.mock import patch

# Set backend BEFORE any other imports, and via env var so child imports respect it
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent.parent
examples = list((root / "docs" / "source" / "examples").glob("*.py"))

@pytest.fixture(autouse=True)
def mock_plotter_show():
    with patch("rational_linkages.PlotterPyqtgraph.PlotterPyqtgraph.show"), \
         patch("PyQt6.QtWidgets.QApplication.exec"), \
         patch("matplotlib.pyplot.show"):  # <-- patch plt.show directly
        yield

@pytest.mark.parametrize("script", examples, ids=lambda p: p.name)
def test_example_script(script):
    runpy.run_path(str(script))