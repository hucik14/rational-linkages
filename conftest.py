from pathlib import Path
from sybil import Sybil
from sybil.parsers.rest import PythonCodeBlockParser, ClearNamespaceParser

# turn off plotting >>>>>
import matplotlib
matplotlib.use("Agg")

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# conftest.py
from unittest.mock import patch
import pytest

@pytest.fixture(autouse=True)
def mock_plotter_show():
    with patch("rational_linkages.PlotterPyqtgraph.PlotterPyqtgraph.show"), \
         patch("PyQt6.QtWidgets.QApplication.exec"):
        yield
# <<<<<

base = Path(__file__).resolve().parent

python_files = Sybil(
    parsers=[
        PythonCodeBlockParser(),
        ClearNamespaceParser(),
    ],
    path=base / "python" / "rational_linkages",
    patterns=["*.py"],
)

rst_files = Sybil(
    parsers=[
        PythonCodeBlockParser(),
        ClearNamespaceParser(),
    ],
    path=base / "docs" / "source",
    patterns=["*.rst"],
)

pytest_collect_file = (python_files + rst_files).pytest()