import subprocess
import sys
from pathlib import Path
import pytest


root = Path(__file__).resolve().parent.parent.parent
examples = sorted((root / "docs" / "source" / "examples").glob("*.py"))

HEADLESS = True  # set to False to see plots locally

_RUNNER_HEADLESS = """
import os
import runpy
import sys
from unittest.mock import patch

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

with patch("rational_linkages.PlotterPyqtgraph.PlotterPyqtgraph.show"), \
     patch("PyQt6.QtWidgets.QApplication.exec"), \
     patch("matplotlib.use"), \
     patch("matplotlib.pyplot.show"):
    runpy.run_path(sys.argv[1], run_name="__main__")
"""

_RUNNER_VISUAL = """
import runpy
import sys

runpy.run_path(sys.argv[1], run_name="__main__")
"""


@pytest.mark.parametrize("script", examples, ids=lambda p: p.name)
def test_example_script(script):
    runner = _RUNNER_HEADLESS if HEADLESS else _RUNNER_VISUAL

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner, str(script)],
            cwd=str(root),
            capture_output=HEADLESS,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"Example timed out after 60s: {script.name}")

    if proc.returncode == 0:
        return

    output = (f"{proc.stdout}\n{proc.stderr}".strip()) if HEADLESS else ""
    pytest.fail(f"Example failed: {script.name}\n{output}")