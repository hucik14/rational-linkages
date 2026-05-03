import runpy
from pathlib import Path
import pytest

root = Path(__file__).resolve().parent.parent.parent
examples = list((root / "docs" / "source" / "examples").glob("*.py"))

@pytest.mark.parametrize("script", examples, ids=lambda p: p.name)
def test_example_script(script):
    runpy.run_path(str(script))