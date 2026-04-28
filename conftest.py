from pathlib import Path
from sybil import Sybil
from sybil.parsers.rest import PythonCodeBlockParser, ClearNamespaceParser

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