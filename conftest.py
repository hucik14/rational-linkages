from pathlib import Path
from sybil import Sybil
from sybil.parsers.rest import PythonCodeBlockParser, ClearNamespaceParser

pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(),
        ClearNamespaceParser(),
    ],
    path=Path(__file__).resolve().parent / "python" / "rational_linkages",
    patterns=["*.py"],
).pytest()