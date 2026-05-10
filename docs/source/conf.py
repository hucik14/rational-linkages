import os
import sys
import toml
from docutils.parsers.rst import Directive, directives


class DummyDirective(Directive):
    """To handle sphinx unknown directive"""
    has_content = True
    def run(self):
        return []

# register '.. clear-namespace::' directive from Sybil to Sphinx
directives.register_directive('clear-namespace', DummyDirective)

docs_source_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(docs_source_dir, '..', '..'))
src_dir = os.path.join(project_root, 'python', 'rational_linkages')
parent_dir = os.path.join(project_root, 'python')

sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

print(sys.path)

# -- Helper functions --------------------------------------------------------


def get_version():
    # Get the directory of this file
    this_dir = os.path.dirname(os.path.realpath(__file__))
    # Construct the path to pyproject.toml relative to this directory
    pyproject_path = os.path.join(this_dir, '..', '..', 'pyproject.toml')
    # Load the pyproject.toml file
    pyproject = toml.load(pyproject_path)
    # Extract the version
    version = pyproject['project']['version']
    return version

# -- Project information -----------------------------------------------------


project = 'Rational Linkages'
copyright = '2026, Daniel Huczala'
author = 'Daniel Huczala'

# The full version, including alpha/beta/rc tags
release = get_version()


# -- General configuration ---------------------------------------------------

autodoc_mock_imports = ["rational_linkages.utils_rust"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
]

suppress_warnings = [
    'autodoc.mocked_object',
]

# for 3D visualization
html_static_path = ['_static']
html_allow_raw_html = True

bibtex_bibfiles = ['refs.bib']

nitpicky = True
nitpick_ignore = [
    ('py:class', 'optional'),
    ('py:class', 'numpy.ndarray'),
    ('py:class', 'numpy.array'),
]

nitpick_ignore_regex = [
    (r'py:func', r'sympy\..*'),
    (r'py:class', r'sympy\..*'),
    (r'py:func', r'scipy\..*'),
    (r'py:class', r'scipy\..*'),
    (r'py:class', r'PyQt6\..*'),
    (r'py:class', r'QtCore\..*'),
    (r'py:class', r'Q\w+'),
    (r'py:data', r'typing\..*'),
]

# NumPy docstring style
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False  # handled by sphinx-autodoc-typehints
napoleon_use_ivar = True  # avoid duplicate attribute objects with autodoc properties

# Types go into parameter descriptions, not repeated in the signature
autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'sympy': ('https://docs.sympy.org/latest/', None),
    'biquaternion_py': ('https://biquaternion-py.readthedocs.io/en/latest/', None),
    'PyQt6': ('https://www.riverbankcomputing.com/static/Docs/PyQt6/', None),
    'pyqt6': ('https://doc.qt.io/qtforpython-6/', None),
    'pyqtgraph': ('https://pyqtgraph.readthedocs.io/en/latest/', None),
}


autodoc_type_aliases = {
    'CollisionFreeOptimization': 'rational_linkages.CollisionFreeOptimization.CollisionFreeOptimization',
    'DualQuaternion': 'rational_linkages.DualQuaternion.DualQuaternion',
    'ExudynAnalysis': 'rational_linkages.ExudynAnalysis.ExudynAnalysis',
    'LineSegment': 'rational_linkages.Linkage.LineSegment',
    'Linkage': 'rational_linkages.Linkage.Linkage',
    'PointsConnection': 'rational_linkages.Linkage.PointsConnection',
    'MotionDesigner': 'rational_linkages.MotionDesigner.MotionDesigner',
    'MotionFactorization': 'rational_linkages.MotionFactorization.MotionFactorization',
    'MotionInterpolation': 'rational_linkages.MotionInterpolation.MotionInterpolation',
    'NormalizedLine': 'rational_linkages.NormalizedLine.NormalizedLine',
    'NormalizedPlane': 'rational_linkages.NormalizedPlane.NormalizedPlane',
    'Plotter': 'rational_linkages.Plotter.Plotter',
    'PointHomogeneous': 'rational_linkages.PointHomogeneous.PointHomogeneous',
    'Quaternion': 'rational_linkages.Quaternion.Quaternion',
    'BezierSegment': 'rational_linkages.RationalBezier.BezierSegment',
    'RationalBezier': 'rational_linkages.RationalBezier.RationalBezier',
    'RationalCurve': 'rational_linkages.RationalCurve.RationalCurve',
    'RationalMechanism': 'rational_linkages.RationalMechanism.RationalMechanism',
    'TransfMatrix': 'rational_linkages.TransfMatrix.TransfMatrix',
}

html_theme = 'sphinx_rtd_theme'


### DOCTEST
# doctest_test_doctest_blocks = 'default'
# set bool skip_this_doctest = True here in order to skip the doctest (set by
# directive :skipif: in the testcode and testcleanup blocks)
# doctest_global_setup = """
# skip_this_doctest = False
# """
