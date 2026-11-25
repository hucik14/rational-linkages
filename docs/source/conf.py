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

# for 3D visualization
html_static_path = ['_static']
html_allow_raw_html = True

bibtex_bibfiles = ['refs.bib']

nitpicky = True
nitpick_ignore = [
    ('py:class', 'optional'),
    ('py:class', 'numpy.array'),
]

nitpick_ignore_regex = [
    (r'py:func', r'sympy\..*'),
    (r'py:class', r'sympy\..*'),
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

html_theme = 'sphinx_rtd_theme'

# -- SEO and Meta Configuration -----------------------------------------------
# HTML meta tags for search engines and social media
html_meta = {
    'description': 'Design collision-free 3D-printable mechanisms from motion specifications. Synthesis and analysis of spatial linkages for robotics and rapid prototyping.',
    'keywords': 'mechanism synthesis, linkage design, Bennett mechanism, 3D printing, robotics, kinematics, Python',
    'og:title': 'Rational Linkages - Mechanism Synthesis for 3D Printing',
    'og:description': 'Design and synthesize spatial and planar collision-free 3D-printable mechanisms programmatically. Perfect for robotics, mechanical design, and rapid prototyping.',
    'og:type': 'website',
}

# Canonical URL to avoid duplicate content issues
html_baseurl = 'https://rational-linkages.readthedocs.io/'

# Language
language = 'en'

# Structured data for search engines and LLMs
html_extra_path = ['_extra']  # create this directory
