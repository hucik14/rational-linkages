# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import toml


# Set an environment variable for skipping doctest
os.environ['SKIP_DOCTEST'] = 'True'

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
copyright = '2024, Daniel Huczala'
author = 'Daniel Huczala'

# The full version, including alpha/beta/rc tags
release = get_version()


# -- General configuration ---------------------------------------------------

autodoc_mock_imports = ["rational_linkages.utils_rust"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'nbsphinx',
              'sphinx.ext.doctest',
              'sphinxcontrib.bibtex',
              'hoverxref.extension',
              ]

bibtex_bibfiles = ['refs.bib']

nitpicky = True
nitpick_ignore = [
    ('py:class', 'np.ndarray'),
    ('py:class', 'numpy.ndarray'),
    ('py:class', 'numpy.array'),
    ('py:class', 'np.array'),
    ('py:class', 'np.polynomial.Polynomial'),
    ('py:class', 'sp.Symbol'),
    ('py:class', 'sp.Poly'),
    ('py:class', 'sp.Matrix'),
    ('py:class', 'bq.Poly'),
    ('py:class', 'gl.GLViewWidget'),
    ('py:class', 'PyQt5.QtWidgets.QWidget'),
]

intersphinx_mapping = {'python': ('http://docs.python.org/3', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('http://matplotlib.org/stable', None),
                       'sympy': ('https://docs.sympy.org/latest/', None),
                       'biquaternion_py': ('https://biquaternion-py.readthedocs.io/en/latest/', None),
                       'PyQt6': ('https://www.riverbankcomputing.com/static/Docs/PyQt6/', None),
                       'pyqt6': ('https://doc.qt.io/qtforpython-6/', None),
                       'pyqtgraph': ('https://pyqtgraph.readthedocs.io/en/latest/', None),
                       }

hoverxref_auto_ref = True
hoverxref_domains = [
    'py',  # Python domain
    'std', # Standard domain
    'cite', # Citation domain

]
hoverxref_role_types = {
    'hoverxref': 'tooltip',
    'ref': 'tooltip',
    'numref': 'tooltip',
    'confval': 'tooltip',
    'term': 'tooltip',
    'mod': 'tooltip',
    'class': 'tooltip',
    'meth': 'tooltip',
    'func': 'tooltip',
    'attr': 'tooltip',
    'cite': 'tooltip',
    'footcite': 'tooltip',
}

hoverxref_tooltip_maxwidth = 300  # Set the maximum width of tooltips
hoverxref_tooltip_border_color = 'blue'  # Customize the border color
hoverxref_tooltip_border_width = 1  # Set the border width

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

doctest_test_doctest_blocks = 'default'

# set bool skip_this_doctest = True here in order to skip the doctest (set by
# directive :skipif: in the testcode and testcleanup blocks)
doctest_global_setup = """
skip_this_doctest = False
"""
