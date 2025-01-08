.. _installation:

Installation
============

The recommended version of Python to run this package is version 3.11 (3.10 or higher
are supported), when it provides the smoothest plotting. It is also the version used
for development.

There are more ways to install the Python package. For regular use we recommend
installation from PyPI using **pip**, which will provide the latest released version.
For development, you can install from source by cloning the repository.

A) using pip:

.. code-block:: bash

    pip install rational-linkages

or

.. code-block:: bash

    pip install rational-linkages[opt]

for installing also optional dependencies (ipython - inline plotting from console,
gmpy2 - faster symbolic computations, exudyn - multibody simulations, numpy-stl -
work with meshes in exudyn).


B) from source as editable package:

.. code-block:: bash

    git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git

    cd rational-linkages

    pip install -e .[opt]

C) from source as editable package with development and documentation dependencies:

.. code-block:: bash

    git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git

    cd rational-linkages

    pip install -e .[opt,dev,doc]