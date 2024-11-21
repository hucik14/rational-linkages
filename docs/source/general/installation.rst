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


B) from source as editable package:

.. code-block:: bash

    git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git

    cd rational-linkages

    pip install -e .

C) from source as editable package with development and documentation dependencies:

.. code-block:: bash

    git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git

    cd rational-linkages

    pip install -e .[dev,doc]