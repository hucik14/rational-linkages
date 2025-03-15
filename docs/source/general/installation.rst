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

Mac or Linux users might need to add backslashes to escape the brackets, e.g.:

.. code-block:: bash

    pip install rational-linkages\[opt\]

for installing also optional dependencies (ipython - inline plotting from console,
gmpy2 - faster symbolic computations, exudyn - multibody simulations, numpy-stl -
work with meshes in exudyn).

B) from source as editable package:

.. code-block:: bash

    git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git

    cd rational-linkages

    pip install -e .[opt]

Mac or Linux users might need to add backslashes to escape the brackets, e.g.:

.. code-block:: bash

    pip install -e .\[opt\]

C) from source as editable package with development and documentation dependencies:

.. code-block:: bash

    git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git

    cd rational-linkages

    pip install -e .[opt,dev,doc]

Additionally, on linux systems, some additional libraries are required for plotting with PyQt6. Using
Ubuntu, you can install them as follows:

.. code-block:: bash

    sudo apt install libgl1-mesa-glx libxkbcommon-x11-0 libegl1 libdbus-1-3