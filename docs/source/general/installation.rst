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

    pip install rational-linkages[opt,exu]

Mac users might need to add backslashes to escape the brackets, e.g.:

.. code-block:: bash

    pip install rational-linkages\[opt,exu\]

for installing also **opt**ional dependencies (ipython - inline plotting, gmpy2 - faster
symbolic computations, matplotlib - 3D plotting) and **exu**dyn dependencies (exudyn - multibody
simulations, numpy-stl + ngsolve - work with meshes in exudyn).

On **Linux** systems, to run GUI interactive plotting, some additional
libraries are required for plotting with PyQt6. Using
Ubuntu, it can be installed as follows:

.. code-block:: bash

    sudo apt install libgl1-mesa-glx libxkbcommon-x11-0 libegl1 libdbus-1-3

or on Ubuntu 24.04 and higher:

.. code-block:: bash

    sudo apt install libgl1 libxkbcommon-x11-0 libegl1 libdbus-1-3

B) from source as editable package:

.. code-block:: bash

    git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git

    cd rational-linkages

    pip install -e .[opt]

Mac users might need to add backslashes to escape the brackets, e.g.:

.. code-block:: bash

    pip install -e .\[opt\]

C) from source as editable package with development and documentation dependencies:

.. code-block:: bash

    git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git

    cd rational-linkages

    pip install -e .[opt,dev,doc]
