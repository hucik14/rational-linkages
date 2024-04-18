.. Rational Linkages documentation master file, created by
   sphinx-quickstart on Fri Sep  8 13:45:57 2023.

.. include:: refs-weblinks.rst

Rational Linkages Documentation
===============================

.. image:: https://img.shields.io/gitlab/v/release/21918?gitlab_url=https%3A%2F%2Fgit.uibk.ac.at%2F&style=social&logo=gitlab&label=repository
   :target: https://git.uibk.ac.at/geometrie-vermessung/rational-linkages
   :alt: GitLab (self-managed)

.. image:: https://img.shields.io/pypi/v/rational-linkages.svg
   :target: https://pypi.org/project/rational-linkages/
   :alt: PyPI

.. image:: https://img.shields.io/github/issues/hucik14/rl-issues
   :target: https://github.com/hucik14/rl-issues/issues
   :alt: GitHub issues

.. image:: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg
   :target: https://www.python.org/downloads/release/python-3117/
   :alt: Python

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/hucik14/rl-issues/HEAD?labpath=jupyter_ntbs%2Fsynthesis_bennett.ipynb

Welcome to the Rational Linkages package documentation, which serves as a reference for
the provided package. This toolbox is a collection of functions for the analysis of
rational linkages and their rapid prototyping. It is written in Python and uses the
Numpy and SymPy libraries for computations, and Matplotlib for plotting. It is developed
at the Unit of Geometry and Surveying, University of Innsbruck, Austria.

The source code is available as `Gitlab repository`_ hosted by UIBK. The *installation
instructions* can be found in the :ref:`installation manual<installation>`.

Since the self-hosted repository does not allow external users to create issues,
please, use the `external issue tracker`_
hosted on Github for submitting **issues** and **feature requests**.

Run live examples of the package in your browser using the **Binder** service. Click
on the badge below to start the Binder session:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/hucik14/rl-issues/HEAD?labpath=jupyter_ntbs%2Fsynthesis_bennett.ipynb

In case of other questions or contributions, please, email the author at:
daniel.huczala@uibk.ac.at

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   general
   modules
   tutorials
   background-math


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Authors
=======

The package is developed by the Unit of Geometry and Surveying, University of Innsbruck,
Austria. Namely, the following people contributed to the development and writing
documentation:

- Daniel Huczala (author, maintainer)
- Daren A. Thimm (contributor)
- Martin Pfurner (contributor)
- Hans-Peter Schröcker (supervisor, contributor)

Citing the Package
==================

If you use the Rational Linkages package in your research, please, cite it as follows:

Huczala, D., Siegele J., Thimm, D., Pfurner, M., Schröcker, H.-P. (2024).
Rational Linkages: From Poses to 3D-printed Prototypes.
Advances in Robot Kinematics 2024. ARK 2024.
Preprint **arXiv:2403.00558**, url: `https://arxiv.org/abs/2403.00558 <https://arxiv.org/abs/2403.00558>`_.

.. code-block:: text

   @inbook{huczala2024linkages,
      title={Rational Linkages: From Poses to 3D-printed Prototypes},
      author={Daniel Huczala and Johannes Siegele and Daren A. Thimm and Martin Pfurner and Hans-Peter Schröcker},
      year={2024},
      booktitle="Advances in Robot Kinematics 2024, ARK 2024",
      url={https://arxiv.org/abs/2403.00558}
   }

Acknowledgements
================

.. figure:: figures/eu.png
   :align: left
   :alt: EU Flag
   :width: 250px

:sub:`Funded by the European Union. Views and opinions expressed are however those of
the author(s) only and do not necessarily reflect those of the European Union
or the European Research Executive Agency (REA). Neither the European Union
nor the granting authority can be held responsible for them.`

We would like to thank our colleagues outside the Unit of Geometry and Surveying, who
contributed to the development of the package and helped with implementation of
their algorithms and suggestions. These people are namely:

- **Johannes Gerstmayr**, University of Innsbruck, Austria, for the help with creating the interface to his `Exudyn`_ :footcite:p:`Gerstmayr2023` software. More on this in section :ref:`exudyn_info`.
- **Georg Nawratil**, Technical University of Vienna, Austria,
- and **Zijia Li**, Chinese Academy of Sciences, China, for their help with the implementation of the Combinatorial Search Algorithm of collision-free linkages :footcite:p:`Li2020`. More on this in section :ref:`combinatorial_search`.
- **Johannes Siegele**, Austrian Academy of Sciences, Austria, for his help with the implementation of the algorithm for motion interpolation of 4 poses. More on this in section :ref:`interpolation_background`.

**References**

.. footbibliography::