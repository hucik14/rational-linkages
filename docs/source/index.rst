.. Rational Linkages documentation master file, created by
   sphinx-quickstart on Fri Sep  8 13:45:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Rational Linkages Toolbox Documentation
=======================================
.. image:: https://img.shields.io/gitlab/v/release/21918?gitlab_url=https%3A%2F%2Fgit.uibk.ac.at%2F&style=social&logo=gitlab&label=repository
   :target: https://git.uibk.ac.at/geometrie-vermessung/rational-linkages
   :alt: GitLab (self-managed)

.. image:: https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/badges/main/pipeline.svg?job=run_tests
   :target: https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/jobs
   :alt: Build Status

.. image:: https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/badges/main/coverage.svg?job=run_tests
   :target: https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/jobs
   :alt: Coverage Status

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
   :target: https://www.python.org/downloads/release/python-3117/
   :alt: Python 3.11

Welcome to the Rational Linkages Toolbox Documentation, which serves as a reference for
the toolbox. The toolbox is a collection of functions for the analysis of rational
linkages and their rapid prototyping. It is written in Python and uses the Numpy and
SymPy libraries for computations, and Matplotlib for plotting. The toolbox is developed
at the Unit of Geometry and Surveying, University of Innsbruck, Austria.

The source code is available as `Gitlab repository`_ hosted by UIBK. The *installation
instructions* can be found in the `installation manual`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   general
   modules
   tutorials
   background-math

.. _Gitlab repository: https://git.uibk.ac.at/geometrie-vermessung/rational-linkages
.. _installation manual: general/installation.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
