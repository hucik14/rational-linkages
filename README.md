[![build](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/badges/main/pipeline.svg)](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/jobs)
[![Documentation Status](https://readthedocs.org/projects/rational-linkages/badge/?version=latest)](https://rational-linkages.readthedocs.io/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/rational-linkages.svg)](https://pypi.org/project/rational-linkages/)
[![maintenance-status](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/network/main)
[![coverage](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/badges/main/coverage.svg?job=test_coverage)](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/jobs)
[![GitHub issues](https://img.shields.io/github/issues/hucik14/rl-issues)](https://github.com/hucik14/rl-issues/issues)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hucik14/rl-issues/HEAD?labpath=jupyter_ntbs%2Fsynthesis_bennett.ipynb)


# Rational Linkages

This Python-based package provides a collection of methods for the synthesis, 
analysis, rapid prototyping 
of the single-loop rational linkages, allowing one to create 3D-printable 
collision-free mechanisms synthesised for a given task (set of poses). 

<img src="/docs/source/figures/r4.JPEG" width="24%">
<img src="/docs/source/figures/r6li.JPEG" width="24%">
<img src="/docs/source/figures/r6hp.JPEG" width="24%">
<img src="/docs/source/figures/r6joh.JPEG" width="24%">

The package is developed as a part of the research project at the 
Unit of Geometry and Surveying, University of Innsbruck, Austria. 

## Documentation, tutorials, issues

[Rational Linkages Documentation](https://rational-linkages.readthedocs.io/) is 
hosted on Read the Docs, and provides a comprehensive overview of the package with 
[examples and tutorials](https://rational-linkages.readthedocs.io/latest/general/overview.html).

Since the self-hosted repository does not allow external users to create issues,
please, use the [external issue tracker](https://github.com/hucik14/rl-issues) 
hosted on GitHub for submitting **issues** and **feature requests**. Additionally,
you can *"watch/star"* the issue tracker package **to get notified about the updates**
(new releases will be also announced there).

You can test live-example of Jupyter notebook using Binder, by clicking on the 
following badge:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hucik14/rl-issues/HEAD?labpath=jupyter_ntbs%2Fsynthesis_bennett.ipynb)

In case of other questions or contributions, please, email the author at:
`daniel.huczala@uibk.ac.at`

STL files of some mechanisms may be found as
[models on Printables.com](https://www.printables.com/@hucik14_497869/collections/443601).

## Intallation instuctions

Recommended Python version is **3.11**, when it provides the smoothest plotting 
(but 3.10 or higher are supported). Python 3.11 is also the version used for 
development.

### Install from PyPI

Using pip:

<code>pip install rational-linkages</code>

### Install from source

1. Clone the repository (use preferably your client, or clone with the button on top of this page or using the following line)
    
    <code>git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git </code>

2. Navigate to the repository folder

    <code>cd rational-linkages</code>

3. Install the *editable* version of the package using pip:

    <code>pip install -e .</code>

## Preprint and citing the package

For additional information, see our preprint paper, and in the case of usage, please, 
cite it:

Huczala, D., Siegele, J., Thimm, D.A., Pfurner, M., Schröcker, HP. (2024). 
Rational Linkages: From Poses to 3D-Printed Prototypes. 
In: Lenarčič, J., Husty, M. (eds) Advances in Robot Kinematics 2024. ARK 2024. 
Springer Proceedings in Advanced Robotics, vol 31. Springer, Cham. 
DOI: [10.1007/978-3-031-64057-5_27](https://doi.org/10.1007/978-3-031-64057-5_27).

```bibtex
@inproceedings{huczala2024linkages,
    title={Rational Linkages: From Poses to 3D-printed Prototypes},
    author={Daniel Huczala and Johannes Siegele and Daren A. Thimm and Martin Pfurner and Hans-Peter Schröcker},
    year={2024},
    booktitle = {Advances in Robot Kinematics 2024. ARK 2024},
    publisher = {Springer International Publishing},
    url={https://doi.org/10.1007/978-3-031-64057-5_27},
    doi = {10.1007/978-3-031-64057-5_27},
}
```

Preprint **arXiv:2403.00558**: [https://arxiv.org/abs/2403.00558](https://arxiv.org/abs/2403.00558).

## Acknowledgements

Funded by the European Union. Views and opinions expressed are however those of 
the author(s) only and do not necessarily reflect those of the European Union 
or the European Research Executive Agency (REA). Neither the European Union 
nor the granting authority can be held responsible for them.

<img src="./docs/source/figures/eu.png" width="250" />