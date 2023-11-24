# Collision analysis of single-loop linkages

[![build](https://git.uibk.ac.at/geometrie-vermessung/mechanisms-collisions/badges/main/pipeline.svg)](https://git.uibk.ac.at/geometrie-vermessung/mechanisms-collisions/-/jobs)
[![coverage](https://git.uibk.ac.at/geometrie-vermessung/mechanisms-collisions/badges/main/coverage.svg?job=run_tests)](https://git.uibk.ac.at/geometrie-vermessung/mechanisms-collisions/-/jobs)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)]()


## Intallation instuctions
1. Clone the repository
    (use preferably your client, or clone with the button on top of this page)

2. Install dependencies listed in [requirements.txt](requirements.txt), globally or in your virtual environment using for example pip with line:
    <code>pip install -r requirements.txt</code>

### Documentation
The documentation is generated as latex using Sphinx, output PDF can be found in [Artifacts](https://git.uibk.ac.at/geometrie-vermessung/mechanisms-collisions/-/artifacts) of the pipeline under job name <code>build_docs</code>, where you can download <code>artifacts.zip</code> that includes the PDF.

## Notes
At some point, this package will be divided in two modules:
* math-based backend (possibly joined with biquaterions package)
* mechanisms, collision analysis and plotting

