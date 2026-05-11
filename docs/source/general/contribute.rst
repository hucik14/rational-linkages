How to contribute
=================

Thanks for helping improve Rational Linkages.

Where to contribute
~~~~~~~~~~~~~~~~~~~

- Main repository: https://git.uibk.ac.at/geometrie-vermessung/rational-linkages
- Issue tracker (public): https://github.com/hucik14/rational-linkages/issues
- Questions and coordination: ``daniel.huczala@snu.ac.kr``


Project layout at a glance
~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository is organized as follows:

.. code-block:: text

   rational-linkages/
   |- python/rational_linkages/   # package source code (Python + Rust extension bindings)
   |- python/tests/               # pytest test suite
   |- rust/                       # Rust extension module (PyO3)
   |- docs/source/                # Sphinx docs (.rst, notebooks, examples)
   |- python/scripts/             # helper scripts (release notes, model regeneration)
   |- CHANGELOG.md                # project changelog (source of docs changelog page)
   `- pyproject.toml              # build metadata, dependencies, tool config


Set up a local development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package supports Python ``>=3.10``. Python ``3.11`` is recommended.

.. code-block:: bash

   git clone https://git.uibk.ac.at/geometrie-vermessung/rational-linkages.git
   cd rational-linkages
   pip install -e .[opt,dev,doc]

If you also work on CAD/export features, install the CAD extras as well:

.. code-block:: bash

   pip install -e .[opt,dev,doc,cad]

If you modify files in ``rust/``, install Rust toolchain and build the extension:

.. code-block:: bash

   cd rust
   cargo build --release

Add tests
~~~~~~~~~

For every new feature or significant change, add tests
to ensure correctness of the code. Use ``pytest`` and
place tests in the ``python/tests/`` directory,
following the existing structure.


Run tests and checks
~~~~~~~~~~~~~~~~~~~~

Before opening a contribution, run at least:

.. code-block:: bash

   pytest python/tests
   ruff check python

Notes:

- Tests are located in ``python/tests``.
- Code examples in ``docs/source/examples`` are executed by
  ``python/tests/test_docs_examples.py``.
- Python code blocks in package docstrings are collected via Sybil (see
  ``conftest.py``), so example snippets in docstrings should be runnable.
  Don't forget to put the code in a ``.. code-block:: python`` directive and
  close it with ``.. clear-namespace::`` if it defines variables that shouldn't
  leak into subsequent snippets.


Documentation workflow
~~~~~~~~~~~~~~~~~~~~~~

Documentation is built with Sphinx from ``docs/source``.

.. code-block:: bash

   cd docs
   make clean
   make html

Open ``docs/build/html/index.html`` in your browser to review the output.

To appear your changes in the changelog, add an entry at the end
of your commit message body in the format:

.. code-block:: text

    Describe your commit message here...

    changelog: <type>

Where ``<type>`` is one of ``added``, ``changed``, ``deprecated``, ``removed``, ``fixed``, or ``security``.
The description should be a concise summary of the change, suitable for end-users.

Docstrings and style expectations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add type hints to public functions/methods.
- Prefer NumPy-style docstrings (``Parameters``, ``Returns``, ``Raises``,
  ``Examples``), matching the Sphinx ``napoleon`` configuration.
- Include short runnable examples when useful.


References and citations in docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For bibliographic references in docs, add entries to
``docs/source/refs.bib`` and cite them in reST pages/docstrings.

Example BibTeX entry:

.. code-block:: text

   @article{Surname2024,
       author = {Author, Name},
       title = {Title of the Source},
       journal = {Journal Name},
       year = {2024},
       doi = {number},
       url = {weblink}
   }

You can then cite this with ``:footcite:t:`Surname2024```, don't
forget to add ``.. footbibliography::`` at the end of the page
or docstring to render the bibliography.
