Installation
=============

Requirements
-------------

- **OS**: Linux or macOS (Windows is not currently supported).
- **Python**: 3.10 or higher (required by the ``jax`` dependency).
- Key runtime dependencies: ``geopandas``, ``pandas``, ``numpy``, ``scipy``,
  ``jax``/``jaxlib``, ``statsmodels``, ``matplotlib``, and the mesh/geospatial
  stack (``shapely``, ``gmsh``, ``meshio``, ``pygmsh``, ``pyproj``).

See :file:`pyproject.toml` or :file:`environment.yml` in the repository for
the complete, versioned dependency list.

Option 1: pip
--------------

.. code-block:: bash

   pip install geossm

Option 2: from source with Conda (recommended for development)
-----------------------------------------------------------------

Several of geossm's dependencies (``gmsh``, ``mfem``, ``jax``) are easiest to
install consistently through Conda, so this is the recommended route if you
plan to build meshes, run the low-rank models, or work on the package itself.

#. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/jacopoRodeschini/geossm.git
      cd geossm

#. Create the ``geossm`` Conda environment from :file:`environment.yml`
   (using the faster ``libmamba`` solver is recommended):

   .. code-block:: bash

      conda config --set solver libmamba
      conda update -n base -c defaults conda
      conda env create -f environment.yml

#. Activate the environment and install the package in editable mode:

   .. code-block:: bash

      conda activate geossm
      pip install -e .

Verifying the installation
----------------------------

.. code-block:: python

   import geossm
   print(geossm.__version__)

Optional extras
-----------------

geossm defines a few optional dependency groups in :file:`pyproject.toml`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Purpose
   * - ``test``
     - ``pytest`` / ``pytest-cov`` for running the test suite.
   * - ``docs``
     - Sphinx, the Furo theme, and sphinx-design, to build this
       documentation locally (see :doc:`user_guide/index`'s development
       notes below).
   * - ``dev``
     - Linting (``ruff``), type checking (``mypy``), packaging
       (``build``, ``twine``), and pre-commit hooks, on top of ``test``.
   * - ``interactive``
     - Jupyter/IPython tooling for interactive/notebook usage.

Install one or more extras with, e.g.:

.. code-block:: bash

   pip install -e ".[docs]"
   pip install -e ".[dev,test]"

Building this documentation locally
--------------------------------------

Once the ``docs`` extra is installed (on top of a full geossm installation,
since the API reference imports the package), build the HTML docs with:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

and preview them with a simple HTTP server:

.. code-block:: bash

   python -m http.server --directory docs/build/html 8000

Then open http://localhost:8000 in a browser. The ``docs/build/`` directory
is generated output and is not committed to the repository (see
:file:`.gitignore`).

Uninstalling
-------------

.. code-block:: bash

   pip uninstall geossm
   # and, if you created it, the Conda environment:
   conda remove -n geossm --all
