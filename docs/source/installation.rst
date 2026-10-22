Installation
=============

Requirements
-------------

- **OS**: Linux (x86-64). macOS and Windows are not supported: the ``mfem``
  wheels required on Python 3.10 are Linux-only.
- **Python**: 3.10 to 3.13 (tested). On 3.10, pip resolves ``jax`` 0.6.x,
  the last JAX series that supports it.
- Key runtime dependencies: ``geopandas`` (with ``pyarrow``, ``pyogrio`` and
  ``pyproj`` for its I/O and CRS support), ``pandas``, ``numpy``, ``scipy``,
  ``jax``/``jaxlib``, ``statsmodels``, ``matplotlib``, and the mesh/FEM stack
  (``shapely``, ``gmsh``, ``meshio``, ``pygmsh``, ``mfem``, ``gstools``).

:file:`pyproject.toml` in the repository holds the complete dependency list and
its supported ranges. The lower bounds are tested in CI. The wheel includes the
example datasets, so it is about 43 MB.

Option 1: pip
--------------

.. code-block:: bash

   pip install geossm

Option 2: from source with Conda (recommended for development)
-----------------------------------------------------------------

:file:`environment.yml` takes the native geospatial/mesh stack (``geopandas``,
``pyarrow``, ``gmsh``) from conda-forge, then uses pip to install geossm in
editable mode with the ``test`` and ``interactive`` extras. The CI workflows
use the same recipe. Versions are not pinned: the supported ranges live in
:file:`pyproject.toml`.

#. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/jacopoRodeschini/geossm.git
      cd geossm

#. Create and activate the ``geossm`` Conda environment:

   .. code-block:: bash

      conda env create -f environment.yml
      conda activate geossm

#. (GPU only) Install the CUDA build of JAX:

   .. code-block:: bash

      pip install -U "jax[cuda12]"

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
