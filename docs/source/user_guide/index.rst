User Guide
===========

This guide explains the concepts behind geossm's building blocks and how
they fit together. It complements the :doc:`../quickstart` (a single
worked example) and the :doc:`../api` (full parameter-level reference).

The typical pipeline is:

#. **Data preparation** -- turn a :class:`geopandas.GeoDataFrame` and a
   model formula into design matrices organized by space and time.
#. **Covariance modeling** -- build a mesh over the spatial domain and a
   finite-element (SPDE) approximation of the Matérn covariance on it.
#. **State-space modeling** -- filter, smooth, and estimate a
   linear-Gaussian state-space model, either directly from system
   matrices or, for large spatial domains, through the low-rank model
   that ties the previous two steps together.

.. toctree::
   :maxdepth: 1

   data_preparation
   covariance_models
   state_space_models
   low_rank_ssm

Development: building this documentation
-------------------------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/build/html
   python -m http.server --directory docs/build/html 8000

See :doc:`../installation` for the full local-build and preview
instructions.
