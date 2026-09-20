.. geossm documentation master file

==============================================
geossm: Geostatistics with State-Space Models
==============================================

**geossm** is a Python package for applying **state-space models** to
**spatial and spatio-temporal data**. It works natively with
:class:`geopandas.GeoDataFrame` objects and is built for modern
geostatistical workflows in environmental, climate, and geospatial
applications.

The package couples classical linear-Gaussian state-space modeling
(Kalman filtering and smoothing) with a finite-element (SPDE)
approximation of the Matérn covariance, so that large spatial and
spatio-temporal datasets can be modeled through a **low-rank
state-space representation** without giving up a principled spatial
covariance structure.

It is developed by `Jacopo Rodeschini <https://github.com/jacopoRodeschini>`_
and is built on the research presented in the PhD thesis *A State-Space
Modelling Framework in Geostatistics with Application to Environmental
Data*, and in Rodeschini et al., *"Multivariate Low-Rank State-Space Model
with SPDE Approach for High-Dimensional Data"*, Spatial Statistics (2025).

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Installation
      :link: installation
      :link-type: doc

      Install geossm with pip or via the provided Conda environment.

   .. grid-item-card:: :octicon:`play` Quick Start
      :link: quickstart
      :link-type: doc

      Load a bundled dataset and fit a low-rank state-space model in a
      few steps.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide/index
      :link-type: doc

      Concepts behind data preparation, covariance models, and the
      state-space models themselves.

   .. grid-item-card:: :octicon:`code-square` API Reference
      :link: api
      :link-type: doc

      Full reference for every public module, class, and function.

Who is this for?
=================

geossm targets researchers and practitioners working on geostatistical
and spatio-temporal problems (e.g. air quality, climate, environmental
monitoring) who want to:

- fit classical state-space models (Kalman filter/smoother) to
  spatio-temporal observations tied to a :class:`geopandas.GeoDataFrame`;
- scale that modeling approach to a large number of spatial locations via
  a low-rank, SPDE-based spatial basis (:class:`~geossm.stmodel.LRStateSpaceModel`);
  and
- go from raw geospatial data to fitted parameters, predictions, and
  diagnostics using a small, composable set of building blocks (design
  matrices, covariance models, meshes, state-space models).

Key features
=============

- **GeoDataFrame-native**: work directly with :mod:`geopandas` objects.
- **State-space modeling**: filtering, smoothing, simulation, and
  prediction for linear-Gaussian systems (:class:`~geossm.ssm.StateSpaceModel`).
- **Low-rank spatial models**: :class:`~geossm.stmodel.LRStateSpaceModel`
  scales the above to large spatial domains via an SPDE/finite-element
  covariance approximation (:class:`~geossm.covmodel.spdeAppoxCov`).
- **Formula-based design matrices**: build regressors from a Patsy/R-style
  formula string against a :class:`geopandas.GeoDataFrame`
  (:class:`~geossm.data_preparation.DesignMatricesBuilder`).
- **JAX-accelerated**: filtering/smoothing/estimation kernels run on
  CPU or GPU through `JAX <https://docs.jax.dev/en/latest/>`_.

Project links
==============

- Source code and issue tracker: https://github.com/jacopoRodeschini/geossm
- License: `MIT <https://github.com/jacopoRodeschini/geossm/blob/main/LICENSE>`_
- Package version documented here: |release|

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide
   :hidden:

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
