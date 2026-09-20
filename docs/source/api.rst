API Reference
==============

This reference is generated from the docstrings in the source code via
``sphinx.ext.autosummary`` / ``sphinx.ext.autodoc``. Each entry below links
to a page with the full signature, parameter descriptions, and (for
classes) a summary of public methods and attributes. Private helpers
(names prefixed with ``_``) are intentionally not part of this reference.

Main package
-------------

Importing :mod:`geossm` re-exports the main classes and functions from
its submodules (:mod:`geossm.covmodel`, :mod:`geossm.datasets`,
:mod:`geossm.ssm`, :mod:`geossm.stmodel`, plus :mod:`geossm.data_preparation`
and :mod:`geossm.utils`) at the top level, e.g. ``geossm.LRStateSpaceModel``
is the same object as ``geossm.stmodel.LRStateSpaceModel``. The sections
below document each name once, grouped by the submodule it is defined in;
``geossm.__version__`` holds the installed package version.

Models
-------

State-space model
~~~~~~~~~~~~~~~~~~~

The full-rank linear-Gaussian state-space model: Kalman filtering,
smoothing, simulation, and prediction.

.. currentmodule:: geossm.ssm

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   StateSpaceModel
   StateSpaceResults

Low-rank state-space model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

geossm's flagship model for large spatial domains: a state-space model
whose latent spatial field is represented through a low-rank, SPDE-based
covariance basis (see *Covariance* below).

.. currentmodule:: geossm.stmodel

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   LRStateSpaceModel
   LRStateSpaceResults
   Param
   ModelParams
   FitOptions

Covariance
-----------

Finite-element (SPDE) approximation of the Matérn covariance, and the mesh
builders used to discretize the spatial domain it is defined on.

.. currentmodule:: geossm.covmodel

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   spdeAppoxCov
   FEMSolver
   buildMesh2d
   buildMesh2d_density
   buildMesh2d_new
   buildMeshGrid2d

Spatial data preparation
--------------------------

Turning a :class:`geopandas.GeoDataFrame` plus a model formula into the
design matrices consumed by the models above, and the datasets bundled
with the package.

.. currentmodule:: geossm.data_preparation

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   DesignMatrices
   DesignMatricesBuilder
   check_regular_timestamps

.. currentmodule:: geossm.datasets

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   load_dataset
   list_datasets

Utilities
----------

General-purpose helpers shared across the package.

.. currentmodule:: geossm.utils

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   block_diag_3D
   write
   getHardware
   KeyStream
