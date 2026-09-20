Quick Start
============

This walks through the shortest realistic workflow: load a bundled
dataset, build a spatial mesh, attach a covariance model, and fit a
:class:`~geossm.stmodel.LRStateSpaceModel`. A full runnable version with
plotting is available at
`examples/example_LRSSM_build.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_build.py>`_
in the repository; see :doc:`examples` for more.

1. Load a dataset
-------------------

geossm ships a couple of example spatio-temporal datasets. Each one
returns a :class:`geopandas.GeoDataFrame` of observations together with a
shapefile describing the region they cover.

.. code-block:: python

   import geossm.datasets as ds

   # List the datasets bundled with the package
   print(ds.list_datasets())

   # Load the Agrimonia air-quality dataset
   agri, shape = ds.load_dataset("agrimonia")
   print(agri.head())

2. Define the spatial domain
------------------------------

The domain is the region over which the latent spatial field is
represented. Here it is derived from the dataset's own shapefile,
buffered outward so the mesh extends a little past the observed area.

.. code-block:: python

   from shapely.geometry import Polygon

   boundary = list(shape.geometry[0].geoms)[0].boundary
   buffer = list(boundary.buffer(0.3).boundary.geoms)[0]
   domain = [Polygon(buffer)]

3. Build the low-rank state-space model
------------------------------------------

:class:`~geossm.stmodel.LRStateSpaceModel` takes the observations, one
Patsy/R-style formula per response variable, and the corresponding
domain(s). The formula below regresses ``AQ_pm10`` on an intercept and
temperature.

.. code-block:: python

   from geossm.stmodel import LRStateSpaceModel

   model = LRStateSpaceModel(
       agri, ["AQ_pm10 ~ 1 + WE_temp_2m"], verbose=True, domain=domain
   )
   print(model)

4. Build a mesh and attach a covariance model
-------------------------------------------------

The latent spatial field is represented on a triangular mesh, via a
finite-element (SPDE) approximation of the Matérn covariance
(:class:`~geossm.covmodel.spdeAppoxCov`).
:func:`~geossm.covmodel.utils.buildMesh2d` builds a mesh covering the
domain, refined around the observation locations:

.. code-block:: python

   from geossm.covmodel import spdeAppoxCov
   from geossm.covmodel.utils import buildMesh2d

   points = model.points[0]  # observation locations for this formula/block
   mesh_io, _ = buildMesh2d(points, domain=domain, max_edge=0.35, min_edge=0.05)

   cov_fun = spdeAppoxCov(latlon=True).setup(mesh_io, domain=domain)
   model = model.setup(cov_fun=[cov_fun])

5. Fit the model
-------------------

Fitting runs an EM-type procedure built on the Kalman filter/smoother.

.. code-block:: python

   results = model.fit()
   print(results.summary())

``results`` is a :class:`~geossm.stmodel.LRStateSpaceResults`: use
``results.compute_cov_params()`` to obtain standard errors and
significance (``results.bse``, ``results.pvalues``, ``results.conf_int()``),
``results.predict(...)`` to predict at new locations, and
``results.to_geo()`` to export fitted/predicted values back to a
GeoDataFrame.

Next steps
-----------

- :doc:`user_guide/index` explains the concepts (design matrices,
  covariance models, meshes, state-space models) behind each step above.
- :doc:`examples` links to further runnable scripts, including model
  simulation, prediction, and parameter inference.
- :doc:`api` documents every public class and function used here in
  detail.
