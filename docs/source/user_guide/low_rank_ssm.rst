Low-rank state-space models
==============================

:class:`~geossm.stmodel.LRStateSpaceModel` is geossm's model for
large-scale spatial and spatio-temporal data. It subclasses
:class:`~geossm.ssm.StateSpaceModel` (:doc:`state_space_models`), but
instead of requiring the system matrices to be supplied directly, it
builds them from:

- a :class:`geopandas.GeoDataFrame` of observations,
- one Patsy/R-style ``formula`` per response variable (block), and
- a spatial ``domain`` and a fitted covariance model
  (:class:`~geossm.covmodel.spdeAppoxCov`, see :doc:`covariance_models`)
  giving a **low-rank basis** for the latent spatial field.

Representing the latent field through the low-rank, mesh-based SPDE
basis (rather than one latent dimension per observed site) is what makes
the model scale to a large number of spatial locations, per the
package's associated research (see the citations on the
:doc:`../index` page).

Typical workflow
-------------------

This is the same workflow used in :doc:`../quickstart`:

.. code-block:: python

   from geossm.stmodel import LRStateSpaceModel
   from geossm.covmodel import spdeAppoxCov
   from geossm.covmodel.utils import buildMesh2d

   # 1. Build the model from the data, formula(s) and domain
   model = LRStateSpaceModel(geodf, ["y ~ 1 + x1"], domain=domain)

   # 2. Build a mesh and a covariance model, then attach it
   mesh_io, _ = buildMesh2d(model.points[0], domain=domain)
   cov_fun = spdeAppoxCov(latlon=True).setup(mesh_io, domain=domain)
   model = model.setup(cov_fun=[cov_fun])

   # 3. Fit
   results = model.fit()
   print(results.summary())

``fit()`` estimates the model parameters (regression coefficients,
observation noise variance, and the spatial covariance parameters) with
an EM-type procedure built on the parent class's Kalman filter/smoother;
:class:`~geossm.stmodel.FitOptions` controls the stopping rule
(``max_iter``, ``tol_relat``), and :class:`~geossm.stmodel.ModelParams`
/ :class:`~geossm.stmodel.Param` let you pass custom initial values or
fix specific parameters instead of estimating them (see
`examples/example_LRSSM_estimate_2.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_estimate_2.py>`_
and
`example_LRSSM_estimate_3.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_estimate_3.py>`_).

Multiple response variables / domains
-----------------------------------------

``formulas`` and ``domain`` are lists so that several response
variables (each with its own spatial domain and covariance model) can
be modeled jointly, as separate "blocks" sharing the same latent
state-space machinery.

Results, inference, and prediction
--------------------------------------

:class:`~geossm.stmodel.LRStateSpaceResults` extends
:class:`~geossm.ssm.StateSpaceResults` with:

- :meth:`~geossm.stmodel.LRStateSpaceResults.compute_cov_params` and the
  derived :attr:`~geossm.stmodel.LRStateSpaceResults.bse`,
  :attr:`~geossm.stmodel.LRStateSpaceResults.tvalues`,
  :attr:`~geossm.stmodel.LRStateSpaceResults.pvalues`,
  :meth:`~geossm.stmodel.LRStateSpaceResults.conf_int` -- parameter
  standard errors and significance from the observed-information
  Hessian (see
  `examples/example_LRSSM_inference.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_inference.py>`_).
  Before calling ``compute_cov_params()``, these show as ``NaN``
  placeholders in ``summary()``.
- :attr:`~geossm.stmodel.LRStateSpaceResults.aic` /
  :attr:`~geossm.stmodel.LRStateSpaceResults.bic` -- information
  criteria for model comparison.
- :meth:`~geossm.stmodel.LRStateSpaceResults.predict` and
  :meth:`~geossm.stmodel.LRStateSpaceResults.to_geo` -- predict at new
  locations/times and export fitted/predicted values back to a
  :class:`geopandas.GeoDataFrame`.
- :meth:`~geossm.stmodel.LRStateSpaceResults.back_transform` -- map
  parameters back through a formula transformation (e.g. undo a
  ``np.sqrt(...)`` response transform) via the delta method.

See :doc:`../examples` for the full set of runnable LRSSM scripts, and
:doc:`../api` for the complete parameter reference.
