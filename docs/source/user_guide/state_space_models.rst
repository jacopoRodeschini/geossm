State-space models
=====================

:class:`~geossm.ssm.StateSpaceModel` implements a classical
**linear-Gaussian state-space model**: an unobserved (latent) state that
evolves over time according to a linear transition, and observations
that are a noisy linear function of that state. In geossm's notation:

- **Observation equation**: relates the observed data at time ``t`` to
  the latent state (via the observation/design matrix ``H``), plus an
  optional exogenous-regressor term (``Xbeta``, ``beta``) and Gaussian
  observation noise with covariance ``R``.
- **State equation**: propagates the latent state forward in time via
  the transition matrix ``F``, with Gaussian process noise covariance
  ``Q``, starting from an initial state ``x0`` with covariance
  ``Sigma0``.

Given these system matrices, :class:`~geossm.ssm.StateSpaceModel`
provides:

- :meth:`~geossm.ssm.StateSpaceModel.filter` -- the Kalman filter,
  producing the filtered (one-step-ahead-updated) state mean/covariance
  at each timestep from the data observed up to that point.
- :meth:`~geossm.ssm.StateSpaceModel.smoother` -- a backward (Rauch-Tung-Striebel-style)
  pass that refines the filtered estimates using *all* observations,
  not just those up to time ``t``.
- :meth:`~geossm.ssm.StateSpaceModel.estimate` -- runs filtering and
  smoothing together.
- :meth:`~geossm.ssm.StateSpaceModel.sim` -- simulate data from the
  model's system matrices.
- :meth:`~geossm.ssm.StateSpaceModel.predict` -- project the state
  forward without new observations.

:class:`~geossm.ssm.StateSpaceResults` wraps the output of
`estimate`/`filter`/`smoother` with derived summaries: residuals,
MSE/RMSE, confidence intervals, and a `statsmodels`-style
:meth:`~geossm.ssm.StateSpaceResults.summary`.

The filter/smoother recursions are implemented as JAX kernels
(vectorized and, optionally, GPU-accelerated via the ``backend``
argument), which is what lets :class:`~geossm.stmodel.LRStateSpaceModel`
(see :doc:`low_rank_ssm`) reuse the same machinery for much larger
spatial state vectors.

Minimal example
-----------------

See `examples/example_SSM_build.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_SSM_build.py>`_,
`example_SSM_filter.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_SSM_filter.py>`_
and `example_SSM_smooth.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_SSM_smooth.py>`_
for constructing a model directly from system matrices and running the
filter/smoother. In practice, most geossm users will instead build a
:class:`~geossm.stmodel.LRStateSpaceModel` (:doc:`low_rank_ssm`), which
builds and manages its own system matrices from a
:class:`geopandas.GeoDataFrame`, a formula, and a spatial covariance
model, rather than constructing ``H``/``F``/``Q``/``R`` by hand.

See :doc:`../api` for the full parameter-level documentation of every
system matrix and method.
