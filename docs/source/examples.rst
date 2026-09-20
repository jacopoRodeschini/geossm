Examples
=========

The `examples/ <https://github.com/jacopoRodeschini/geossm/tree/develop/examples>`_
directory in the repository contains runnable scripts demonstrating the
package end to end. They are plain Python scripts with ``# %%`` cell
markers (compatible with the Jupyter/VS Code "run cell" workflow), not
Sphinx-executed notebooks, so this page links to them rather than
reproducing their full content.

Run any of them from a checkout of the repository, e.g.:

.. code-block:: bash

   cd examples
   python example_datasets_load.py

Datasets
---------

- `example_datasets_load.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_datasets_load.py>`_
  -- list and load the bundled example datasets (:mod:`geossm.datasets`).
- `example_datasets_grid.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_datasets_grid.py>`_
  -- create and manipulate spatial grids from a dataset.

Design matrices
-----------------

- `example_DesignMatrix_OLS.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_DesignMatrix_OLS.py>`_
  -- build a design matrix with :class:`~geossm.data_preparation.DesignMatricesBuilder`
  and fit a plain OLS regression against it.
- `example_DesignMatrix_OLS_simulation.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_DesignMatrix_OLS_simulation.py>`_
  -- simulate data and recover known coefficients through the same
  design-matrix pipeline.
- `example_Params_ad.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_Params_ad.py>`_
  -- work with :class:`~geossm.stmodel.Param` / :class:`~geossm.stmodel.ModelParams`
  directly.

Covariance models and meshes
-------------------------------

- `example_mesh_1.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_mesh_1.py>`_,
  `example_mesh_italy.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_mesh_italy.py>`_,
  `example_mesh_lowrank.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_mesh_lowrank.py>`_
  -- build 2D meshes with :func:`~geossm.covmodel.utils.buildMesh2d` and its
  density/grid variants.
- `example_Cov_Model.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_Cov_Model.py>`_,
  `example_Cov_Model_italy.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_Cov_Model_italy.py>`_
  -- build a mesh and set up :class:`~geossm.covmodel.spdeAppoxCov`, the
  finite-element Matérn covariance approximation.
- `example_spde_pyfem.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_spde_pyfem.py>`_,
  `example_GP_FEM_1.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_GP_FEM_1.py>`_
  -- lower-level finite-element (:class:`~geossm.covmodel.FEMSolver`) usage.

State-space models (full-rank)
---------------------------------

- `example_SSM_build.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_SSM_build.py>`_
  -- construct a :class:`~geossm.ssm.StateSpaceModel` directly from system
  matrices.
- `example_SSM_estimate.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_SSM_estimate.py>`_
  -- estimate model parameters.
- `example_SSM_filter.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_SSM_filter.py>`_,
  `example_SSM_smooth.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_SSM_smooth.py>`_
  -- run the Kalman filter and RTS smoother.

Low-rank state-space models
-------------------------------

- `example_LRSSM_build.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_build.py>`_
  -- the workflow used in :doc:`quickstart`: load data, build a mesh,
  attach a covariance model, and initialize a
  :class:`~geossm.stmodel.LRStateSpaceModel`.
- `example_LRSSM_estimate.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_estimate.py>`_,
  `example_LRSSM_estimate_2.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_estimate_2.py>`_,
  `example_LRSSM_estimate_3.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_estimate_3.py>`_
  -- fit the model with custom :class:`~geossm.stmodel.FitOptions`, initial
  parameter values, and fixed parameters.
- `example_LRSSM_inference.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_inference.py>`_
  -- standard errors, significance, and confidence intervals via
  ``results.compute_cov_params()``.
- `example_LRSSM_simulate.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_simulate.py>`_
  -- simulate data from a (fitted or user-specified) model.
- `example_LRSSM_prediction.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_prediction.py>`_
  -- predict at new locations/times and export results with ``to_geo()``.
- `example_LRSSM_cpu_gpu.py <https://github.com/jacopoRodeschini/geossm/blob/develop/examples/example_LRSSM_cpu_gpu.py>`_
  -- select the JAX compute backend (``backend='cpu'|'gpu'|'auto'``).

Case studies
-------------

The `case_studies/ <https://github.com/jacopoRodeschini/geossm/tree/develop/case_studies>`_
directory contains larger, applied scripts (e.g. a PM10 air-quality
analysis) that combine several of the pieces above on a full dataset,
closer to a real analysis than a minimal example.
