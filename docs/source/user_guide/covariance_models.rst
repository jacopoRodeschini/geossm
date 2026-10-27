Covariance models and meshes
===============================

Large spatial datasets make a classical, dense Matérn covariance matrix
too expensive to work with directly (an ``N x N`` matrix for ``N``
sites). geossm instead builds the latent spatial field's covariance
through a **finite-element (SPDE) approximation**: the Matérn field is
represented as the solution of a stochastic partial differential
equation, discretized on a triangular mesh covering the spatial domain,
which yields a *sparse* precision matrix. This is what
:class:`~geossm.stmodel.LRStateSpaceModel` uses as its low-rank spatial
basis (see :doc:`low_rank_ssm`).

Building a mesh
-----------------

:func:`~geossm.covmodel.utils.buildMesh2d` builds a 2D triangular mesh
(via `gmsh <https://gmsh.info/>`_) covering a spatial domain, with a
vertex budget and an element size that follows the density of the
observed locations:

.. code-block:: python

   from geossm.covmodel.utils import buildMesh2d

   mesh_io, convex_hull = buildMesh2d(
       points, lowrank=0.5, domain=domain, offset=0.3
   )

- ``points`` are the observed locations; they drive the default domain
  and the local mesh density.
- ``lowrank`` (in ``(0, 1]``) sets the budget to
  ``round(lowrank * len(points))`` vertices inside the domain (or give
  ``n_vertices`` directly) -- refined where observations are dense,
  coarser elsewhere. This is the main lever for trading spatial
  resolution against computational cost in
  :class:`~geossm.stmodel.LRStateSpaceModel`.
- ``domain`` is the scientific-interest region (one or more polygons);
  the mesh is built over its convex hull, extended outward by ``offset``
  to limit boundary effects on the SPDE solution.
- ``eps`` (uniform share of the budget), ``gradation`` (smoothness of
  the size transitions) and ``anchor_spacing`` (co-locate vertices with
  a subset of ``points``) tune the mesh; see the function's docstring.

:func:`~geossm.covmodel.utils.buildMeshGrid2d` builds a regular-grid
mesh instead of a point-driven one. See :doc:`../examples` for scripts
using each of them.

The SPDE/Matérn covariance
-----------------------------

:class:`~geossm.covmodel.SPDEApproxCov` (a subclass of
:class:`gstools.covmodel.Matern`) solves the SPDE

.. math::

   (\kappa^2 - \Delta)^{\alpha/2} x = W

by finite elements on a mesh, with Neumann boundary conditions, giving a
sparse precision matrix for the discretized field (optionally
marginalized onto the domain's "inner" vertices). It is built by first
constructing a mesh (see above), then calling ``setup()``:

.. code-block:: python

   from geossm.covmodel import SPDEApproxCov

   cov_fun = SPDEApproxCov(latlon=True, nu=1, var=1.0, rescale=1.0)
   cov_fun = cov_fun.setup(mesh_io, domain=domain)
   print(cov_fun.summary())

- ``nu`` is the Matérn smoothness parameter.
- ``rescale`` acts as the SPDE's :math:`\kappa`, controlling the
  spatial correlation range.
- ``var`` is the marginal variance of the field.
- ``latlon=True`` treats coordinates as geographic (longitude/latitude)
  rather than planar, so distances are computed accordingly.

Methods inherited unchanged from :class:`gstools.covmodel.Matern` (e.g.
``variogram``, ``covariance``, ``correlation``) describe the *nominal*,
stationary, infinite-domain Matérn process that the SPDE approximates --
useful for diagnostics against the nominal target kernel, but not an
exact description of the finite-mesh model. For the actual fitted
covariance, use ``precision()`` (see the full API reference for
:class:`~geossm.covmodel.SPDEApproxCov`).

:class:`~geossm.covmodel.FEMSolver` is the lower-level finite-element
engine behind ``SPDEApproxCov`` (mesh handling, mass/stiffness matrix
assembly, mesh diagnostics/plotting via ``plot_mesh``); most users only
need it directly for mesh inspection/plotting, as in the quickstart.
