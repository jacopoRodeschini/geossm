"""
Low-Rank State-Space Model (LRSSM) for large-scale spatio-temporal data.

This module implements :class:`LRStateSpaceModel`, the package's model for
geostatistical data with a large number of observation sites, where a
full-rank spatial covariance over all sites is not tractable. The latent
spatial field for each response variable ("block") is instead represented
in a low-rank basis derived from a finite-element (SPDE/GMRF) discretization
of a Matern-type covariance over a mesh covering the spatial domain (see
:class:`geossm.covmodel.spdeAppoxCov`), and the model is fit with an EM
algorithm built around the Kalman filter/smoother of the parent
:class:`geossm.ssm.StateSpaceModel` class (`_E_step`/`_M_step`).
"""
import numpy as np
import jax.numpy as jnp

# inmport the state space model
from geossm.ssm.statespace import StateSpaceModel, _filter_kernelJAX, _itype_for
from geossm.covmodel.covmodels import spdeAppoxCov
from statsmodels.iolib.summary import Summary

import time
import warnings
from jax.scipy.linalg import block_diag
import jax
from scipy.optimize import minimize

from geossm.data_preparation import DesignMatricesBuilder
from geossm.utils import _select_device, _to_backend, _on_device, split_by_block, format_info_table, _summary_table_source
from geossm.utils import block_diag_3D, _check_x64_for_dtype, _validate_domain, _domain_hull, _print_info


from shapely.geometry import MultiPoint
from dataclasses import replace
from geossm.stmodel.param import Param, FitOptions, ModelParams

from .stmodel_results import LRStateSpaceResults
from geossm.stmodel._fitted import (FittedState, _read_only_copy, _spde_marginal_precision, mesh_arrays,
                                    latent_basis, build_H, build_RF, observed_loglik, cov_sizes)
from geossm.stmodel._design import prediction_design, split_design, stack_design
from geossm.stmodel._em import (_compute_inital_values_jax_kernel, _compute_beta_jax_kernel,
                               _compute_s2e_jax_kernel, _compute_A2_jax_kernel, _ks_value_and_grad)


# %% Low Rank State-Space Model adapter to statsmodels MLEModel API


def _continuous_dividers(text, labels):
    """Redraw section-divider rows (e.g. "Grid ...", "Latent. ...") as a
    single unbroken dashed line spanning the full row width.

    `add_table_2cols` renders a divider row as up to four separate cells
    (left name/value, right name/value), each padded/truncated to that
    column's own width, so a fixed-length "-" * N run lines up with the
    surrounding columns only by coincidence and otherwise leaves visible
    gaps. This instead finds each already-rendered divider line (one whose
    only content after a known `label` is dashes/whitespace) and replaces
    it wholesale with `label` followed by dashes filling the rest of the
    line, using the real width of the rendered table.
    """
    lines = text.split("\n")
    width = max((len(line) for line in lines), default=0)

    out = []
    for line in lines:
        stripped = line.strip()
        label = next(
            (
                lb for lb in labels
                if stripped.startswith(lb)
                and stripped[len(lb):].strip("- \t") == ""
            ),
            None,
        )
        if label is None:
            out.append(line)
        else:
            out.append(f"{label} {'-' * max(width - len(label) - 1, 3)}")

    return "\n".join(out)


class LRStateSpaceModel(StateSpaceModel):
    """
    Low-Rank State-Space Model (LRSSM) for large-scale spatio-temporal data.

    Subclass of :class:`geossm.ssm.StateSpaceModel` specialised for
    geostatistical data observed at many sites over time. Rather than a
    full-rank spatial covariance over all observation sites (which does not
    scale), the latent spatial field of each response variable ("block",
    one per formula) is written as a linear combination of a low-rank set
    of latent factors living on a finite-element (SPDE/GMRF) mesh -- one
    already-`setup()`-ed :class:`geossm.covmodel.spdeAppoxCov` per latent
    factor (see `setup`). The state-space measurement matrix `H` is built
    from a loading matrix `A` (one row per response variable, one column
    per latent factor) and each factor's finite-element basis evaluated at
    the observation sites, and the model is fit with an EM algorithm built
    on the parent class's Kalman filter/smoother (`_E_step`/`_M_step`).

    Typical workflow
    -----------------
    1. Build the model from a `geopandas.GeoDataFrame`, one Patsy/R-style
       formula per response variable, and one domain polygon per formula::

           model = LRStateSpaceModel(df, formulas=["y ~ 1 + x"], domain=[poly])

    2. Build a mesh over (a buffer of) the domain and attach one already
       `setup()`-ed covariance function per latent factor::

           cov_fun = spdeAppoxCov(latlon=True).setup(mesh_io, domain=[poly])
           model = model.setup(cov_fun=[cov_fun])

    3. Fit the model with the EM algorithm::

           results = model.fit()  # a LRStateSpaceResults

    4. Optionally simulate (`sim`), predict at new locations/times
       (`predict`), and inspect/report the fit (`results.summary()`).

    See the `examples/example_LRSSM_*.py` scripts and the README Quick
    Start section for complete, runnable versions of this workflow.

    Relationship to `StateSpaceModel`
    ---------------------------------
    The subclass reuses the parent's machinery (system matrices, Kalman
    filter/smoother, prediction kernels, dtype and device handling); it is
    not a drop-in replacement for it. Intentional deviations:

    - `predict(df, modelresults, ...)` predicts at new locations from fitted
      results and returns `(y_pred, var_pred)`; the parent's
      `predict(H, x_T, P_T, ...)` maps given states.
    - `sim(formulas, seed, params, ...)` simulates from model parameters;
      the parent's `sim(seed, R, F, ...)` takes system matrices.
    - The constructor builds the design matrices from data and formulas.

    The inherited `filter`, `smoother` and `estimate` run on the model's
    current system matrices: after `fit`, those of the returned parameters
    (e.g. `model.filter(model.y_train)` reproduces the fitted
    log-likelihood); before `fit` there are none. The inherited `set`
    changes those matrices; fitted results are not affected (they are
    snapshots), and the next `fit` rebuilds them.
    """

    def __init__(self, df, formulas:list, domain:list,
        verbose=True, backend="auto", dtype=jnp.float32):
        """
        Parameters
        ----------
        df : geopandas.GeoDataFrame
            Source data: one row per observation, with a `geometry` column
            (observation site) and a time column, plus every column
            referenced by `formulas`. Copied on construction (`self.df`).
        formulas : list of str
            One Patsy/R-style formula per response variable/block (e.g.
            `"np.log(AQ_pm10) ~ 1 + standardize(WE_temp_2m)"`). Each formula
            is turned into a design matrix by its own
            `geossm.data_preparation.DesignMatricesBuilder`
            (`self.builders`). May be `None` to build an "empty" model
            (e.g. only for simulation, where `formulas` is instead passed
            directly to `sim`).
        domain : list of shapely Polygon/MultiPolygon
            One spatial domain per formula/response variable, used to keep
            only the observed sites that fall inside it (see
            `DesignMatricesBuilder._filter_domain`) -- this is the
            *measurement-equation* domain (`self.domain`/`domain` property),
            generally different from the *latent* domain of each
            covariance function set in `setup()`. Must have the same length
            as `formulas`. If `None`, each formula's domain defaults to the
            convex hull of its own observed sites (see `_setDomain`).
        verbose : bool, default True
            Default verbosity for this model's logging (`self._log`);
            individual calls can override it via their own `verbose`
            argument.
        backend : {"auto", "cpu", "gpu", ...}, default "auto"
            Compute device passed to the parent `StateSpaceModel`; JAX
            arrays created while running the model's public methods are
            pinned to it (see `geossm.utils._on_device`).
        dtype : numpy/jax dtype, default `jnp.float32`
            Floating-point precision used for the design matrices and all
            model computations. `jnp.float64` requires JAX's x64 mode to be
            enabled beforehand (``jax.config.update("jax_enable_x64", True)``);
            otherwise a `ValueError` is raised.

        Raises
        ------
        ValueError
            If `formulas` and `domain` are both given but have different
            lengths, or if `domain` fails validation (see `_checkDomain`).
        """

        # Set dtype/itype before building the observation grid below, so the
        # design matrices are built directly in the model's precision instead
        # of defaulting to float64 and being downcast later. super().__init__()
        # re-derives the same values from `dtype` further down (harmless).
        self.dtype = jnp.dtype(dtype)
        self.itype = _itype_for(self.dtype)
        _check_x64_for_dtype(self.dtype)

        self.df = df.copy()
        self.formulas = formulas
        self.verbose = verbose

        # Overwrite the model attributes
        self.model_name = "Low-Rank State Space Model"
        self.type = "Linear (Gaussian)"
        self.order = (
            1,
            0,
        )  # ARMA order (p, d, q) - not used in this model but kept for compatibility

        self._log(f"Initializing {self.model_name}...")
        self._log(f"Model type: {self.type}")
        self._log(f"Model order: {self.order}")

        if formulas is not None and domain is not None and len(formulas) != len(domain):
            raise ValueError(
                f"Number of formulas ({len(formulas)}) must match number of domains ({len(domain)})"
            )

        # Validate an explicit domain up front (see `_checkDomain`), before
        # any design-matrix construction below, so an invalid domain fails
        # fast rather than after the (possibly expensive) observation grid
        # has already been built and filtered against it. `self.domain`
        # (with its convex-hull-of-points default) is still set the usual
        # way further down, once `self.points` exists.
        if domain is not None:
            flag, msg = self._checkDomain(domain)
            if flag:
                raise ValueError(msg)

        if formulas is not None:
            self.nvar, self.points, self.gridList, self.ndim, self.pdim, self.block_p, T, self.builders, \
                self.y_train, Xbeta, self.y_name, xbeta_names, \
                self.y_train_list, self.Xbeta_list = self.BuildDesignMatrix(
                    df, formulas, verbose=verbose, domain=domain
                )

        else:
            self._log("Formulas not provided. The model will be initialized without them")

            Xbeta = None
            xbeta_names = None


        # Check and set the domain (see `_checkDomain`/`_setDomain`); a
        # domain=None defaults to the convex hull of each group's observed
        # points.
        self.domain = domain
        self._log(f"{len(self._domain)} valid domain(s) set.")

        self._cov_matern = None
        # Latent dimensions: set by setup() (None before, Section D)
        self.qdim = self.block_q = self.nlat = None

        # Inizialisate the StateSpaceModel as a null model (we will set the parameters later)
        # y_train will be used later for estimation and for the results
        super().__init__(Xbeta=Xbeta, beta=None, xbeta_names=xbeta_names, backend=backend, dtype=dtype)

    def setup(self, cov_fun: list):
        """
        Build the model's latent covariance functions, one per latent
        factor, from already-built (and already `setup()`-ed) covariance
        functions.

        Parameters
        ----------
        cov_fun : list of spdeAppoxCov
            Already-built (and already `setup()`-ed) covariance functions,
            one per latent factor, used as-is. Each factor's latent domain
            is already fixed as `covi.domain`, set when `covi` was itself
            `setup()`-ed (validated by `spdeAppoxCov`, via
            `_validate_domain`) -- generally a *different*, usually smaller
            list than `self.domain` (the measurement-equation domain, one
            entry per formula); it is not defaulted from `self.domain`.
        """
        if cov_fun is None:
            raise ValueError("cov_fun must be provided")

        self._cov_matern = []

        self._log(f"Checking {len(cov_fun)} covariance functions...")

        for i, covi in enumerate(cov_fun):
            if not isinstance(covi, spdeAppoxCov):
                raise ValueError(
                    "Covariance function must be an instance of spdeAppoxCov"
                )
            self._log(f"Cov.Fun.-{i}: rescale = {covi.rescale}, nu = {covi.nu}, var = {covi.var}")
            self._cov_matern.append(covi)

        # Latent dimension (rank), one block per latent factor -- needed by
        # sim()/fit() regardless of how _cov_matern was built.
        self.qdim = jnp.array([cov.n_latent for cov in self._cov_matern], dtype=self.itype)
        self.block_q = jnp.hstack((0, jnp.cumsum(self.qdim)))
        self._log(f"Set the latent dimension (q) to {self.block_q[-1]}")

        # set the number of latent factors (i.e. the number of covariance functions)
        self.nlat = len(self._cov_matern)

        return self

    def BuildDesignMatrix(self, df, formulas, verbose=None, domain=None):
        # Compute the design matrices
        self._log("Building observation grid...")

        # Each formula's own domain entry (if any) is forwarded to its
        # DesignMatricesBuilder, which drops observed sites outside it
        # (see `DesignMatricesBuilder._filter_domain`) before the design
        # matrix is built.
        nvar, points, gridList, ndim, pdim, block_p, T, builders = (
            self._buildObservationGrid(df, formulas, verbose=verbose, domain=domain)
        )

        self._log("Building observation grid... Done.")

        self._log("Building the design matrix...")

        # self.train will be used later for estimation and for the results
        # Xbeta_train -> Xbeta in the parant class, y_train -> y_obs in the parent class
        y_list, Xbeta_list = split_design(gridList)
        y_train, Xbeta = stack_design(y_list, Xbeta_list)

        # get response name
        y_name = [g.y_name for g in gridList]
        xbeta_names = [g.x_names for g in gridList]

        self._log("Building the design matrix... Done.")

        return (
            nvar, points, gridList, ndim, pdim, block_p, T, builders,
            y_train, Xbeta, y_name, xbeta_names, y_list, Xbeta_list,
        )

    def getDesignMatrix(self, df, verbose=None):
        """
        Build the design matrix for new (out-of-sample) locations/times --
        e.g. a held-out test set -- exactly the way `predict()` builds it
        internally, plus the response `y_obs`, so the result can be
        compared directly against `predict()`'s output (e.g.
        `y_obs - y_pred_list`).

        Thin public wrapper around `geossm.stmodel._design.prediction_design`
        (the single source of truth also used by `predict()`) with
        `include_response=True`: reuses each formula's already-fitted
        `DesignMatricesBuilder` (`self.builders`) so stateful transforms
        (e.g. `standardize()`) are evaluated with the *training* mean/std
        rather than refit on `df`, and the model's own measurement-equation
        domain (`self.domain`, fixed at `__init__` time) -- there is
        nothing left to check here that `__init__` didn't already validate
        once. Unlike `predict()`'s own use of it, `df` must contain the
        response column(s) referenced by `self.formulas`.

        Parameters
        ----------
        df : geopandas.GeoDataFrame or list of geopandas.GeoDataFrame
            New locations/times, with the same columns required by the
            model's formulas (including the response) and the same CRS as
            the training data. A single GeoDataFrame is used for every
            formula (e.g. a shared held-out set); a list with one
            GeoDataFrame per formula (see `geossm.stmodel._design.as_prediction_dataframes`) lets
            each response variable be compared at its own, different set
            of locations -- e.g. `model.getDesignMatrix([gdf_no2_test,
            gdf_pm10_test])` for a bivariate model, one test set per
            pollutant, instead of a single combined `df` that would give
            every formula the union of both pollutants' locations.
        verbose : bool, optional
            Overrides `self.verbose` for this call's logging.

        Returns
        -------
        points : list of ndarray
            One entry per response variable/formula, aligned with
            `Xbeta`'s/`y_obs`'s rows for that variable (see
            `geossm.stmodel._design.prediction_grid`).
        gridList : list of DesignMatrices
            The per-formula `DesignMatrices` `Xbeta`/`y_obs` were built
            from (timestamps, CRS, ...).
        ndim : int
            Total number of rows stacked across all response variables.
        pdim : list of int
            Per-variable row counts (`block_p[i+1] - block_p[i]`).
        block_p : ndarray
            Cumulative row-index boundaries, one more entry than `pdim`.
        T : list of int
            Per-variable number of time steps.
        Xbeta : ndarray
            Stacked, block-diagonal-by-variable fixed-effect design matrix
            -- the same array `predict()` feeds through `H`/`beta` to
            build `y_pred`.
        y_obs : ndarray
            Stacked response, one row per observation (same row order as
            `Xbeta`), evaluated from `df` with the training `y_design_info`.
        """
        verbose = self.verbose if verbose is None else verbose
        return prediction_design(df, self.builders, self.domain, verbose=verbose, include_response=True,
                                 log=self._log)

    

    @property
    def shape(self):
        """
        Dimensions of the model as `(p, q, T)`.

        Returns
        -------
        tuple of (int or None, int or None, int or None)
            `p` : total number of observed sites stacked across all response
            variables (from `y_train`/`Xbeta`, whichever is set).
            `q` : total latent dimension (rank), i.e. the sum of
            `cov.n_latent` over every latent factor set in
            `setup`. `T` : number of time steps. Any entry is `None` if the
            corresponding data (`y_train`/`Xbeta`, `_cov_matern`) has not
            been set yet.
        """
        p = None
        q = None
        T = None

        if self.y_train is not None:
            p, T = self.y_train.shape
        
        if self.Xbeta is not None:
            p, b, T = self.Xbeta.shape
        
        if self._cov_matern is not None and len(self._cov_matern) > 0:
            q = sum([cov.n_latent for cov in self._cov_matern])

        return p, q, T



    @_on_device
    def sim(self, formulas:list = None , seed=1234, params: ModelParams = None, verbose=None, stats=False):
        """
        Simulate data from the low-rank state-space model (overrides
        `StateSpaceModel.sim` with a different signature, see the class
        notes).

        Builds the model matrices `H`, `R`, `F`, `Q` from `params` (or from
        default initial values if a field is left unset, see `_parseParams`)
        and the latent covariance functions set in `setup`, then delegates
        to the parent class's `StateSpaceModel.sim`.

        Parameters
        ----------
        formulas : list of str, optional
            Patsy/R-style formulas (right-hand side only is used to build
            `Xbeta`, e.g. `"1 + temperature"`), one per response variable,
            used to build a fresh observation grid/design matrix for the
            simulation. If `None`, reuses the design matrix/grid the model
            was built with at `__init__` (`self.Xbeta`, `self.points`, ...).
            `self.formulas` must not be `None` in that case.
        seed : int, default 1234
            Random seed for the simulation, forwarded to
            `StateSpaceModel.sim`.
        params : ModelParams, optional
            Model parameters to simulate with (`beta`, `A`, `s2e`, `f`,
            `ks`, one `Param` each). Fields left as `None` fall back to the
            model's default initial values (see `_parseParams`). `ks`, if
            given, is also written into each covariance function's
            `rescale` attribute (`self.cov_function[i].rescale`).
        verbose : bool, optional
            Overrides `self.verbose` for this call's logging.
        stats : bool, default False
            If `True`, asks `StateSpaceModel.sim` to also compute and
            return variance diagnostics of the simulated state/observations
            (see `StateSpaceModel.summarize_ssm_variances`).

        Returns
        -------
        y_sim : jax.numpy.ndarray
            Simulated observations, stacked across all response variables
            (shape `(p, T)`, `p = sum(pdim)`).
        x_sim : jax.numpy.ndarray
            Simulated latent state trajectory `[x_0, ..., x_T]` (shape
            `(q, T+1)`, as in `StateSpaceModel.sim`).
        info : dict
            Metadata about the simulation: `formulas`, `y_name`,
            `xbeta_names`, `Xbeta`, `params`, `points`, `T`, `stats`
            (variance diagnostics, if `stats=True`), `qdim`, `nvar`, `nlat`,
            `pdim`, `block_p`, `block_q`, and the ad-hoc `sim_model`
            (a plain `StateSpaceModel` built with the matrices used for the
            simulation).
        tdelta : float
            Wall-clock time (seconds) spent in `StateSpaceModel.sim`.

        Raises
        ------
        ValueError
            If `formulas` and `self.formulas` are both `None`; if the
            covariance functions have not been set via `setup`; or if the
            shapes of `beta`, `A`, or `ks` in `params` are inconsistent with
            the model (`Xbeta`'s width, `(nvar, nlat)`, and the number of
            covariance functions, respectively).
        """

        self._require_setup("sim")
        if formulas is None and self.formulas is None:
            raise ValueError("Formulas must be provided for simulation")
        
        if formulas is None:
            formulas = self.formulas
            Xbeta = self.Xbeta
            Xbeta_list = self.Xbeta_list
            xbeta_names = self.xbeta_names
            y_name = self.y_name
            nvar = self.nvar
            nlat = self.nlat
            pdim = self.pdim
            block_p = self.block_p
            points = self.points
            T = self.T
            self._log("Using the formulas provided at initialization, lenght = {}.".format(len(formulas)))
        else:
            self._log("Building observation grid...")
            
            nvar, points, gridList, ndim, pdim, block_p, T, _ = (
                self._buildObservationGrid(self.df, formulas, verbose=verbose)
            )
            self._log("Building observation grid... Done.")


            # self.train will be used later for estimation and for the results
            # Xbeta_train -> Xbeta in the parant class, y_train -> y_obs in the parent class
            self._log("Building the design matrix...")
            y_list, Xbeta_list = split_design(gridList)
            y, Xbeta = stack_design(y_list, Xbeta_list)

            y_name = [g.y_name for g in gridList]
            xbeta_names = [g.x_names for g in gridList]


            # get response name
            # self.y_name = [g.y_name for g in self.gridList]
            # self.xbeta_names = [g.x_names for g in self.gridList]

            self._log("Building the design matrix... Done.")
        
        qdim = self.qdim
        block_q = self.block_q
        nlat = self.nlat

        
        # Get the model parameters (if not provided, they will be set to None and the model will use default initial values)
        self._log("Parsing the parameters (ModelParams|None)...")
        
        params = self._parseParams(params)
        params = self._cast_params(params)
        A = params.A.value
        s2e = params.s2e.value
        f = params.f.value
        beta = params.beta.value
        ks = params.ks.value

        # TODO: check if the parameters are valid (e.g. positive variance, etc.)
        
        # Chech the length of beta
        if beta is not None and len(beta) != Xbeta.shape[1]:
            raise ValueError(f"Length of beta ({len(beta)}) must match number of columns in Xbeta ({Xbeta.shape[1]})")
        
        # Check the lenght of A
        if A is None or A.shape[0] != nvar or A.shape[1] != self.nlat:
            raise ValueError(f"Shape of A ({A.shape}) must match (nvar, nlat) = ({nvar}, {self.nlat})")
        
         # Check the lenght of ks
        if ks is None or len(ks) != len(self._cov_matern):
            raise ValueError(f"Length of ks ({len(ks)}) must match number of covariance functions ({len(self._cov_matern)})")
        
        # set the covarriance rescale paramiter to the value of ks (if provided, otherwise it will be set to 1)
        if ks is not None:
            for cov, ksi in zip(self.cov_function, ks):
                cov.rescale = ksi
        
        self._log("Parsing the parameters (ModelParams|None)... Done.")
        
        # Get the dimensions of the latent variable
        qdim = np.sum(self.block_q)

        self._log("Computing the SSM model matrices H, R, F and Q...")

        # Compute the basis matrix (just one) - no boundary
        basis = latent_basis(self.cov_function, points, self.dtype)
        
        # ---- build parametrised matrices
        H = build_H(A, basis)  # dense
        self._log("Computing the H {} matrix... Done.".format(H.shape))

        # R, F = buildRF(est_s2e, est_f, pdim, qdim)
        R, F = build_RF(s2e, f, pdim, qdim, self.dtype)
        self._log("Computing the R {} matrix... Done.".format(R.shape))
        self._log("Computing the F {} matrix... Done.".format(F.shape))


        # Compute the maginal precision matrix
        # precision() returns NumPy float64; cast explicitly so the result does
        # not depend on the global x64 flag (F03).
        invQ = [jnp.asarray(fcov.precision(marginal=True), dtype=self.dtype)
                for fcov in self.cov_function]

        # Compute the block diagonal covariance matrix Q of the latent factors (i.e. the points of the latent domain)
        Q = block_diag(
            *[
                jnp.linalg.solve(mt, jnp.eye(mt.shape[0], dtype=self.dtype))
                for mt in invQ
            ]
        )
        self._log("Computing the Q {} matrix... Done.".format(Q.shape))

        self._log("Start simulating the SSM...")

        # Simulate the SSM using the parent class method (we need to pass the parameters to it)
        # Create a new SSM with the same parameters as the current model, but with the matrices H, R, F and Q computed above

        sim_model = StateSpaceModel(H=H, R=R, F=F, Q=Q,Xbeta=Xbeta, beta=beta, xbeta_names=xbeta_names, x0=None, Sigma0=None, backend=self.backend, dtype=self.dtype)

        y_sim, x_sim, variance_stats, tdelta = sim_model.sim(
            seed, R=R, F=F, H=H, Q=Q, Xbeta=Xbeta, beta=beta, block_p=block_p, block_q=self.block_q, stats=stats, verbose=verbose
        )

        self._log("Simulation done. Time elapsed: {}.".format(tdelta))

        # Per-variable/per-latent-factor views, so callers don't have to
        # re-slice y_sim/x_sim by hand with block_p/block_q themselves.
        y_sim_list = split_by_block(y_sim, block_p)
        x_sim_list = split_by_block(x_sim, block_q)

        info = {}
        info['formulas'] = formulas
        info['y_name'] = y_name
        info['xbeta_names'] = xbeta_names
        info['Xbeta'] = Xbeta
        info['Xbeta_list'] = Xbeta_list
        info['y_sim_list'] = y_sim_list
        info['x_sim_list'] = x_sim_list
        info['params'] = params
        info['points'] = points
        info['T'] = T
        info['stats'] = variance_stats
        info['qdim'] = qdim
        info['nvar'] = nvar
        info['nlat'] = nlat
        info['pdim'] = pdim
        info['block_p'] = block_p
        info['block_q'] = block_q
        info['sim_model'] = sim_model

        return y_sim, x_sim, info, tdelta    
    
    def predict(self, df, modelresults: LRStateSpaceResults, variance=True, verbose=True):
        """
        Predicted mean and point-wise variance of the response variable(s)
        at new locations/times, from an already-fitted model (overrides
        `StateSpaceModel.predict` with a different signature, see the class
        notes): a shortcut for
        `modelresults.predict(df, variance, verbose)` (see
        `LRStateSpaceResults.predict`; `LRStateSpaceResults.get_prediction`
        returns an object with the full covariance, back-transformation and
        export). The prediction is computed from the results' `fitted_state`
        alone, not from the model, and neither the model nor the results are
        modified.

        Parameters
        ----------
        df : geopandas.GeoDataFrame or list of geopandas.GeoDataFrame
            New locations/times to predict at, with the same columns
            required by the model's formulas (minus the response, which is
            not needed for prediction) and the same CRS as the training
            data. A single GeoDataFrame is used for every formula (e.g. a
            shared regular grid); a list with one GeoDataFrame per formula
            (see `geossm.stmodel._design.as_prediction_dataframes`) predicts
            each response variable at its own, different set of locations.
        modelresults : LRStateSpaceResults
            Results of a previous `fit()` call.
        variance : bool, default True
            Also compute the point-wise variance.
        verbose : bool, default True
            Verbosity of the design-matrix builders.

        Returns
        -------
        y_pred : numpy.ndarray, shape (p, T)
            Predicted mean, response variables stacked as in the fit.
        var_pred : numpy.ndarray, shape (p, T), or None
            Point-wise predictive variance `diag(H P_t H')` (`None` if
            `variance=False`).

        Raises
        ------
        ValueError
            If `modelresults` is `None`.
        """
        if modelresults is None:
            raise ValueError("Model results must be provided for prediction")
        return modelresults.predict(df, variance=variance, verbose=verbose)

    @_on_device
    def _predict(self, H, x_T, P_T=None, Xbeta=None, beta=None, std=True):
        """
        Core SSM prediction, stacked across all response variables -- the
        single source of truth used both for in-sample fitted values
        (`fit()`) and out-of-sample predictions (`predict()`). Splitting the
        result into per-variable views is a separate, explicit step (see
        `geossm.utils.split_by_block`), left to the caller. With
        `std=False` the predictive covariance is skipped and returned as
        `None` (see `StateSpaceModel.predict`).
        """
        self._log("Start Prediction the SSM...")

        # Compute the prediction of linear SSM
        tStart = time.time()
        y_hat_full, Sigma_y_hat_full = super().predict(H, x_T, P_T, Xbeta, beta, std=std)
        tdelta = time.time()- tStart

        self._log("Prediction done. Time elapsed: {}.".format(tdelta))

        return y_hat_full, Sigma_y_hat_full, tdelta

    def _get_fitted_state(self, x_smoothed=None, P_smoothed=None, P_pred_smoothed=None,
                          S11=None, S10=None, S00=None):
        """
        The state that fitted results need later (a frozen `FittedState`,
        F09): the model configuration (covariance functions, observation
        points, block sizes, names, formula builders, domains, dtypes,
        verbosity) and, from `fit`, the estimated latent state (smoothed
        means and covariances, sufficient statistics). Without them it is a
        configuration-only snapshot (used by `_observed_logL`). `params` and
        the training timestamps are added by `LRStateSpaceResults`.
        """
        return FittedState(
            cov_function=tuple(self.cov_function),
            points=tuple(_read_only_copy(p) for p in self.points),
            pdim=_read_only_copy(self.pdim),
            nvar=int(self.nvar),
            nlat=int(self.nlat),
            y_name=tuple(self.y_name),
            xbeta_names=tuple(tuple(names) for names in self.xbeta_names),
            builders=tuple(self.builders),
            domain=tuple(self.domain),
            dtype=np.dtype(self.dtype),
            itype=np.dtype(self.itype),
            verbose=bool(self.verbose),
            cov_sizes=cov_sizes(self.cov_function),
            x_smoothed=x_smoothed,
            P_smoothed=P_smoothed,
            P_pred_smoothed=P_pred_smoothed,
            S11=S11,
            S10=S10,
            S00=S00,
        )

    def _require_setup(self, method):
        """Raise a clear error if `setup()` (the latent covariance functions)
        has not been called before `method`."""
        if not self._cov_matern:
            raise ValueError(
                f"{method}() needs the latent covariance functions: call "
                f"setup(cov_fun=[...]) on the model first."
            )

    def _set_system_matrices(self, params, basis, pdim, qdim):
        """
        Build `H`, `R`, `F`, `Q` from `params` (setting the covariance
        functions' `rescale` to `params.ks`) and set them, with `x0`,
        `Sigma0` and `beta`, on the underlying `StateSpaceModel` for the
        next E-step. Returns the dense `H` and the block-diagonal precision
        `Q^-1` (used by the M-step update of `f`).
        """
        if params.ks.value is not None:
            for fcov, ksi in zip(self.cov_function, params.ks.value):
                fcov.rescale = ksi

        H = build_H(params.A.value, basis)  # dense
        R, F = build_RF(params.s2e.value, params.f.value, pdim, qdim, self.dtype)

        # Block-diagonal covariance of the latent factors, from the marginal
        # precision of each covariance function
        # precision() returns NumPy float64; cast explicitly so the result does
        # not depend on the global x64 flag (F03).
        invQ = [jnp.asarray(fcov.precision(marginal=True), dtype=self.dtype)
                for fcov in self.cov_function]
        Q = block_diag(
            *[
                jnp.linalg.solve(mt, jnp.eye(mt.shape[0], dtype=self.dtype))
                for mt in invQ
            ]
        )

        invQ = block_diag(*invQ)

        super().set(
            H=H,
            R=R,
            F=F,
            Q=Q,
            x0=params.x0.value,
            Sigma0=params.Sigma0.value,
            beta=params.beta.value,
        )
        return H, invQ

    @_on_device
    def fit(
        self, params0: ModelParams | None = None, options: FitOptions | None = None
    ):
        """
        Estimate the model parameters with an EM algorithm.

        Each EM iteration alternates: an E-step (`_E_step`) that runs the
        parent class's Kalman filter/smoother given the current parameters
        to get the smoothed states/covariances and the sufficient
        statistics `S11`/`S10`/`S00`, plus the observed-data
        log-likelihood; and an M-step (`_M_step`) that updates `beta`
        (regression coefficients), `s2e` (measurement-error variances),
        `f` (latent factors' AR(1) coefficients), `A` (loading matrix), and
        `ks` (Matern rescale/range parameters, one per covariance function,
        optimised with L-BFGS-B against exact JAX gradients -- see
        `_build_ks_value_and_grad`) from those statistics. Iteration stops
        when the log-likelihood change per observation drops to `tol_relat`
        or `max_iter` iterations are reached (see `FitOptions`; the reason
        is recorded in the results' `termination`).

        Parameters
        ----------
        params0 : ModelParams, optional
            Initial/fixed values for the parameters. Any field left as
            `None` is filled in with data-driven initial values (OLS `beta`,
            moment-based `s2e`/`A`, a mesh-based initial `ks`, ... -- see
            `_getInitialValues`); a field with `fixed=True` on its `Param`
            is held at its given value and excluded from the M-step updates
            for the whole run (see `_updateParams`). If `None`, every
            parameter is initialised and freely estimated.
        options : FitOptions, optional
            EM stopping-rule and logging options: `max_iter` caps the
            number of EM iterations, `tol_relat` is the tolerance on the
            log-likelihood change per observation below which EM is
            considered converged, and `verbose` overrides `self.verbose`
            for this call only, including the per-iteration log printed via
            `logger` (the model's `verbose` is restored afterwards). Defaults
            to `max_iter=100`, `tol_relat=1e-3` if not
            given (note this differs from `FitOptions`'s own dataclass
            defaults, which are only used if `options` itself is `None`
            -- passing an explicit `FitOptions()` uses its `max_iter=20`).

        Returns
        -------
        LRStateSpaceResults
            The fitted results: estimated parameters, per-iteration EM
            statistics (`nstats`/`llf_path`), in-sample fitted values
            (`y_hat`/`y_hat_list` and their covariances), the smoothed
            states, and the sufficient statistics `S11`/`S10`/`S00`.
            `params`, `llf`, the states, `S11`/`S10`/`S00` and `y_hat` all
            refer to the same parameter vector: the last M-step's output,
            evaluated by a final E-step (or, if that is not an improvement
            or EM stopped on a decrease/NaN, the best evaluated iterate).
            The model's system matrices are left at those parameters.
            `llf_path` is the per-iteration trace; entry ``k`` is the
            log-likelihood of the E-step of iteration ``k``, so `llf` can
            exceed `llf_path[-1]`. Entry 0 (the initial parameters, before
            any E-step) is ``nan``.

        Raises
        ------
        ValueError
            If `setup()` has not been called (no latent covariance
            functions).
        """
        self._require_setup("fit")
        # `options.verbose` applies to this call only; the model's own
        # `verbose` is restored afterwards (F17).
        verbose_before = self.verbose
        if options is not None:
            self.verbose = options.verbose
        try:
            return self._fit(params0, options)
        finally:
            self.verbose = verbose_before

    def _fit(self, params0, options):
        """EM estimation behind `fit` (see there), with `self.verbose` set
        for the call."""
        if self.verbose:
            print(self._get_summary_table())
        self._log("Starting the estimation of the model parameters using EM algorithm...")

        
        self._log("Parsing the fit options (FitOptions|None)...")

        max_iter = options.max_iter if options is not None else 100
        tol_relat = options.tol_relat if options is not None else 1e-3
        force_iter = options.force_iter if options is not None else False
        std = options.std if options is not None else False

        # Get the initial parameters (if not provided, they will be set to None and the model will use default initial values)
        # Create the est_params object, filling in the provided values and leaving the rest as None (or default) for the model to handle
        self._log("Parsing the initial parameters (ModelParams|None)...")
        params0 = self._parseParams(params0)

        # Get global constants
        nvar = self.nvar  # len(self.pdim)
        nlat = self.nlat
        # cov_function = self.cov_function
        pdim = jnp.asarray(self.pdim, dtype=self.itype)
        block_p = jnp.asarray(self.block_p, dtype=self.itype)

        # Get the observed data
        y_obs = jnp.asarray(self.y_train, dtype=self.dtype)
        Xbeta = jnp.asarray(self.Xbeta, dtype=self.dtype)

        points = self.points
        p, T = y_obs.shape
        # Observed entries: the stopping rule uses the log-likelihood change
        # per observation, which (unlike a change relative to |llf|) does not
        # depend on additive constants in the log-likelihood.
        n_obs = int(jnp.sum(~jnp.isnan(y_obs)))

        # Get latent dimension (i.e. the rank)
        qdim = jnp.array(
            [cov.n_latent for cov in self.cov_function],
            dtype=self.itype,
        )
        block_q = jnp.hstack((0, jnp.cumsum(qdim)))

        # Get the initial values
        self._log("Computing the initial parameter values...")
        est_params = self._getInitialValues(y_obs, Xbeta, block_p, block_q)

        # Set the initial values of the parameters (if not provided, they will be set to the estimated initial values)
        self._log("Updating the initial parameter values...")
        est_params = self._updateParams0(params0, est_params)

        # Compute the basis matrix (just one) - no boundary
        
        self._log("Computing the basis matrix...")
        basis = latent_basis(self.cov_function, points, self.dtype)
        Phi = build_H(jnp.ones((nvar, nlat), dtype=self.dtype), basis)

        # Build the M-step's ks (Matern rescale) objective once - JIT
        # compilation happens on its first call inside the EM loop below,
        # then every subsequent EM iteration (and every one of L-BFGS-B's
        # internal evaluations within each of them) reuses the compiled
        # function instead of rebuilding/recompiling it.
        self._log("Building the ks (rescale) M-step objective...")
        ks_value_and_grad = self._build_ks_value_and_grad()

        self._log("Starting the EM iterations...")

        # Flag of the EM convergence
        flag = True
        niter = 0
        logL_prev = jnp.nan  # no E-step before iteration 1 (F41)
        logL_cur = jnp.nan
        delta_par = jnp.nan
        delta_lik = jnp.nan
        relat_lik = jnp.nan
        tdelta_iter = 0
        tdelta_Edet = np.zeros(3)
        tdelta_Mdet = np.zeros(3)
        nstat = []  # list to store the results of each iteration
        converged = False  # True only if the tolerance was met
        termination = None  # why EM stopped: tolerance | decrease | invalid | max_iter
        best = None  # snapshot of the highest-llf (params, E-step outputs) so far
        restored = False  # True if EM stopped by restoring `best`

        # Log the initial state before starting the EM iterations
        nstat = self._log_iteration(
            nstat,
            niter,
            est_params,
            0,
            jnp.nan,  # iteration 0 has no log-likelihood value (F41)
            delta_par,
            delta_lik,
            relat_lik,
            tdelta_iter,
            tdelta_Edet,
            tdelta_Mdet,
        )

        if self.verbose:
            msg = self.logger(nstat[-1])
            print(msg)

        # Start EM iteration
        decreased = False  # flag to indicate if the log-likelihood decreased
        while flag:
            niter += 1

            # Start the timer for the iteration
            tStart_iter = time.time()

            # ---- build parametrised matrices and set them for the E step
            H, invQ = self._set_system_matrices(est_params, basis, pdim, qdim)

            # ---- E step
            # y_hat, x_T, P_T, S11, S10, S00, logL_cur, tdelta_Edet = self._E_step(
            #    y_obs, R, F, H, Q, est_params.x0.value, est_params.Sigma0.value, Xbeta, est_params.beta.value)

            y_hat, x_T, P_T, S11, S10, S00, logL_cur, tdelta_Edet = self._E_step(y_obs)

            # ---- M step, get the updated parameters
            update_params, opt_success, tdelta_Mdet = self._M_step(
                y_obs,
                y_hat,
                self.F,
                self.H,
                Xbeta,
                self.cov_function,
                block_p,
                block_q,
                x_T,
                P_T,
                S11,
                S10,
                S00,
                invQ,
                Phi,
                ks_value_and_grad,
                f_fixed=est_params.f.fixed,
            )

            # Log-likelihood change, absolute and per observation
            delta_lik = logL_cur - logL_prev
            relat_lik = delta_lik / n_obs if niter > 1 else jnp.inf

            # End the timer for the iteration
            tdelta_iter = time.time() - tStart_iter

            # Check the step. Skipped at the first iteration: there is no
            # previous E-step (logL_prev is nan), so delta_lik is nan.
            # - NaN/Inf always stops.
            # - A decrease within tol_relat counts as convergence: the iterates
            #   oscillate on the plateau.
            # - A larger decrease stops unless force_iter is True.
            # Every stop on a decrease or NaN/Inf returns the best iterate so far.
            if niter > 1:
                invalid = bool(jnp.isnan(delta_lik) or jnp.isinf(delta_lik))
                decreased = bool(delta_lik < 0)
                within_tol = not invalid and abs(float(relat_lik)) <= tol_relat
                if decreased and not within_tol and force_iter:
                    msg = self._log_warning_emstep(delta_lik, niter)
                    self._log(msg)
                    warnings.warn(msg + " (force_iter=True: continuing).", RuntimeWarning, stacklevel=2)
                elif invalid or decreased:
                    if within_tol:
                        termination, converged = "tolerance", True
                    else:
                        termination = "invalid" if invalid else "decrease"
                        msg = self._log_warning_emstep(delta_lik, niter)
                        self._log(msg)
                        warnings.warn(
                            msg + ". The EM algorithm was stopped and the best "
                            "estimate so far is returned (convergence=False).",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                    # restore the best estimate and its E-step outputs
                    est_params, H, x_T, P_T, S11, S10, S00, logL_cur = best
                    restored = True
                    break

            # snapshot the current parameters and their E-step outputs if they
            # are the best so far
            if best is None or logL_cur >= best[-1]:
                best = (est_params, H, x_T, P_T, S11, S10, S00, logL_cur)

            # Update the paramiters to be used in the next EM iteration
            est_params = self._updateParams(est_params, update_params)
            logL_prev = logL_cur

            # append the results
            nstat = self._log_iteration(
                nstat,
                niter,
                est_params,
                opt_success,
                logL_cur,
                delta_par,
                delta_lik,
                relat_lik,
                tdelta_iter,
                tdelta_Edet,
                tdelta_Mdet,
            )

            # print the results of the iteration
            if self.verbose:
                msg = self.logger(nstat[-1])       
                if decreased:
                    msg += "\n (force_iter=True: continuing)"
               
                print(msg)
               

            # Check the EM convergence (if the log-likelihood is not improving more than tol_lik or the max number of iterations is reached)
            if abs(relat_lik) <= tol_relat:
                termination, converged = "tolerance", True
                flag = False

            elif niter == max_iter:
                termination = "max_iter"
                warnings.warn(
                    "The EM algorithm reached max_iter={} without meeting "
                    "tol_relat={}.".format(max_iter, tol_relat),
                    RuntimeWarning,
                    stacklevel=2,
                )
                flag = False

        # Make every returned quantity refer to the same parameters. On a
        # normal stop est_params is the output of the last M-step, which no
        # E-step has evaluated yet: run a final E-step at it, unless it is not
        # better than the best evaluated iterate. On a restore, `best` already
        # holds its E-step outputs. Either way the model's matrices (and the
        # covariance functions' rescale) are left at the returned parameters.
        if not restored:
            self._log("Final E-step at the estimated parameters...")
            H, _ = self._set_system_matrices(est_params, basis, pdim, qdim)
            final = self._E_step(y_obs)
            logL_final = final[6]
            if bool(jnp.isfinite(logL_final)) and logL_final >= best[-1]:
                _, x_T, P_T, S11, S10, S00, logL_cur, _ = final
            else:
                if not bool(jnp.isfinite(logL_final)):
                    warnings.warn(
                        "The log-likelihood at the last M-step's parameters is "
                        "{}; the best estimate so far is returned.".format(logL_final),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                est_params, H, x_T, P_T, S11, S10, S00, logL_cur = best
                self._set_system_matrices(est_params, basis, pdim, qdim)
        else:
            self._set_system_matrices(est_params, basis, pdim, qdim)

        if converged:
            self._log("EM algorithm converged after {} iterations.".format(niter))
        else:
            self._log("EM algorithm stopped after {} iterations without converging.".format(niter))
        self._log("Final log-likelihood: {}.".format(logL_cur))
        self._log("Create the results object...")

        # predict the respose variable using the fitted model parameters
        beta_est = est_params.beta.value
        y_hat_full, Sigma_y_hat_full, tdelta_hat = self._predict(
            H, x_T, P_T, Xbeta, beta_est, std=std
        )

        # Training grid (points/timestamps/CRS), one entry per response
        # variable -- snapshotted here for the same reason as block_p above:
        # self.points/self.gridList are mutable and would go stale for this
        # results object the next time setup()/fit() runs on this model.
        points_hat = self.points
        timestamps_hat = [grid.timestamps for grid in self.gridList]
        crs_hat = self.gridList[0].crs

        # reuturn the final results as a LRStateSpaceResults object
        results = LRStateSpaceResults(
            model=self,
            params=est_params,
            nstats=nstat,
            options=options,
            convergence=converged,
            termination=termination,
            llf=logL_cur,
            fitted_state=self._get_fitted_state(x_smoothed=x_T, P_smoothed=P_T, P_pred_smoothed=None,
                                                S11=S11, S10=S10, S00=S00),
            # main arrays (stacked across all variables -- feeds the
            # generic, model-agnostic uncertainty machinery in the base
            # StateSpaceResults class: conf_int_y, coverage_probability, ...)
            y_hat=y_hat_full,
            Sigma_y_hat=Sigma_y_hat_full,
            tdelta_hat=tdelta_hat,
            # block_p: snapshotted here rather than left to be read live
            # off `self.block_p` -- that attribute is mutable and would go
            # stale for this results object the next time `fit()`/`setup()`
            # runs on this same model instance. Per-variable views
            # (y_hat_list, ...) are derived from this + the stacked arrays
            # above as lazy properties on LRStateSpaceResults.
            block_p=block_p,
            points_hat=points_hat,
            timestamps_hat=timestamps_hat,
            crs_hat=crs_hat)

        return results

    @property
    def cov_function(self):
        """
        list of spdeAppoxCov : The latent covariance functions set by
        `setup`, one per latent factor (`self._cov_matern`); `None` (or an
        empty list) until `setup` has been called.
        """
        return self._cov_matern

    def _E_step(self, y_t):
        """
        EM E-step: run the parent class's Kalman filter/smoother
        (`StateSpaceModel.estimate`, `light=True`) at the current
        parameters to get the smoothed states/covariances, the sufficient
        statistics `S11`/`S10`/`S00`, and the log-likelihood.
        """

        # E step: compute the expected values of the latent factors and the log-likelihood
        # 1) Create the SSM object with the current parameters
        # 2) Run the Kalman filter and smoother to get the expected values of the latent factors and the log-likelihood

        # Run the Kalman filter and smoother to get the expected values of the latent factors and the log-likelihood
        # Call the parent class's estimate method to perform the Kalman filter and smoother.
        # light=True: this EM loop never reads back the filter-stage arrays
        # (x_filtered, P_filtered, K, x_pred, P_pred), only the smoothed
        # states and the sufficient statistics below - so they don't need to
        # be kept alive as numpy copies on `results`.
        results = super().estimate(y_t, light=True)

        y_hat = results.y_hat
        x_T = results.x_smoothed
        P_T = results.P_smoothed
        # P_T_1 = results.P_pred_smoothed
        S11 = results.S11
        S10 = results.S10
        S00 = results.S00
        logL = results.llf
        tdelta_filter = results.time_filter
        tdelta_smoother = results.time_smoother
        tdelta_expectation = results.time_expectation

        tdelta = np.array(
            [tdelta_filter, tdelta_smoother, tdelta_expectation], dtype=jnp.float32
        )

        return y_hat, x_T, P_T, S11, S10, S00, logL, tdelta

    def _M_step(
        self,
        y_t,
        y_hat,
        F,
        H,
        Xbeta,
        est_covList,
        block_p,
        block_q,
        x_T,
        P_T,
        S11,
        S10,
        S00,
        invQ,
        Phi,
        ks_value_and_grad,
        f_fixed=False,
    ):
        """
        EM M-step: given the E-step's smoothed states/sufficient statistics,
        update `f` (closed form), `beta` (`_compute_beta_jax_kernel`), `s2e`
        (`_compute_s2e_jax_kernel`), `A` (`_compute_A2_jax_kernel`), and
        `ks` (L-BFGS-B against `ks_value_and_grad`), and package them into a
        new `ModelParams` via `_createParams`.

        The state parameters are updated in ECM order (F40): `f` maximises
        the expected complete-data log-likelihood given the current `ks`
        (`invQ`, the precision `Q^-1` of the E-step), then `ks` maximises it
        given the new `f` (the current `F` if `f_fixed`).
        """

        # convert all input to save memory
        p, T = y_t.shape
        q = block_q[-1]
        b = Xbeta.shape[1]
        nvar = len(block_p) - 1
        nlat = len(block_q) - 1
        # ndim = block_p[1:] - block_p[:-1]  # observed dimensions
        ldim = block_q[1:] - block_q[:-1]  # latent dimensions

        # Update f (Eq 3a)
        est_f = jnp.zeros((nlat), dtype=self.dtype)
        tStart = time.time()
        for q in range(nlat):
            s = slice(block_q[q], block_q[q + 1])

            # Added the fix for the f update (Eq 3a)
            # [F is parametrized as diagonal, and so we need to weight the trace by Q^{-1}]
            # tr(A B) = sum(A * B'), without forming the product
            num = jnp.sum(invQ[s, s] * S10[s, s].T)
            den = jnp.sum(invQ[s, s] * S00[s, s].T)

            # print(q, num, den)
            est_f = est_f.at[q].set(num / den)

        tdelta_f = time.time() - tStart

        # beta (the same as Eq.5 Calculli)
        err = y_t - y_hat  # compute the prediction error

        tStart = time.time()
        beta = _compute_beta_jax_kernel(b, y_t, x_T, H, Xbeta)
        jax.block_until_ready(beta)
        tdelta_beta = time.time() - tStart

        # s2 (see Eq, 14c)
        # compute the s2e
        # tStart = time.time()
        # s2e = compute_s2e(
        #     err, H, P_T, block_p, nvar, T, ndim)
        # tdelta_s2e = time.time() - tStart

        tStart = time.time()
        s2e = _compute_s2e_jax_kernel(err, H, P_T, tuple(block_p.tolist()))
        jax.block_until_ready(s2e)
        tdelta_s2e = time.time() - tStart

        # update A
        # est_A, tdelta_A = compute_A2(
        #     y_t, Xbeta, est_beta, x_T, P_T, block_p, block_q, nvar, nlat, T, ldim, Phi)

        tStart = time.time()

        est_A = _compute_A2_jax_kernel(
            y_t,
            Xbeta,
            beta,
            x_T,
            P_T,
            tuple(block_p.tolist()),
            tuple(block_q.tolist()),
            nvar,
            nlat,
            T,
            ldim,
            Phi,
        )

        jax.block_until_ready(est_A)
        tdelta_A = time.time() - tStart

        # Set the parameter of the minimise object

        tStart = time.time()
        # F is diagonal (stored as its 1D diagonal, see
        # `StateSpaceModel._prepare_diag_array`), so each F-matmul below
        # reduces to elementwise scaling by f_diag instead of a dense (q, q)
        # matmul: S10 @ F.T scales S10's columns, F @ S10.T scales S10.T's
        # rows, and F @ S00 @ F.T scales S00 by outer(f_diag, f_diag) -- the
        # same pattern already used for `FF` in `_filter_kernelJAX`.
        # ECM: Omega at the new f (F40), or at the current F if f is fixed.
        f_diag = F if f_fixed else jnp.repeat(est_f, ldim, total_repeat_length=int(block_q[-1]))
        Omega = (
            S11
            - S10 * f_diag[None, :]
            - f_diag[:, None] * S10.T
            + jnp.outer(f_diag, f_diag) * S00
        )
        par0 = jnp.log(
            jnp.array([fcov.rescale for fcov in est_covList], dtype=self.dtype)
        )

        # 'L-BFGS-B' with exact JAX gradients from `ks_value_and_grad`
        # (`_build_ks_value_and_grad`; compiled once per process and mesh
        # size, reused for every fit, EM iteration and evaluation within
        # each L-BFGS-B call): converges in far
        # fewer objective evaluations than derivative-free Nelder-Mead,
        # since it exploits the analytic gradient instead of building a
        # discrete simplex around the current point.
        def _fun_and_grad(params_np):
            val, grad = ks_value_and_grad(jnp.asarray(params_np, dtype=self.dtype), T, Omega)
            return float(val), np.asarray(grad, dtype=np.float64)

        opt = minimize(
            _fun_and_grad,
            np.asarray(par0, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            tol=1e-3,
            options={"maxiter": 50},
        )

        tdelta_ks = time.time() - tStart

        # update the initial state mu and variance
        x0 = x_T[:, 0]
        Sigma0 = P_T[:, :, 0]

        tdelta = jnp.array(
            [tdelta_beta, tdelta_s2e, tdelta_f, tdelta_ks, tdelta_A], dtype=jnp.float32
        )

        # opt.x is NumPy float64: cast before exp, so that the result does not
        # depend on the global x64 flag (F03)
        est_ks = jnp.exp(jnp.asarray(opt.x, dtype=self.dtype))
        # print("est_ks", est_ks)

        # Note that the Cov. function is updated in place with the new rescale parameter (i.e. the range parameter of the Matern covariance function)
        update_params = self._createParams(beta, s2e, est_f, x0, Sigma0, est_ks, est_A)

        return update_params, opt.success, tdelta

    def _spde_mesh_arrays(self, cov_function=None, dtype=None, itype=None):
        """
        Mesh inputs of `_spde_marginal_precision` for `cov_function` (default:
        the model's covariance functions), in the model's dtypes by default
        (see `geossm.stmodel._fitted.mesh_arrays`).
        """
        return mesh_arrays(self.cov_function if cov_function is None else cov_function,
                           self.dtype if dtype is None else dtype,
                           self.itype if itype is None else itype)

    def _build_ks_value_and_grad(self):
        """
        Value-and-gradient function `(params, T, Omega) -> (value, grad)` of
        the M-step's `ks` objective (`_ks_objective`), for L-BFGS-B with
        exact gradients. It binds this model's mesh arrays to the
        module-level compiled `_ks_value_and_grad`, so a new `fit()` (or a
        new model on a mesh of the same size) reuses the compiled function
        instead of recompiling it (F20).
        """
        mesh = self._spde_mesh_arrays()

        def value_and_grad(params, T, Omega):
            return _ks_value_and_grad(params, T, Omega, *mesh)

        return value_and_grad

    def _observed_logL(self, y_obs, Xbeta, x0, Sigma0, fitted_state=None):
        """
        `ModelParams -> scalar` closure computing the observed log-likelihood
        at fixed `y_obs`/`Xbeta`/`x0`/`Sigma0` (see
        `geossm.stmodel._fitted.observed_loglik`). `fitted_state` (a
        `FittedState`) gives the observation points, covariance functions,
        block sizes and dtypes; by default the model's current ones.
        """
        state = self._get_fitted_state() if fitted_state is None else fitted_state
        return observed_loglik(state, y_obs, Xbeta, x0, Sigma0)



    def _getInitialValues(self, y_obs, Xbeta, block_p, block_q):
        """
        Compute data-driven initial parameter values used by `fit()` when
        `params0` (or one of its fields) is not provided: OLS/moment-based
        `beta`/`s2e`/`A`/`f`/`x0`/`Sigma0` (`_compute_inital_values_jax_kernel`)
        and a mesh-based initial `ks` (from each covariance function's
        FEM bounding box).
        """

        # Compute the initial values of the parameters
        est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_A = (
            _compute_inital_values_jax_kernel(
                y_obs, Xbeta, tuple(block_p.tolist()), tuple(block_q.tolist())
            )
        )

        # Compute the initial values of the range parameters (i.e. the rescale parameter of the Matern covariance function) using the distance between the points of the latent domain (i.e. the inner points of the mesh)
        box = [cv.box for cv in self.cov_function]
        # The box is NumPy float64: cast the extents to the model dtype first
        # (as JAX did implicitly with x64 off), so that the result does not
        # depend on the global x64 flag (F03)
        dt = self.dtype
        est_ks = [
            jnp.sqrt(jnp.asarray(8.0, dtype=dt))
            / (jnp.minimum(jnp.abs(jnp.asarray(bx[2] - bx[0], dtype=dt)),
                           jnp.abs(jnp.asarray(bx[3] - bx[1], dtype=dt))) / 3.0)
            for bx in box
        ]

        # Create the est_params object with the estimated initial values (if not provided, they will be set to None and the model will use default initial values)
        est_params = self._createParams(
            est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_ks, est_A
        )

        return est_params

    def _createParams(
        self, est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_ks, est_A
    ):
        """Package raw parameter arrays into a `ModelParams` of (free) `Param`s."""
        est_params = ModelParams(
            beta=Param("beta", est_beta),
            s2e=Param("s2e", est_s2e),
            f=Param("f", est_f),
            A=Param("A", est_A),
            ks=Param("ks", est_ks),
            x0=Param("x0", est_x0),
            Sigma0=Param("Sigma0", est_Sigma0),
        )

        return est_params

    def _parseParams(self, params0: ModelParams | None):
        """Return `params0` unchanged, or an all-`None`/free `ModelParams`
        placeholder if `params0` is `None` (filled in later, e.g. by
        `_getInitialValues`/`_updateParams0`)."""

        return (
            params0
            if params0 is not None
            else ModelParams(
                beta=Param("beta", None),
                s2e=Param("s2e", None),
                f=Param("f", None),
                A=Param("A", None),
                ks=Param("ks", None),
                x0=Param("x0", None),
                Sigma0=Param("Sigma0", None),
            )
        )

    def _updateParams(self, params: ModelParams, updates: ModelParams) -> ModelParams:
        """
        Update parameters using values from `updates`,
        respecting the `fixed` flags and handling None initial values.
        """

        updated_fields = {}

        for name in params.__dataclass_fields__:
            current_param = getattr(params, name)
            new_param = getattr(updates, name)

            # ---- CASE 1: parameter is fixed, never update
            if current_param.fixed:
                updated_fields[name] = current_param
                continue

            # ---- CASE 3: normal update (free parameter with initial value)
            updated_fields[name] = replace(
                current_param,
                value=jnp.asarray(new_param.value),
            )

        new_params = self._cast_params(ModelParams(**updated_fields))

        # Special handling for ks update covariance scaling
        if not new_params.ks.fixed and new_params.ks.value is not None:
            for fcov, ksi in zip(self.cov_function, new_params.ks.value):
                fcov.rescale = ksi

        return new_params

    def _updateParams0(self, params0: ModelParams, updates: ModelParams) -> ModelParams:
        """
        Merge user-supplied `params0` (from `fit(params0=...)`) with the
        data-driven `updates` (from `_getInitialValues`): a field already
        set on `params0` (`.value is not None`) -- whether free or fixed --
        is kept as-is; a field left as `None` on `params0` is filled in
        from `updates`. Called once, before the EM loop starts.
        """

        updated_fields = {}

        for name in params0.__dataclass_fields__:
            current_param0 = getattr(params0, name)
            new_param = getattr(updates, name)

            if current_param0.value is not None:
                updated_fields[name] = current_param0
                continue

            # ---- CASE 3: normal update (free parameter with initial value)
            updated_fields[name] = replace(
                current_param0,
                value=jnp.asarray(new_param.value),
            )

        new_params = self._cast_params(ModelParams(**updated_fields))

        # Special handling for ks update covariance scaling
        if not new_params.ks.fixed and new_params.ks.value is not None:
            for fcov, ksi in zip(self.cov_function, new_params.ks.value):
                fcov.rescale = ksi

        return new_params

    def _cast_params(self, params: ModelParams) -> ModelParams:
        """
        Return `params` with every value in the model dtype. User-supplied
        values (`Param` converts them with JAX's default dtype) and
        intermediate results would otherwise follow the global x64 flag
        (F03).
        """
        fields = {}
        for name in params.__dataclass_fields__:
            p = getattr(params, name)
            if p is not None and p.value is not None:
                p = replace(p, value=jnp.asarray(p.value, dtype=self.dtype))
            fields[name] = p
        return ModelParams(**fields)

    # dense matrix
    # fix H row functions
    def _setDomain(self, polygon):
        """
        Build `self.domain`: a list with one entry per *measurement
        equation* (`len(self.points) == len(formulas)`), each entry a
        single shapely Polygon or MultiPolygon describing that equation's
        observation domain. This is generally a different count from the
        number of *latent* factors (`self.nlat`, set in `setup()`) -- there
        can be fewer latent factors than measurement equations, e.g. when
        several equations share one latent field via `A`.

        Parameters
        ----------
        polygon : list/tuple of shapely Polygon/MultiPolygon, or None
            Already validated by `_checkDomain` (via the `domain` setter)
            -- stored as-is. If None, defaults to the convex hull of each
            equation's own observed points (`self.points`).
        """
        if polygon is None:
            return [MultiPoint(pts).convex_hull for pts in self.points]

        return list(polygon)

    def _checkDomain(self, domain):
        """
        Validate `domain`: a list/tuple where each entry is a single
        shapely Polygon or MultiPolygon, one per measurement equation
        (`self.domain`) -- the length is the caller's responsibility to
        match to the right count.

        Builds on `spdeAppoxCov`/`FEMSolver`'s own contract (see
        `geossm.covmodel.covmodels._validate_domain`), applied per-entry --
        wrapping each entry the same way a `spdeAppoxCov` would
        (`spdeAppoxCov(...).setup(mesh, domain=[domi])`) -- rather than to
        the whole list at once: validating the whole list
        in one call would flatten a MultiPolygon entry into several
        separate entries and break the 1:1 correspondence with the list
        it's meant to match (`self.points`).

        `domain=None` is treated as valid here -- `_setDomain` fills in a
        default in that case.
        """
        flag = False
        msg = ""

        if domain is not None:
            if not isinstance(domain, (list, tuple)):
                return True, "domain must be a list of Polygon/MultiPolygon objects, one per latent factor"

            for i, poly in enumerate(domain):
                try:
                    _validate_domain([poly])
                except (TypeError, ValueError) as e:
                    return True, f"Domain {i}: {e}"

                bounds_str = ", ".join(f"{b:.2f}" for b in poly.bounds)
                self._log("Domain-{}: area = {:2f}, box = ({})".format(i+1, poly.area, bounds_str))

        return flag, msg

    @property
    def domain(self):
        """
        The measurement-equation domain: a list with one shapely Polygon or
        MultiPolygon per *measurement equation* (`len(formulas)`) -- see
        `_setDomain`. Distinct from the *latent* domain (one entry per
        latent factor, generally a different, usually smaller count): each
        factor's latent domain is fixed on its own `spdeAppoxCov` when that
        object is `setup()`-ed, before it is passed to `setup(cov_fun=...)`,
        and is available via `cov_function[i].domain`. See `domain_hull`
        for a single Polygon summarising every measurement equation's
        domain combined.
        """
        return self._domain

    @domain.setter
    def domain(self, value):
        """
        Validate `value` (see `_checkDomain`) and set `self.domain` (see
        `_setDomain`).
        """
        flag, msg = self._checkDomain(value)
        if flag:
            raise ValueError(msg)
        self._domain = self._setDomain(value)

    @property
    def domain_hull(self):
        """
        Convex hull of the union of `domain`, i.e. of *all* measurement
        equations combined -- a single Polygon summarising the model's
        overall geographic footprint, e.g. to pass as `domain` to
        `buildMesh2d`. See `_domain_hull`.
        """
        return _domain_hull(self._domain)

    def _buildObservationGrid(self, df, formulas, verbose=True, tmin=None, tmax=None, domain=None):
        """
        `domain`, if given, is a list with one entry per formula (already
        validated by the caller, see `_checkDomain`), forwarded to the
        matching `DesignMatricesBuilder` so it drops observed sites outside
        it -- see `DesignMatricesBuilder._filter_domain`.
        """
        nvar = len(formulas)  # numer of the response variable

        domain_per_formula = domain if domain is not None else [None] * nvar

        builders = [
            DesignMatricesBuilder(df, f, dtype=self.dtype, verbose=verbose, tmin=tmin, tmax=tmax, domain=d)
            for f, d in zip(formulas, domain_per_formula)
        ]
        dfs = [b.build() for b in builders]

        T = [gr.T for gr in dfs]
        points = [gr.points for gr in dfs]

        # get dimnesion of each grid
        pdim = [grid.N for grid in dfs]
        block_p = np.hstack((0, np.cumsum(pdim)))
        ndim = block_p[-1]

        return nvar, points, dfs, ndim, pdim, block_p, T, builders

    def _log_warning_emstep(self, delta_lik, niter):
        msg = ""
        if delta_lik < 0:
            msg = "Warning: the log-likelihood decreased in iteration {} (delta_lik = {})".format(
                    niter, delta_lik
                )
        elif jnp.isnan(delta_lik):
            msg = "Warning: the log-likelihood is NaN in iteration {} (delta_lik = {})".format(
                    niter, delta_lik
                )
        elif jnp.isinf(delta_lik):
            msg = "Warning: the log-likelihood is Inf in iteration {} (delta_lik = {})".format(
                    niter, delta_lik
                )

        return msg
                

    def logger(self, stats, beta_decimals=2, scalar_decimals=2, relat_decimals=5):
        """
        Format one EM iteration's statistics (as built by `_log_iteration`)
        into a human-readable, multi-line block for console logging during
        `fit()`.

        Parameters
        ----------
        stats : dict
            One entry of the `nstats` history produced by `_log_iteration`
            (keys `niter`, `logL`, `deltaL`, `relatL`, `beta`, `s2e`, `f`,
            `ks`, `opt_success`, `A`, `x0`, `Sigma0`, `time_tot`,
            `tdelta_E`, `tdelta_M`).
        beta_decimals : int, default 2
            Decimal places used to format `stats["beta"]`.
        scalar_decimals : int, default 2
            Decimal places used to format the other numeric fields
            (log-likelihood, `s2e`, `f`, `ks`, `A`, `x0`, `Sigma0`, timings).
        relat_decimals : int, default 5
            Decimal places used to format the per-observation
            log-likelihood change `stats["relatL"]` (the quantity compared
            with `tol_relat`).

        Returns
        -------
        str
            The formatted, multi-line iteration summary.
        """

        # --- Identify and format scalars vs arrays ---
        def format_value(x, decimals=2):
            x_np = np.asarray(x)

            # Scalar case
            if x_np.ndim == 0:
                return f"{float(x_np):.{decimals}f}"

            # Array case → keep full array print
            return np.array2string(
                x_np,
                precision=decimals,
                separator=", ",
                floatmode="fixed",
                suppress_small=False,
            )

        msg = f"""
------------------------------------------------------------------
Iteration : {stats['niter']}
logL      : {format_value(stats['logL'], scalar_decimals)}
delta L   : {format_value(stats['deltaL'], scalar_decimals)}
relat L   : {format_value(stats['relatL'], relat_decimals)}
beta      : {format_value(stats['beta'], beta_decimals)}
s2e       : {format_value(stats['s2e'], scalar_decimals)}
f param   : {format_value(stats['f'], scalar_decimals)}
rescale   : {format_value(stats['ks'], scalar_decimals)} (Status: {stats['opt_success']})
A (flat)  : {format_value(np.asarray(stats['A']).flatten(), scalar_decimals)}
x0        : {format_value(stats['x0'], scalar_decimals)}
S0 diag   : {format_value(stats['Sigma0'], scalar_decimals)}
Run time  : Tot: {format_value(stats['time_tot'], scalar_decimals)}, Estep: {format_value(stats['tdelta_E'], scalar_decimals)}, Mstep: {format_value(stats['tdelta_M'], scalar_decimals)}
------------------------------------------------------------------
"""
        return msg

    def _log_iteration(
        self,
        history: list,
        niter: int,
        params: ModelParams,
        opt_success: bool,
        logL_cur,
        delta_par,
        delta_lik,
        relat_lik,
        tdelta_iter,
        tdelta_Edet,
        tdelta_Mdet,
    ):
        """
        Create iteration dictionary and append to history list.
        """

        it = {
            "niter": niter,
            "beta": params.beta.value,
            "s2e": params.s2e.value,
            "f": params.f.value,
            "ks": params.ks.value,
            "opt_success": opt_success,
            "A": params.A.value.flatten(),
            "x0": params.x0.value.mean(),
            "Sigma0": jnp.diag(params.Sigma0.value).mean(),
            "logL": logL_cur,
            "deltaP": delta_par,
            "deltaL": delta_lik,
            "relatL": relat_lik,
            "time_tot": tdelta_iter,
            "tdelta_E": tdelta_Edet.sum(),
            "tdelta_E_detail": tdelta_Edet,
            "tdelta_M": tdelta_Mdet.sum(),
            "tdelta_M_detail": tdelta_Mdet,
        }

        history.append(it)

        return history

    
    def generate_summary_header(self):
        
        # top-left / top-right small tables
        p = self.shape[0] if hasattr(self, "shape") else "N/A"
        q = self.shape[1] if hasattr(self, "shape") else "N/A"
        T = self.shape[2] if hasattr(self, "shape") else "N/A"

        if p is None:
            p = "N/A"
        if q is None:
            q = "N/A"
        if T is None:
            T = "N/A"

        # Header information for the summary table
        top_left = dict(
            [
                ("Model name:", lambda: [self.__class__.__name__]),
                (
                    "Model type (order):",
                    lambda: [f"{self.type if hasattr(self, 'type') else 'N/A'}, {self.order if hasattr(self, 'order') else 'N/A'}"],
                ),
                (
                    "Dep. Variables:",
                    lambda: [", ".join(self.y_name) if hasattr(self, "y_name") else "N/A"],
                ),
                ("Shape:", lambda: [f"(p = {p}, q = {q}, T = {T})"]),
            ]
        )

        top_right = dict(
            [
                ("Date:", lambda: [self._today]),
                (
                    "Rank",
                    lambda: (
                        [f"{q / p:.4f}"] if q != "N/A" and p != "N/A" and p > 0 else ["N/A"]
                    ),
                ),
                ("Model backend:", lambda: [f"{self.backend}, (dtype {self.dtype})"]),
                ("JAX default:", lambda: [f"{jax.default_backend()}"]),
            ]
        )

        # Generate the dictionaly
        gen_top_left = []
        for item in top_left.keys():
            gen_top_left.append((item, list(top_left[item]())))

        gen_top_right = []
        for item in top_right.keys():
            gen_top_right.append((item, list(top_right[item]())))

        len_empty = len(gen_top_left)- len(gen_top_right) 
        if len_empty > 0:
            gen_top_right = gen_top_right + [("", [""])] * len_empty
        elif len_empty < 0:
            gen_top_left = gen_top_left + [("", [""])] * (-len_empty)

        return gen_top_left, gen_top_right

    def generate_summary(self, print_short=False):
        """
        Build the left/right key-value rows used by `summary()`'s header
        table: model name/type/shape, then (if `print_full=True`) one
        "Grid ..." section per response variable's `DesignMatrices` and one
        "Latent. ..." section per latent covariance function, each via that
        object's own `generate_summary`.

        Parameters
        ----------
        print_full : bool, default True
            If `False`, only the top model-level rows are returned (used
            when this table is embedded elsewhere, e.g. by
            `LRStateSpaceResults.generate_summary`, without repeating the
            per-grid/per-covariance detail).

        Returns
        -------
        gen_top_left, gen_top_right : list of (str, list)
            Two lists of `(label, [value])` rows, of equal length.
        """


        gen_top_left, gen_top_right = self.generate_summary_header()
        if not print_short:
            return gen_top_left, gen_top_right

        else:  
            # Get the generate table from the gridlist
            if hasattr(self,"gridList") and self.gridList is not None:

                gen_top_left_grid = []
                gen_top_right_grid = []
                for grid in self.gridList:

                    left, righ = grid.generate_summary()

                    # check the length of the left and right tables and add empty rows if they are different
                    len_empty = len(left) - len(righ)
                    if len_empty > 0:
                        righ = righ + [("", [""])] * len_empty
                    elif len_empty < 0:
                        left = left + [("", [""])] * (-len_empty)

                    # Section divider: the label goes once on the left; the
                    # right side just carries a dash placeholder so it isn't
                    # repeated. `summary()` redraws this whole row as one
                    # continuous dashed line via `_continuous_dividers`.
                    yname = (
                        grid.y_design_info.column_names[0]
                        if getattr(grid, "y_design_info", None) is not None
                        else grid.y_name
                    )
                    left = [(f"Grid {yname}", ["-"])] + left
                    righ = [("-", ["-"])] + righ

                    gen_top_left_grid = gen_top_left_grid + left
                    gen_top_right_grid = gen_top_right_grid + righ

                gen_top_left = gen_top_left + gen_top_left_grid
                gen_top_right = gen_top_right + gen_top_right_grid

            
            # Get the generate table from the covariance
            if hasattr(self,"_cov_matern") and self._cov_matern is not None:

                gen_top_left_cov = []
                gen_top_right_cov = []
                for i, cov in enumerate(self._cov_matern):
                    left, righ = cov.generate_summary()

                    # check the length of the left and right tables and add empty rows if they are different
                    len_empty = len(left) - len(righ)
                    if len_empty > 0:
                        righ = righ + [("", [""])] * len_empty
                    elif len_empty < 0:
                        left = left + [("", [""])] * (-len_empty)

                    # See the Grid divider above: label once on the left,
                    # dash placeholder on the right, redrawn as one
                    # continuous line by `_continuous_dividers` in summary().
                    left = [(f"Latent. {i}", ["-"])] + left
                    righ = [("-", ["-"])] + righ

                    gen_top_left_cov = gen_top_left_cov + left
                    gen_top_right_cov = gen_top_right_cov + righ

                gen_top_left = gen_top_left + gen_top_left_cov
                gen_top_right = gen_top_right + gen_top_right_cov
            
            return gen_top_left, gen_top_right 

    def summary(self, print_short=False) -> Summary:
        """
        Return a `statsmodels`-style structured summary of the model
        (before fitting -- for the fitted results' own summary, see
        `LRStateSpaceResults.summary`).

        Parameters
        ----------
        print_full : bool, default True
            If `True`, includes one section per response variable's
            observation grid and one per latent covariance function (see
            `generate_summary`); if `False`, only the top model-level rows
            (name, type, shape) are included -- used by `fit()` to print a
            compact header before the EM iterations start.

        Returns
        -------
        statsmodels.iolib.summary.Summary
            Printable summary object (`str(...)`/`print(...)`).
        """
        # Generate the summary tables
        gen_top_left, gen_top_right = self.generate_summary(print_short=print_short)

        # Add the header to the summary
        smry = Summary()
        smry.add_table_2cols(
            _summary_table_source(),
            title="LR State Space Model",
            gleft=gen_top_left,
            gright=gen_top_right,
            yname= self.yname if self.yname is not None else "None",
            xname= self.xbeta_names if self.xbeta_names is not None else "None",
        )

        # Redraw the "Grid .../Latent. ..." divider rows as one continuous
        # dashed line (see `_continuous_dividers`). Divider rows are
        # identified by their left-hand label, i.e. every row whose value
        # is the placeholder ["-"] set above.
        divider_labels = [stub for stub, val in gen_top_left if val == ["-"]]
        if divider_labels:
            smry.as_text = lambda _orig=smry.as_text, labels=divider_labels: (
                _continuous_dividers(_orig(), labels)
            )

        return smry

    def __str__(self):
        """Return a human-readable summary of the model (see `summary`)."""
        return str(self.summary(print_short=False))

    def _get_summary_table(self):
        gen_top_left, gen_top_right = self.generate_summary_header()
        items = [
            (key, value)
            for key, value in gen_top_left + gen_top_right
            if str(key).strip()
        ]
        return f"{self.__class__.__name__}\n" + format_info_table(items, indent=2)

    def __repr__(self):
        """Cheap, side-effect-free representation for interactive use
        (model name/type/shape/backend only, from `generate_summary_header`)
        -- see `summary()`/`str()` for the full, more expensive report
        (per-grid/per-covariance detail). Does not touch `self.model` or
        any other state, unlike `summary()`.
        """
        return self._get_summary_table()

    def _is_verbose(self, verbose=None) -> bool:
        """Resolve an optional per-call `verbose` override against `self.verbose`."""
        return self.verbose if verbose is None else verbose

    def _log(self, msg: str, verbose=None) -> None:
        """Print `msg` via `print_info` if verbose (see `_is_verbose`)."""
        if self._is_verbose(verbose):
            self.print_info(msg)

    def print_info(self, msg):
        """Print `msg` prefixed with a UTC timestamp (see `geossm.utils._print_info`)."""
        _print_info(msg)
