"""
Data preparation module for GEOSSM.

This module provides functionality to prepare spatial-temporal datasets
for modeling, including design matrix construction and dataset validation.

DesignMatrices class encapsulates the design matrices and related metadata.

DesignMatricesBuilder class handles the preparation of the DesignMatrices object.

"""

from contextlib import contextmanager
from datetime import datetime, timezone
import ast
import numpy as np
import shapely
from scipy.spatial.distance import cdist
from patsy import ModelDesc, NAAction, build_design_matrices, dmatrices, dmatrix
import geopandas as geopd
import pandas as pd
from shapely.geometry import Point

from dataclasses import dataclass, field

from statsmodels.iolib.summary import Summary


try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def _ensure_x64_for_dtype(dtype):
    """Enable JAX's x64 mode if 64-bit precision was explicitly requested.

    JAX silently truncates float64 arrays back to 32-bit unless x64 mode is
    enabled. Mirrors geossm.ssm.statespace._ensure_x64_for_dtype; duplicated
    here (rather than imported) to avoid a circular import with geossm.ssm
    during package initialization.
    """
    if JAX_AVAILABLE and np.dtype(dtype).itemsize == 8 and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


class _NAPassthrough(NAAction):
    """Tell Patsy not to treat anything (NaN included) as missing.

    Missing response values must survive formula evaluation as NaN rows
    rather than being dropped, since the [N x T] reshape downstream
    requires one row per (site, timestamp) regardless of whether the
    response was observed.
    """

    def __init__(self):
        super().__init__(NA_types=[])


def _extract_formula_metadata(termlist):
    """
    Extract variable names, transformations, and underlying dataframe columns
    from a Patsy term list.

    Returns
    -------
    names : list[str]
        Original term expressions.

    transformations : list[str]
        Transformation applied to each term.

    columns : list[str]
        Underlying dataframe column(s) used by each term.
    """

    def extract_columns(expr):
        """
        Extract dataframe column names from a Patsy expression.

        Examples
        --------
        np.sqrt(np.abs(AQ_pm10)) -> ['AQ_pm10']
        I(t2m**2)               -> ['t2m']
        t2m                     -> ['t2m']
        """

        if expr.startswith("I(") and expr.endswith(")"):
            expr = expr[2:-1]

        tree = ast.parse(expr, mode="eval")

        columns = []

        class ColumnVisitor(ast.NodeVisitor):

            def visit_Name(self, node):
                columns.append(node.id)

            def visit_Call(self, node):
                # Do not descend into node.func: for e.g. standardize(x) that
                # is the Name "standardize" itself, not a dataframe column.
                for arg in node.args:
                    self.visit(arg)
                for kw in node.keywords:
                    self.visit(kw.value)

        ColumnVisitor().visit(tree)

        blacklist = {
            "np",
            "pd",
            "math",
        }

        return sorted(set(c for c in columns if c not in blacklist))

    def extract_functions(expr):
        """Extract nested function chain from an expression."""

        if expr.startswith("I(") and expr.endswith(")"):
            return [expr]

        tree = ast.parse(expr, mode="eval")

        funcs = []

        class FunctionVisitor(ast.NodeVisitor):

            def visit_Call(self, node):

                if isinstance(node.func, ast.Attribute):
                    funcs.append(ast.unparse(node.func))

                elif isinstance(node.func, ast.Name):
                    funcs.append(node.func.id)

                self.generic_visit(node)

        FunctionVisitor().visit(tree)

        return funcs

    names = []
    transformations = []
    columns = []

    for term in termlist:

        # Intercept
        if len(term.factors) == 0:
            names.append("Intercept")
            transformations.append("1")
            columns.append("Intercept")
            continue

        factor_exprs = [
            str(getattr(factor, "code", factor.name())).strip()
            for factor in term.factors
        ]

        # Interaction terms
        if len(factor_exprs) > 1:

            interaction = ":".join(factor_exprs)

            names.append(interaction)
            transformations.append(interaction)

            interaction_cols = []

            for expr in factor_exprs:
                interaction_cols.extend(extract_columns(expr))

            columns.append(":".join(sorted(set(interaction_cols))))

        else:

            expr = factor_exprs[0]

            names.append(expr)

            # Plain variable
            if "(" not in expr:

                transformations.append("1")
                columns.append(expr)

            # I(...)
            elif expr.startswith("I("):

                transformations.append(expr)

                cols = extract_columns(expr)

                columns.append(":".join(cols))

            # Generic function(s)
            else:

                funcs = extract_functions(expr)

                transformations.append(":".join(funcs))

                cols = extract_columns(expr)

                columns.append(":".join(cols))

    return names, transformations, columns


def check_regular_timestamps(timestamps):
    """
    Check whether a collection of timestamps is regularly spaced.

    The unique timestamps are sorted and the gaps between consecutive
    values are compared. A single fixed gap (in seconds down to
    microseconds, or in whole days) is reported as a ``(delta, unit)``
    pair; failing that, a calendar-aware frequency (month or year, which
    do not have a constant number of seconds) is tried via
    :func:`pandas.infer_freq`.

    Parameters
    ----------
    timestamps : array-like
        Sequence of timestamp-like values (e.g. strings, `datetime`
        objects, or a :class:`pandas.Series`/column) to check. Values are
        parsed with :func:`pandas.to_datetime`.

    Returns
    -------
    is_regular : bool
        ``True`` if the (unique) timestamps are evenly spaced, ``False``
        if parsing failed or the spacing is irregular.
    delta : int, float or None
        The size of the regular step, in the unit given by `unit`. ``0``
        if there is only one unique timestamp, ``None`` if `is_regular`
        is ``False``.
    unit : str or None
        One of ``"None"`` (single timestamp), ``"month"``, ``"year"``,
        ``"day"``, ``"hour"``, ``"minute"``, ``"second"``,
        ``"millisecond"`` or ``"microsecond"``. ``None`` if `is_regular`
        is ``False``.

    Notes
    -----
    - Uses UNIQUE timestamps (important for panel data with repeated times per geometry).
    - Handles parsing errors robustly: unparsable values make the function
      return ``(False, None, None)`` rather than raising.
    - Supports fixed deltas (sec/min/hour/day) and calendar frequencies
      (month/year) when inferable.
    """

    # 1) Parse safely
    ts = pd.Series(timestamps)
    ts = pd.to_datetime(ts, errors="coerce", utc=True)

    if ts.isna().any():
        return False, None, None

    # 2) Use unique sorted times (avoid duplicated rows across geometries)
    ts_unique = pd.DatetimeIndex(ts.drop_duplicates().sort_values())

    if ts_unique.size == 1:
        return True, 0, "None"

    if ts_unique.size < 2:
        return False, None, None

    # 3) Try calendar-aware frequency first (works for month/year)
    inferred = pd.infer_freq(ts_unique) if ts_unique.size >= 3 else None
    if inferred is not None:
        # Normalize aliases like '2MS', 'M', 'YS', etc.
        n = ""
        code = inferred
        i = 0
        while i < len(code) and code[i].isdigit():
            n += code[i]
            i += 1
        mult = int(n) if n else 1
        base = code[i:]

        # Month-like frequencies
        if base in {"M", "MS", "BM", "BMS", "ME"}:
            return True, mult, "month"

        # Year-like frequencies
        if base in {"A", "AS", "Y", "YS", "YE", "BA", "BAS"}:
            return True, mult, "year"

    # 4) Fallback: fixed timedeltas
    diffs = pd.Series(ts_unique).diff().dropna()
    if diffs.empty:
        return False, None, None

    first = diffs.iloc[0]
    if not (diffs == first).all():
        return False, None, None

    seconds = first.total_seconds()
    if seconds <= 0:
        return False, None, None

    if seconds % 86400 == 0:
        return True, int(seconds // 86400), "day"
    if seconds % 3600 == 0:
        return True, int(seconds // 3600), "hour"
    if seconds % 60 == 0:
        return True, int(seconds // 60), "minute"
    if float(seconds).is_integer():
        return True, int(seconds), "second"

    # Sub-second support
    milliseconds = seconds * 1000
    if float(milliseconds).is_integer():
        return True, int(milliseconds), "millisecond"

    microseconds = seconds * 1_000_000
    if float(microseconds).is_integer():
        return True, int(microseconds), "microsecond"

    # Last fallback
    return True, float(seconds), "second"


@dataclass
class _FormulaInfo:
    """Parsed metadata for one Patsy formula, produced by `_check_formula`."""

    lhs_termlist: list
    rhs_termlist: list
    response_name: str = None
    response_expr: list = None
    response_column: str = None
    covariate_names: list = field(default_factory=list)
    covariate_exprs: list = field(default_factory=list)
    covariate_columns: list = field(default_factory=list)


@dataclass
class DesignMatrices:
    """
    Container for the response and covariate design matrices of a
    spatio-temporal dataset, organized on an [N x T] (site x time) grid,
    together with the spatial/temporal/formula metadata needed to
    interpret and reproduce them.

    Instances are normally produced by :class:`DesignMatricesBuilder`
    (via :meth:`DesignMatricesBuilder.build` or
    :meth:`DesignMatricesBuilder.build_predict`), not constructed
    directly; the constructor's validation (see `__post_init__`) exists
    mainly to keep the builder's output self-consistent.

    Parameters
    ----------
    X : numpy.ndarray
        Covariate design array of shape ``[N, P, T]`` (sites x covariates
        x timesteps), built from the right-hand side of `formula`.
    X_design_info : patsy.design_info.DesignInfo
        Patsy metadata describing how `X`'s columns were built from
        `formula` (term names, factor info, etc.). Reused by
        :meth:`DesignMatricesBuilder.build_predict` to evaluate
        prediction covariates with the same (e.g. `standardize`)
        statistics as the training data.
    x_names : list of str
        Column names of `X` (one per covariate), taken from
        `X_design_info.column_names`.
    x_exprs : list of str
        Transformation/expression applied to each covariate term (e.g.
        ``"1"`` for a plain variable, or the full expression for
        ``I(...)``/function terms), as extracted from `formula`.
    points : list of shapely.geometry.Point or numpy.ndarray
        The N unique site locations, in the CRS given by `crs`. Passed in
        as `Point` geometries; converted in `__post_init__` to an
        ``[N, 2]`` coordinate array of dtype `dtype`.
    formula : str
        The Patsy/R-style model formula (e.g.
        ``"AQ_pm10 ~ 1 + WE_temp_2m"``) that `X` (and `y`, if present)
        were built from.
    crs : pyproj.CRS
        Coordinate reference system of `points`, taken from the source
        GeoDataFrame.
    box : list of float
        Bounding box ``[minx, miny, maxx, maxy]`` of the source
        GeoDataFrame's geometries.
    geometry : array-like of str
        Unique geometry type(s) found in the source GeoDataFrame (e.g.
        ``["Point"]``).
    timestamps : numpy.ndarray
        The T unique, sorted timestamps that make up the time axis of `X`
        (and `y`, if present).
    delta : int or float
        Size of the regular time step between consecutive `timestamps`,
        in the unit given by `unit` (see `check_regular_timestamps`).
    unit : str
        Unit of `delta` (e.g. ``"day"``, ``"hour"``, ``"month"``).
    y : numpy.ndarray, optional
        Response array of shape ``[N, T]``, built from the left-hand side
        of `formula`. ``None`` if `formula` has no response (covariates
        only). Missing observations are ``NaN`` rather than dropped, so
        the site x time grid stays complete.
    y_design_info : patsy.design_info.DesignInfo, optional
        Patsy metadata for `y`, analogous to `X_design_info`. ``None``
        when `y` is ``None``.
    y_name : str, optional
        Column name of the response, taken from
        `y_design_info.column_names`. ``None`` when `y` is ``None``.
    y_expr : list of str, optional
        Transformation/expression applied to the response term, as
        extracted from `formula`.
    time_col_name : str, default "Time"
        Name of the column in the source GeoDataFrame that held the
        timestamps.
    dtype : numpy.dtype, default numpy.float32
        Numeric precision used for `X`, `y` and `points`. Passing a
        64-bit dtype (e.g. `numpy.float64`) also enables JAX's x64 mode,
        since JAX otherwise silently truncates float64 arrays to 32-bit.

    Attributes
    ----------
    N : int
        Number of sites (derived from `X.shape[0]`).
    b : int
        Number of covariates/columns in `X` (derived from `X.shape[1]`).
    T : int
        Number of timesteps (derived from `X.shape[2]`).
    terms : patsy.desc.ModelDesc
        Parsed representation of `formula`, computed in
        `__post_init__`.

    Notes
    -----
    ``geometry_id`` (a further, non-constructor field of this dataclass)
    holds the name used internally for the per-site identifier column
    (``"geometry_id"``).

    Raises
    ------
    TypeError
        If `dtype` is not a valid NumPy dtype.
    ValueError
        If the shapes of `y`, `X` and `timestamps` are inconsistent (see
        `__post_init__`).
    """

    X: np.ndarray
    X_design_info: object
    x_names: list[str]
    x_exprs: list[str]
    points: list[Point]
    formula: str
    crs: object
    box: object
    geometry: object
    timestamps: np.ndarray
    delta: int
    unit: str
    y: np.ndarray = None
    y_design_info: object = None
    y_name: str = None
    y_expr: list[str] = None
    time_col_name: str = "Time"
    geometry_id: str = field(default="geometry_id", init=False)
    dtype: np.dtype = field(default=np.float32)
    # backend: Literal["numpy", "jax"] = field(default="numpy")

    def __post_init__(self):
        """Validate and normalize the fields set by the dataclass constructor.

        Coerces `dtype`, casts `X`/`y`/`points` to it, validates array
        shapes, and derives `N`, `b`, `T` and `terms`.
        """
        # Validate backend
        # if self.backend == "jax" and not JAX_AVAILABLE:
        #     raise ImportError(
        #         "JAX is not installed. Install it with `pip install jax` "
        #         "or use backend='numpy'."
        #     )
        # if self.backend not in ("numpy", "jax"):
        #     raise ValueError(f"backend must be 'numpy' or 'jax', got '{self.backend}'")

        # Validate dtype
        try:
            self.dtype = np.dtype(self.dtype)
        except TypeError:
            raise TypeError(f"Invalid dtype: {self.dtype}")
        _ensure_x64_for_dtype(self.dtype)

        # Cast arrays to dtype
        self.y = np.asarray(self.y, dtype=self.dtype) if self.y is not None else None
        self.X = np.asarray(self.X, dtype=self.dtype)
        self.timestamps = np.asarray(self.timestamps)
        self.type = np.unique(self.geometry).astype(str)
        # shapely.get_coordinates is a vectorized (C-level) ufunc -- ~35x
        # faster than a per-point Python loop at N in the millions.
        self.points = shapely.get_coordinates(np.asarray(self.points)).astype(self.dtype)

        # Validate shapes
        if self.y is not None and self.y.ndim != 2:
            raise ValueError(f"y must be [N x T], got shape {self.y.shape}")
        if self.X.ndim != 3:
            raise ValueError(f"X must be [N x P x T], got shape {self.X.shape}")
        if self.y is not None and self.y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"N mismatch: y has {self.y.shape[0]} sites, "
                f"X has {self.X.shape[0]} sites"
            )
        if self.y is not None and self.y.shape[1] != self.X.shape[2]:
            raise ValueError(
                f"T mismatch: y has {self.y.shape[1]} timesteps, "
                f"X has {self.X.shape[2]} timesteps"
            )
        if self.y is not None and self.timestamps.shape[0] != self.y.shape[1]:
            raise ValueError(
                f"timestamps length {self.timestamps.shape[0]} does not match "
                f"T={self.y.shape[1]}"
            )

        # Derived dimensions

        self.N, self.b, self.T = self.X.shape
        self.terms = ModelDesc.from_formula(self.formula)

        # Convert to JAX arrays if requested
        # if self.backend == "jax":
        self.y = jnp.array(self.y) if self.y is not None else None
        self.X = jnp.array(self.X)
            # Only arrays of numeric types are supported by JAX.
            # self.timestamps = jnp.array(self.timestamps)

    def astype(self, dtype) -> "DesignMatrices":
        """
        Cast `X`, `y` and `points` to a new numeric dtype, in place.

        Lets an already-built DesignMatrices be realigned with a model's
        dtype (e.g. LRStateSpaceModel.dtype) without re-running formula
        parsing / matrix construction, which is comparatively expensive.

        Parameters
        ----------
        dtype : numpy.dtype or str
            Target numeric precision (e.g. `numpy.float32`,
            `numpy.float64`). If it is 64-bit, JAX's x64 mode is enabled
            (see `dtype` on the class), since JAX otherwise silently
            truncates float64 arrays to 32-bit.

        Returns
        -------
        DesignMatrices
            `self`, with `X`, `y`, `points` and `dtype` updated. Returned
            unchanged (no copy/cast) if `dtype` already matches
            `self.dtype`.
        """
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self

        _ensure_x64_for_dtype(dtype)

        self.dtype = dtype
        self.X = jnp.asarray(self.X, dtype=dtype)
        self.y = jnp.asarray(self.y, dtype=dtype) if self.y is not None else None
        self.points = np.asarray(self.points, dtype=dtype)

        return self

    @property
    def isnan(self) -> np.ndarray:
        """Boolean mask [N x T] — True where y is NaN."""
        y_np = np.asarray(self.y)
        return np.isnan(y_np)

    @property
    def n_obs(self) -> int:
        """Total number of observed (non-NaN) values in y."""
        return int((~self.isnan).sum())

    @property
    def nan_ratio(self) -> float:
        """Fraction of missing (NaN) values over all [N x T] entries."""
        return 1 - (self.n_obs / (self.N * self.T))

    @property
    def shape(self):
        """Shape of the design matrices as a tuple (N, P, T)."""
        return (self.N, self.b, self.T)

    def generate_summary(self,):
        """
        Build the two side-by-side tables of key/value pairs shown by
        `summary`.

        Returns
        -------
        gen_top_left : list of (str, list[str])
            Left-hand column: formula, response/covariate names and
            transformations, observed/missing counts, and `dtype`.
        gen_top_right : list of (str, list[str])
            Right-hand column: array dimensions (N, b, T), CRS, geometry
            type, bounding box, timestamp range, and (`delta`, `unit`).
        """
        y_np = np.asarray(self.y) if self.y is not None else None

        def get_print_string(str):
            return str if len(str) <= 20 else str[:25] + "..."

        top_left = dict([
            ("Formula",        lambda: [get_print_string(self.formula)]),
            ("Response (y)",   lambda: [self.y_design_info.column_names[0] if self.y_design_info else self.y_name] if self.y is not None else ["N/A"]),
            ("Covariates (X)", lambda: [get_print_string(", ".join(self.x_names))] if self.x_names else ["N/A"]),
            ("[min, max, mean]", lambda: [f"[{np.nanmin(y_np):.4g}, {np.nanmax(y_np):.4g}, {np.nanmean(y_np):.4g}]"] if self.y is not None else ["N/A"]),
            ("Observed",         lambda: [f"{self.n_obs} / {self.N * self.T} ({(1 - self.nan_ratio) * 100:.1f}%)"] if self.y is not None else ["N/A"]),
            ("Missing",          lambda: [f"{int(np.isnan(y_np).sum())} ({self.nan_ratio * 100:.1f}%)"] if self.y is not None else ["N/A"]),
            ("Transformation (y)",       lambda: [get_print_string(", ".join(self.y_expr) if self.y_expr else "No")] if self.y is not None else ["N/A"]),
            ("Transformation (X)",       lambda: [get_print_string(", ".join(self.x_exprs) if self.x_exprs else "No")]),
            ("dtype",          lambda: [str(self.dtype)]),
        ])

        top_right = dict([
            ("Sites    [N]",     lambda: [str(self.N)]),
            ("Covariates [b]",   lambda: [str(self.b)]),
            ("Timesteps  [T]",   lambda: [str(self.T)]),
            ("CRS",             lambda: [str(self.crs.name)]),
            ("Units",           lambda: [", ".join([axis.unit_name for axis in self.crs.coordinate_system.axis_list])]),
            ("Geometry type",   lambda: [", ".join(self.type)]),
            ("Box",             lambda: [f"{self.box}"]),
            ("Timestamps",      lambda: [f'{pd.Timestamp(self.timestamps.min()).strftime("%Y-%m-%d")} to {pd.Timestamp(self.timestamps.max()).strftime("%Y-%m-%d")}']),
            ("Delta, Unit",       lambda: [f"{self.delta}, {self.unit}"]),
        ])

        # Generate the dictionaly
        gen_top_left = []
        for item in top_left.keys():
            gen_top_left.append((item, list(top_left[item]())))

        gen_top_right = []
        for item in top_right.keys():
            gen_top_right.append((item, list(top_right[item]())))

        return gen_top_left, gen_top_right

    def summary(self) -> Summary:
        """
        Build a human-readable summary of the design matrices.

        Assembles the key/value tables from `generate_summary` into a
        :class:`statsmodels.iolib.summary.Summary`, in the same style as
        a statsmodels model-fit summary. Also used by `__str__`.

        Notes
        -----
        As a side effect, sets placeholder `model`/`params`/`bse`/
        `tvalues`/`pvalues` attributes on `self`, required by
        `Summary.add_table_2cols`; these are not meaningful statistics
        (`DesignMatrices` holds no fitted model).

        Returns
        -------
        statsmodels.iolib.summary.Summary
            Two-column summary table describing the formula, response
            and covariates, missingness, array shapes, CRS and time
            range.
        """

        smry = Summary()

        self.model = None
        self.params = np.zeros(1)
        self.param_names = "Placeholder" #self.xbeta_names
        self.bse = np.zeros(len(self.params))
        self.tvalues = np.zeros(len(self.params))
        self.pvalues = np.zeros(len(self.params))

        gen_top_left, gen_top_right = self.generate_summary()

        smry.add_table_2cols(
            self,
            title="Design Matrices Summary",
            gleft=gen_top_left,
            gright=gen_top_right,
            yname=None,
            xname=None,
        )

        return smry

    def __str__(self):
        """Return the text rendering of `summary`."""
        return self.summary().as_text()

    def __repr__(self) -> str:
        """Return a short ``DesignMatrices(N=..., b=..., T=..., dtype=...)`` repr."""
        return (
            f"DesignMatrices("
            f"N={self.N}, b={self.b}, T={self.T}, "
            f"dtype={self.dtype}')"
        )


class DesignMatricesBuilder:
    """
    Turn a spatio-temporal :class:`geopandas.GeoDataFrame` plus a
    Patsy/R-style model `formula` into a :class:`DesignMatrices` object
    (response + covariates organized on an [N x T] site x time grid),
    ready to be consumed by geossm's state-space models.

    The constructor parses and validates `formula` against `geodf`
    (spatial CRS/geometry checks, time column detection, regular-spacing
    check, optional time-window/domain filtering) but does not itself
    build the arrays; call `build` (or `__call__`) to actually construct
    the :class:`DesignMatrices`. Once built, the same builder can be
    reused via `build_predict` to construct design matrices for new
    (prediction) locations/times using the training formula's fitted
    state (e.g. `standardize(...)` statistics).

    Examples
    --------
    >>> builder = DesignMatricesBuilder(geodf, "AQ_pm10 ~ 1 + WE_temp_2m")
    >>> dm = builder.build()
    >>> dm_new = builder.build_predict(new_geodf)
    """

    def __init__(self, geodf: geopd.GeoDataFrame, formula: str, dtype=np.float32, verbose: bool = True,
                 tmin: datetime = None, tmax: datetime = None, domain=None):
        """
        Prepare the spatial-temporal dataset for modeling.

        Runs all validation/coercion steps on `geodf` (spatial checks,
        time column detection and coercion, formula-column existence,
        time-regularity check, numeric dtype casting, and optional
        time-window/domain filtering) so that `build` can construct the
        design matrices directly. Does not build the design matrices
        itself.

        Parameters
        ----------
        geodf : geopandas.GeoDataFrame
            Spatio-temporal dataset in long format: one row per
            (site, timestamp) observation, with a ``geometry`` column
            (a single geometry type, e.g. all `Point`) and a CRS set. A
            time column named ``"Time"`` is used if present, otherwise
            the first datetime64 column found. All rows for a given
            site must share exactly one geometry per site, and after any
            `tmin`/`tmax`/`domain` filtering the dataset must form a
            complete site x time grid (every site has one row per
            timestamp; the response may be `NaN`).
        formula : str
            Patsy/R-style model formula, e.g.
            ``"AQ_pm10 ~ 1 + WE_temp_2m"``. The left-hand side (if any)
            names the response variable; the right-hand side lists the
            covariates (including transformations such as
            ``np.sqrt(...)``, ``I(...)`` or ``standardize(...)``). A
            formula with no left-hand side (e.g. ``"~ 1 + WE_temp_2m"``)
            is allowed and builds covariates only (`y` is `None`).
            Every column referenced by `formula` must exist in `geodf`.
        dtype : numpy.dtype, default numpy.float32
            Numeric precision used when casting `geodf`'s numeric columns
            and when building the resulting `DesignMatrices` (`X`, `y`,
            `points`). Passing a 64-bit dtype also enables JAX's x64
            mode, which is otherwise off by default.
        verbose : bool, default True
            If `True`, log progress (spatial/time checks, formula
            parsing, etc.) to stdout via `print_info`.
        tmin : datetime, optional
            Lower bound (inclusive) of the time window to keep. Rows
            with a timestamp earlier than `tmin` are dropped before the
            design matrices are built.
        tmax : datetime, optional
            Upper bound (inclusive) of the time window to keep. Rows
            with a timestamp later than `tmax` are dropped before the
            design matrices are built.
        domain : shapely Polygon/MultiPolygon, optional
            The scientific interest domain, in the same CRS as `geodf`. When
            given, every site (unique geometry) outside `domain` is dropped
            -- together with all of its timestamps -- before the design
            matrices are built, so training never sees observations outside
            the domain. Contrast with `build_predict`'s own `domain`
            argument, which only flags out-of-domain points (`self.mask`)
            rather than dropping them.

        Raises
        ------
        ValueError
            If `formula` is empty/invalid, references columns missing
            from `geodf`, or if `geodf` fails a spatial/time/domain
            check (e.g. not a GeoDataFrame, missing/invalid geometry,
            mixed geometry types, no usable time column, irregularly
            spaced timestamps, or an incomplete site x time grid).
        """
        self.verbose = verbose
        self.dtype = np.dtype(dtype)
        self._log("Initializing DesignMatricesBuilder")

        self.formula = formula
        self.tmin = tmin
        self.tmax = tmax
        self.domain = domain

        self.formula_info = self._check_formula(formula)
        if self.formula_info.response_name:
            self._log(f"Formula parsed successfully. Response variable: '{self.formula_info.response_name}'")
        else:
            self._log("Formula parsed successfully. No response variable found")

        (
            self.geodf, self.geometry_id, self.time_col_name,
            self.crs, self.box, self.geometry, self.delta, self.unit,
        ) = self._prepare_geodf(geodf, tmin, tmax, domain=domain)
        self._log(f"Spatial check passed using geometry id column '{self.geometry_id}'")
        self._log(f"Time column detected: '{self.time_col_name}'")
        self._log(f"Time consistency check passed: delta {self.delta}, unit {self.unit}")

    def _prepare_geodf(self, geodf, tmin, tmax, prediction = False, domain=None):
        """
        Run every check/coercion a GeoDataFrame needs before it can be
        turned into design matrices. Shared by `__init__` (on the training
        data) and `build_predict` (on new data), so both paths validate
        identically and can't drift apart.

        `domain`, when given, drops out-of-domain sites here (see
        `_filter_domain`); `build_predict` never passes it through (it uses
        the separate, non-destructive `_check_domain`/`self.mask` instead),
        so this only ever filters training data.
        """
        if not isinstance(geodf, geopd.GeoDataFrame):
            raise ValueError("Input dataset must be a GeoDataFrame")

        geodf = geodf.copy()
        crs = geodf.crs
        box = np.round(geodf.total_bounds, 3).tolist()  # [minx, miny, maxx, maxy]

        geometry_id = self._check_spatial_dataset(geodf)
        geometry = pd.unique(geodf.geom_type)

        time_col_name = self._check_time_dataset(geodf)
        self._coerce_time_column(geodf, time_col_name)

        self._check_formula_columns_exist(geodf, prediction=prediction)

        delta, unit = self._check_time_regularity(geodf, time_col_name)

        self._cast_numeric_columns(geodf, geometry_id)

        # Cut the dataset to the specified time range if tmin and tmax are provided
        if tmin is not None or tmax is not None:
            geodf = self._filter_time_range(geodf, time_col_name, tmin, tmax)

        if domain is not None:
            geodf = self._filter_domain(geodf, geometry_id, domain)

        return geodf, geometry_id, time_col_name, crs, box, geometry, delta, unit

    def _filter_domain(self, geodf, geometry_id, domain):
        """
        Drop every row belonging to a site (`geometry_id`) whose location
        falls outside `domain` -- a whole site (every timestamp), not
        individual rows, so the remaining data stays a complete site x time
        grid for `_check_balanced_panel`. See `_check_domain` for the
        counterpart used by `build_predict`, which only flags out-of-domain
        points via `self.mask` instead of dropping them.
        """
        sites = geodf.drop_duplicates(subset=[geometry_id])

        shapely.prepare(domain)
        inside = shapely.covers(domain, np.asarray(sites.geometry))
        outside_ids = sites.loc[~inside, geometry_id]

        n_out = len(outside_ids)
        n_total = len(sites)
        if n_out:
            self._log(
                f"Domain check: dropping {n_out} of {n_total} site(s) outside "
                "the scientific interest domain"
            )
            geodf = geodf[~geodf[geometry_id].isin(outside_ids)]
        else:
            self._log(f"Domain check: all {n_total} site(s) inside the domain")

        if geodf.empty:
            raise ValueError("No sites remain after filtering to the scientific interest domain")

        return geodf

    def _log(self, msg: str) -> None:
        """Print `msg` via `print_info` if `self.verbose` is True."""
        if self.verbose:
            self.print_info(msg)

    @contextmanager
    def _verbosity(self, verbose):
        """Temporarily override `self.verbose` for the duration of a public call."""
        if verbose is None:
            yield
            return
        previous = self.verbose
        self.verbose = verbose
        try:
            yield
        finally:
            self.verbose = previous

    def build(self, verbose=None) -> "DesignMatrices":
        """
        Build the design matrices from the GeoDataFrame passed to
        `__init__`, using the already-parsed `formula`.

        After building, the underlying GeoDataFrame is discarded (only
        the resulting `DesignMatrices` and the fitted `X_design_info`
        are kept), so a builder kept around for `build_predict` does not
        hold a copy of the training data.

        Parameters
        ----------
        verbose : bool, optional
            Temporarily override `self.verbose` for this call only. If
            `None` (default), `self.verbose` (set in `__init__`) is used.

        Returns
        -------
        DesignMatrices
            The response (if any) and covariate design matrices, plus
            spatial/temporal metadata, built from the training data.
        """
        with self._verbosity(verbose):
            self._log("Building design matrices from GeoDataFrame")
            self.design_matrices = self._build_geodataframe(
                self.formula_info.lhs_termlist, self.formula_info.rhs_termlist
            )
            self._log("Design matrices built successfully")
        # self.geodf is no longer needed once the matrices are built -- free
        # it so a builder kept around (e.g. for build_predict) only holds
        # the small formula/design_info state, not a copy of the training data.
        del self.geodf
        return self.design_matrices

    def build_predict(self, df: geopd.GeoDataFrame, verbose=None, domain=None) -> "DesignMatrices":
        """
        Build the design matrix for new (prediction) locations/times, reusing
        this builder's fitted `X_design_info` so stateful transforms (e.g.
        `standardize(...)`) are evaluated with the training mean/std instead
        of being recomputed on `df`. `build()` must be called first.

        `df` is validated the same way as the training GeoDataFrame (spatial
        checks, time column detection, regular-spacing check, formula-column
        existence -- but no response column is required), and is additionally
        checked for consistency with the training data's CRS and time step,
        and clipped to the training data's observed time range.

        Parameters
        ----------
        df : geopandas.GeoDataFrame
            New (prediction) spatio-temporal dataset, in the same format,
            CRS and time step as the GeoDataFrame `build` was trained on.
            It only needs to supply the covariate columns referenced by
            `formula`, not the response.
        verbose : bool, optional
            Temporarily override `self.verbose` for this call only. If
            `None` (default), `self.verbose` (set in `__init__`) is used.
        domain : shapely Polygon/MultiPolygon, optional
            The scientific interest domain, in the same CRS as `df`. When
            given, every prediction point is checked against it and the
            result is stored as the boolean array `self.mask` (True where
            the point falls inside `domain`), aligned with `points`/the N
            axis of the returned `DesignMatrices`. Points outside `domain`
            are *not* dropped, only flagged -- filtering is left to the
            caller. `self.mask` is left `None` when `domain` is not given.

        Returns
        -------
        DesignMatrices
            Covariate-only design matrices (``y`` is `None`) for `df`,
            built with the training `X_design_info`.

        Raises
        ------
        ValueError
            If `build()` has not been called yet, if `df` fails the same
            validation as the training data, or if `df`'s CRS or time
            step does not match the training data's.
        """
        if not hasattr(self, "design_matrices"):
            raise ValueError("build() must be called before build_predict()")

        with self._verbosity(verbose):
            self._log("Building prediction design matrices from GeoDataFrame")
            # Cut `df` to the time range actually observed in the training
            # build, not the raw tmin/tmax passed to the constructor: the
            # observed range already accounts for any such filtering plus
            # whatever timestamps the training data actually had.
            train_tmin = self.design_matrices.timestamps.min()
            train_tmax = self.design_matrices.timestamps.max()
            geodf, geometry_id, time_col_name, crs, box, geometry, delta, unit = (
                self._prepare_geodf(df, train_tmin, train_tmax, prediction=True)
            )
            self._check_consistent_with_training(crs, delta, unit)
            self._log(f"Spatial check passed using geometry id column '{geometry_id}'")
            self._log(f"Time column detected: '{time_col_name}'")
            self._log(f"Time consistency check passed: delta {delta}, unit {unit}")

            predict_design_matrices = self._compute_predict_design_matrix(
                geodf, geometry_id, time_col_name, crs, box, geometry, delta, unit,
            )
            self._check_domain(predict_design_matrices, domain)
            self._log("Prediction design matrices built successfully")
        return predict_design_matrices

    def _check_consistent_with_training(self, crs, delta, unit) -> None:
        """
        `build()` must have run first, so `self.crs`/`self.delta`/`self.unit`
        already hold the training values -- a mismatch here means the
        prediction covariates would silently be sampled on a different grid
        or in different units than the model was fit on.
        """
        problems = []
        if crs != self.crs:
            problems.append(f"CRS ({crs}) does not match the training CRS ({self.crs})")
        if (delta, unit) != (self.delta, self.unit):
            problems.append(
                f"time step (delta={delta}, unit={unit}) does not match the "
                f"training time step (delta={self.delta}, unit={self.unit})"
            )
        if problems:
            raise ValueError(
                "Prediction dataset is not consistent with the training dataset: "
                + "; ".join(problems)
            )

    def _check_domain(self, design_matrices, domain) -> None:
        """Flag `design_matrices.points` inside/outside `domain` into `self.mask`, without dropping any."""
        if domain is None:
            self.mask = None
            return

        # Vectorized (shapely 2.x ufuncs) rather than a per-point Python loop:
        # ~35x faster at N in the millions. `prepare` builds a spatial index
        # on `domain` for the repeated covers() queries; it mutates `domain`
        # in place (attaches a GEOS PreparedGeometry) but not its coordinates.
        shapely.prepare(domain)
        points = shapely.points(design_matrices.points)
        self.mask = shapely.covers(domain, points)

        n_in = int(self.mask.sum())
        n_out = int(self.mask.size - n_in)
        self._log(
            f"Scientific interest domain check: {n_in} point(s) inside, "
            f"{n_out} point(s) outside the domain ({self.mask.size} total)"
        )

    def __call__(self, verbose=None):
        """
        Return the built `DesignMatrices`, building them first if needed.

        Convenience wrapper around `build`: calls it only if this builder
        has not been built yet, otherwise returns the cached result.

        Parameters
        ----------
        verbose : bool, optional
            Forwarded to `build` if a build is triggered; ignored if the
            cached `DesignMatrices` is returned instead.

        Returns
        -------
        DesignMatrices
            The (possibly cached) result of `build`.
        """
        if not hasattr(self, "design_matrices"):
            self._log("Design matrices not found, calling build()")
            return self.build(verbose=verbose)
        self._log("Returning cached design matrices")
        return self.design_matrices

    # ------------------------------------------------------------------
    # Validation helpers
    #
    # All `_check_*` helpers below either return the data they validated
    # or raise `ValueError` with an explanatory message -- there is no
    # separate (flag, msg) return convention to keep in sync with the
    # caller.
    # ------------------------------------------------------------------

    def _check_formula(self, formula: str) -> _FormulaInfo:
        """Parse `formula` with Patsy and extract response/covariate metadata."""
        self._log(f"Checking formula: {formula}")

        if not isinstance(formula, str) or not formula.strip():
            raise ValueError("Formula must be a non-empty string")

        try:
            model_desc = ModelDesc.from_formula(formula)
        except Exception as exc:
            raise ValueError(f"Invalid formula: {exc}") from exc

        if len(model_desc.lhs_termlist) > 0:
            y_names, y_exprs, y_columns = _extract_formula_metadata(model_desc.lhs_termlist)

            if len(y_names) == 0:
                raise ValueError("Could not identify the response variable from the formula")
            if len(y_names) > 1:
                raise ValueError("Only one response variable is supported in the formula")

            response_name, response_expr, response_column = y_names[0], y_exprs, y_columns[0]
            self._log(f"Response name: {response_name}")
            self._log(f"Response expression(s): {', '.join(y_exprs) if y_exprs else response_name}")
        else:
            response_name, response_expr, response_column = None, None, None

        covariate_names, covariate_exprs, covariate_columns = _extract_formula_metadata(model_desc.rhs_termlist)
        self._log(f"Covariate name(s): {', '.join(covariate_names) if covariate_names else 'intercept only'}")
        self._log(f"Covariate transformation(s): {', '.join(covariate_exprs) if covariate_exprs else 'intercept only'}")

        return _FormulaInfo(
            lhs_termlist=model_desc.lhs_termlist,
            rhs_termlist=model_desc.rhs_termlist,
            response_name=response_name,
            response_expr=response_expr,
            response_column=response_column,
            covariate_names=covariate_names,
            covariate_exprs=covariate_exprs,
            covariate_columns=covariate_columns,
        )

    def _check_spatial_dataset(self, geodf) -> str:
        """Validate CRS/geometry column/validity/single geom type, and assign a `geometry_id` column."""
        self._log("Checking spatial dataset")

        problems = []
        if geodf.crs is None:
            problems.append("Dataset CRS not found")
        if "geometry" not in geodf:
            problems.append("Rename the geometry column to 'geometry'")
        else:
            if not geodf.is_valid.all():
                problems.append("Some geometries are invalid, check with .is_valid")
            geom_types = pd.unique(geodf.geom_type)
            if geom_types.shape != (1,):
                problems.append(f"Only one geometry type is supported, found {list(geom_types)}")

        if problems:
            raise ValueError("; ".join(problems))

        geometry_id = "geometry_id"
        categories = pd.Categorical(geodf["geometry"], categories=geodf.geometry.unique())
        geodf[geometry_id] = categories.codes

        return geometry_id

    def _check_time_dataset(self, geodf) -> str:
        """Return the name of the time column: ``"Time"`` if present, else the first datetime64 column."""
        self._log("Searching for time column")

        if "Time" in geodf:
            return "Time"

        datetime_cols = [
            col for col in geodf.columns if pd.api.types.is_datetime64_any_dtype(geodf[col])
        ]
        if not datetime_cols:
            raise ValueError("No 'Time' column found and no datetime column detected")

        return datetime_cols[0]

    def _coerce_time_column(self, geodf, time_col_name) -> None:
        """Convert `geodf[time_col_name]` to datetime64 in place, if not already."""
        if pd.api.types.is_datetime64_any_dtype(geodf[time_col_name]):
            return

        self._log(f"Converting time column '{time_col_name}' to datetime")
        geodf[time_col_name] = pd.to_datetime(geodf[time_col_name], errors="coerce")
        if geodf[time_col_name].isna().any():
            raise ValueError(
                f"Time column '{time_col_name}' contains values that could not be "
                "converted to datetime"
            )

    def _check_formula_columns_exist(self, geodf, prediction=False) -> None:
        """Raise ValueError if a column referenced by `formula` is missing from `geodf`."""
        referenced = set(self.formula_info.covariate_columns)
        if self.formula_info.response_column and not prediction:
            referenced.add(self.formula_info.response_column)

        # Interaction terms are stored as "colA:colB"; flatten before checking.
        columns = set()
        for entry in referenced:
            columns.update(entry.split(":"))
        columns.discard("Intercept")

        missing = sorted(columns - set(geodf.columns))
        if missing:
            raise ValueError(
                f"Formula references column(s) not found in the dataset: {', '.join(missing)}"
            )

    def _check_time_regularity(self, geodf, time_col_name):
        """Return `(delta, unit)` via `check_regular_timestamps`, raising ValueError if irregular."""
        is_regular, delta, unit = check_regular_timestamps(geodf[time_col_name])
        if not is_regular:
            raise ValueError(f"Timestamps in column '{time_col_name}' are not regularly spaced")
        return delta, unit

    def _cast_numeric_columns(self, geodf, geometry_id) -> None:
        """Cast all numeric columns of `geodf` (except `geometry_id`) to `self.dtype`, in place."""
        numeric_cols = [
            col for col in geodf.columns
            if pd.api.types.is_numeric_dtype(geodf[col]) and col != geometry_id
        ]
        geodf[numeric_cols] = (
            geodf[numeric_cols]
            .apply(lambda col: pd.to_numeric(col, errors="coerce"))
            .astype(self.dtype)
        )
        self._log(f"Converted numeric columns to dtype={self.dtype}")

    def _filter_time_range(self, geodf, time_col_name, tmin, tmax):
        """Keep only rows with `time_col_name` in `[tmin, tmax]` (either bound optional)."""
        self._log("Filtering dataset by time range")
        if tmin is not None:
            geodf = geodf[geodf[time_col_name] >= pd.to_datetime(tmin)]
            self._log(f"Filtered dataset to tmin={pd.to_datetime(tmin).strftime('%Y-%m-%d')}")
        if tmax is not None:
            geodf = geodf[geodf[time_col_name] <= pd.to_datetime(tmax)]
            self._log(f"Filtered dataset to tmax={pd.to_datetime(tmax).strftime('%Y-%m-%d')}")

        if geodf.empty:
            raise ValueError(f"No rows remain after filtering to tmin={tmin}, tmax={tmax}")
        self._log(f"Dataset filtered to {len(geodf)} rows")
        return geodf

    def _check_balanced_panel(self, geodf, N, T) -> None:
        """
        The [N x T] reshape used to build X/y assumes a complete site x time
        grid: every one of the N sites has exactly one row for every one of
        the T timestamps (the response may still be NaN). If that does not
        hold, `len(geodf) != N * T` and numpy's reshape fails with a cryptic
        size-mismatch error -- this raises a clearer one instead.
        """
        if len(geodf) == N * T:
            return

        counts = geodf.groupby(self.geometry_id, observed=True).size()
        n_incomplete = int((counts != T).sum())
        raise ValueError(
            "Dataset is not a complete site x time grid: expected "
            f"{N} sites x {T} timestamps = {N * T} rows, got {len(geodf)}. "
            f"{n_incomplete} site(s) are missing rows for some timestamps. "
            "Every site needs one row per timestamp (with the response left as "
            "NaN where unobserved) for the design matrices to align correctly."
        )

    # ------------------------------------------------------------------
    # Design matrix construction
    # ------------------------------------------------------------------

    def _build_geodataframe(self, lhs_termlist, rhs_termlist) -> DesignMatrices:
        """Compute the training design matrix and wrap it into a `DesignMatrices`."""
        self._log("Creating spatial-temporal design matrices")

        points, y, y_design_info, Xbeta, x_design_info, N, T, timestamps = (
            self._compute_design_matrix(lhs_termlist, rhs_termlist)
        )

        self._log(f"Computed design matrices with N={N}, P={Xbeta.shape[1]}, T={T}")

        return DesignMatrices(
            y=y,
            y_design_info=y_design_info,
            y_name=y_design_info.column_names if y_design_info else None,
            y_expr=self.formula_info.response_expr,
            X=Xbeta,
            X_design_info=x_design_info,
            x_names=x_design_info.column_names,
            x_exprs=self.formula_info.covariate_exprs,
            points=points,
            formula=self.formula,
            crs=self.crs,
            box=self.box,
            geometry=self.geometry,
            timestamps=timestamps,
            delta=self.delta,
            unit=self.unit,
            time_col_name=self.time_col_name,
            dtype=self.dtype,
        )

    def _compute_design_matrix(self, lhs_termlist, rhs_termlist):
        """Evaluate `formula` on `self.geodf` with Patsy and reshape the result to `[N, P, T]`/`[N, T]`."""
        self._log("Computing design matrix from GeoDataFrame")

        geodf = self.geodf.sort_values([self.time_col_name, self.geometry_id])
        geodf = geodf.drop_duplicates(subset=[self.geometry_id, self.time_col_name])
        self._log("Dataset sorted by time/geometry id, duplicate space-time rows dropped")

        # len(lhs_termlist) > 0 -> response required, == 0 -> covariates-only formula
        if len(lhs_termlist) > 0:
            observed_sites = geodf.loc[
                geodf[self.formula_info.response_column].notna(), self.geometry_id
            ].unique().tolist()
        else:
            observed_sites = geodf[self.geometry_id].unique().tolist()

        geodf = geodf[geodf[self.geometry_id].isin(observed_sites)]
        self._log(f"Kept {len(observed_sites)} observed spatial locations")

        timestamps = np.sort(np.unique(geodf[self.time_col_name]))
        T = timestamps.shape[0]

        points = geodf.geometry.unique()
        N = points.shape[0]
        self._log(f"Found {N} spatial locations and {T} unique timestamps")

        self._check_balanced_panel(geodf, N, T)

        na_action = _NAPassthrough()

        if len(lhs_termlist) > 0:
            y_matrix, x_matrix = dmatrices(
                self.formula, data=geodf, NA_action=na_action, return_type="matrix",
            )
            y_matrix[np.isinf(y_matrix)] = np.nan
            y = y_matrix.reshape(T, N).T
            y_design_info = y_matrix.design_info
            self._log(f"y name: {y_design_info.column_names[0]}, shape={y.shape}")
        else:
            x_matrix = dmatrix(
                ModelDesc(lhs_termlist, rhs_termlist),
                data=geodf, NA_action=na_action, return_type="matrix",
            )
            y = None
            y_design_info = None

        x_design_info = x_matrix.design_info
        self._log(f"X names: {', '.join(x_design_info.column_names)}, shape={x_matrix.shape}")

        Xbeta = np.zeros((N, x_matrix.shape[1], T), dtype=self.dtype)
        for i in range(x_matrix.shape[1]):
            Xbeta[:, i, :] = x_matrix[:, i].reshape(T, 1, N).T.squeeze(axis=1)
        self._log(f"Reshaped X to {Xbeta.shape}")

        return points, y, y_design_info, Xbeta, x_design_info, N, T, timestamps

    def _compute_predict_design_matrix(
        self, geodf, geometry_id, time_col_name, crs, box, geometry, delta, unit,
    ) -> DesignMatrices:
        """
        Counterpart to `_build_geodataframe`/`_compute_design_matrix` for new
        (prediction) data: never evaluates a response, and always reuses
        this builder's fitted `X_design_info` (via `build_design_matrices`)
        instead of re-parsing the formula, so stateful transforms (e.g.
        `standardize(...)`) are evaluated with the training mean/std rather
        than being recomputed on `geodf`.
        """
        self._log("Computing prediction design matrix from GeoDataFrame")

        geodf = geodf.sort_values([time_col_name, geometry_id])
        geodf = geodf.drop_duplicates(subset=[geometry_id, time_col_name])
        self._log("Dataset sorted by time/geometry id, duplicate space-time rows dropped")

        timestamps = np.sort(np.unique(geodf[time_col_name]))
        T = timestamps.shape[0]

        points = geodf.geometry.unique()
        N = points.shape[0]
        self._log(f"Found {N} spatial locations and {T} unique timestamps")

        self._check_balanced_panel(geodf, N, T)

        na_action = _NAPassthrough()
        design_info = self.design_matrices.X_design_info
        self._log(
            "Reusing training design_info to build the prediction design "
            "matrix, so stateful transforms (e.g. standardize()) reuse the "
            "training statistics instead of being recomputed on this data"
        )
        (x_matrix,) = build_design_matrices(
            [design_info], data=geodf, NA_action=na_action, return_type="matrix",
        )

        x_design_info = x_matrix.design_info
        self._log(f"X names: {', '.join(x_design_info.column_names)}, shape={x_matrix.shape}")

        Xbeta = np.zeros((N, x_matrix.shape[1], T), dtype=self.dtype)
        for i in range(x_matrix.shape[1]):
            Xbeta[:, i, :] = x_matrix[:, i].reshape(T, 1, N).T.squeeze(axis=1)
        self._log(f"Reshaped X to {Xbeta.shape}")

        self._log(f"Computed prediction design matrices with N={N}, P={Xbeta.shape[1]}, T={T}")

        return DesignMatrices(
            y=None,
            y_design_info=None,
            y_name=None,
            y_expr=self.formula_info.response_expr,
            X=Xbeta,
            X_design_info=x_design_info,
            x_names=x_design_info.column_names,
            x_exprs=self.formula_info.covariate_exprs,
            points=points,
            formula=self.formula,
            crs=crs,
            box=box,
            geometry=geometry,
            timestamps=timestamps,
            delta=delta,
            unit=unit,
            time_col_name=time_col_name,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------
    # Misc helpers (not currently wired into build(), kept for reuse)
    # ------------------------------------------------------------------

    def _getPoints(self, geodf, geometry_id):
        """Return unique geometry centroids as an `[N, 2]` coordinate array (computed in a metric UTM CRS)."""
        self._log("Computing centroid coordinates for geometry points")

        if "geometry" not in geodf:
            raise ValueError("Geometry column not found in the dataset")

        gdf_metric = geodf.to_crs(geodf.estimate_utm_crs())
        centroids = gdf_metric.geometry.drop_duplicates().centroid.to_crs(geodf.crs)
        points = np.column_stack([centroids.x, centroids.y])

        self._log(f"Computed {points.shape[0]} point coordinates")
        return points

    def _computeDistance(self, points, pt=None, distance="euclidean"):
        """Return the (cross-)distance matrix between `points` and `pt` (or `points` itself if `pt` is None), via `scipy.spatial.distance.cdist`."""
        self._log(f"Computing distance matrix using '{distance}' metric")

        if pt is None:
            dist = cdist(points, points, distance)
            self._log(f"Computed square distance matrix with shape {dist.shape}")
            return dist
        else:
            dist = cdist(points, pt, distance)
            self._log(f"Computed cross-distance matrix with shape {dist.shape}")
            return dist

    @staticmethod
    def print_info(msg):
        """
        Print a UTC-timestamped log line.

        Parameters
        ----------
        msg : str
            Message to print, prefixed with the current UTC timestamp
            (``YYYY-MM-DD HH:MM:SS``).
        """
        dt = datetime.now(timezone.utc)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")
