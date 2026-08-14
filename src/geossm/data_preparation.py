"""
Data preparation module for GEOSSM.

This module provides functionality to prepare spatial-temporal datasets
for modeling, including design matrix construction and dataset validation.

DesignMatrices class encapsulates the design matrices and related metadata.

DataPreparation class handles the preparation of the DesignMatrices object.

"""

from datetime import datetime, timezone
import re
import time
import numpy as np
from scipy.spatial.distance import cdist
from patsy import ModelDesc, dmatrices,dmatrix
import geopandas as geopd
import pandas as pd
from shapely.geometry import Point

from dataclasses import dataclass, field
from typing import Literal
import ast

from statsmodels.iolib.summary import Summary


try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class DesignMatrices:
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
    y: np.ndarray = None, 
    y_design_info: object = None
    y_name: str = None
    y_expr: list[str] = None
    time_col_name: str = "Time"
    geometry_id: str = field(default="geometry_id", init=False)
    dtype: np.dtype = field(default=np.float64)
    # backend: Literal["numpy", "jax"] = field(default="numpy")

    def __post_init__(self):
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

        # Cast arrays to dtype 
        self.y = np.asarray(self.y, dtype=self.dtype) if self.y is not None else None
        self.X = np.asarray(self.X, dtype=self.dtype)
        self.timestamps = np.asarray(self.timestamps)
        self.type = np.unique(self.geometry).astype(str)
        self.points = np.array([[p.x, p.y] for p in self.points], dtype=self.dtype)

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
        y_np = np.asarray(self.y) if self.y is not None else None

        def get_print_string(str):
            return str if len(str) <= 20 else str[:25] + "..."
        
        top_left = dict([
            ("Formula",        lambda: [get_print_string(self.formula)]),
            ("Response (y)",   lambda: [self.y_design_info.column_names[0] if self.y_design_info else self.y_name] if self.y is not None else ["N/A"]),
            ("Covariates (X)", lambda: [", ".join(self.x_names)]),
            ("[min, max, mean]", lambda: [f"[{np.nanmin(y_np):.4g}, {np.nanmax(y_np):.4g}, {np.nanmean(y_np):.4g}]"] if self.y is not None else ["N/A"]),  
            ("Observed",         lambda: [f"{self.n_obs} / {self.N * self.T} ({(1 - self.nan_ratio) * 100:.1f}%)"] if self.y is not None else ["N/A"]),
            ("Missing",          lambda: [f"{int(np.isnan(y_np).sum())} ({self.nan_ratio * 100:.1f}%)"] if self.y is not None else ["N/A"]),
            ("Transformed (y)",       lambda: [get_print_string(", ".join(self.y_expr) if self.y_expr else "No")] if self.y is not None else ["N/A"]),
            ("Transformed (X)",       lambda: [get_print_string(", ".join(self.x_exprs) if self.x_exprs else "No")]),
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
            gen_top_right.append((item, top_right[item]()))

        return gen_top_left, gen_top_right

    def summary(self) -> Summary:
        
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
        return self.summary().as_text()

    def __repr__(self) -> str:
        return (
            f"DesignMatrices("
            f"N={self.N}, b={self.b}, T={self.T}, "
            f"dtype={self.dtype}')"
        )

class DesignMatricesBuilder:

    def __init__(self, geodf: geopd.GeoDataFrame, formula: str, dtype=np.float64, verbose: bool = True,
                 tmin: datetime = None, tmax: datetime = None):
        """
        Prepare the spatial-temporal dataset for modeling.
        """
        self.verbose = verbose
        self.dtype = np.dtype(dtype)
        self._log("Initializing DesignMatricesBuilder", verbose)

        self.geodf = geodf.copy()
        self.formula = formula
        self.crs = self.geodf.crs
        self.box = np.round(self.geodf.total_bounds, 3).tolist() # [minx, miny, maxx, maxy]


        # Check the formula and extract response/covariate names and expressions
        (flag,
            msg,
            self.lhs_termlist, 
            self.rhs_termlist,
            self.response_name,
            self.response_expressions,
            self.response_column, 
            self.covariate_names,
            self.covariate_expressions,
            self.covariate_columns,
        ) = self._checkFormula(self.formula, verbose=verbose)
        
        if not flag:
            raise ValueError(msg)

        if len(self.lhs_termlist) == 0:
            self._log(
                "Formula parsed successfully. No response variable found",
                verbose)
            
        else:     
            self._log(
                f"Formula parsed successfully. Response variable: '{self.response_name}'",
                verbose)
            

        if isinstance(self.geodf, geopd.GeoDataFrame):
            flag = self._check_geodataframe(self.geodf, self.dtype, verbose=verbose)
            if not flag:
                raise ValueError(
                    "Input dataset must be a GeoDataFrame with valid geometry and time columns"
                )
            self._log("Input GeoDataFrame validated successfully", verbose)
        else:
            raise ValueError("Input dataset must be a GeoDataFrame")
        
        # Cut the dataset to the specified time range if tmin and tmax are provided       
        if tmin is not None or tmax is not None:
            self._log("Filtering dataset by time range", verbose)
            if tmin is not None:
                self.geodf = self.geodf[self.geodf[self.time_col_name] >= pd.to_datetime(tmin)]
                self._log(f"Filtered dataset to tmin={pd.to_datetime(tmin).strftime('%Y-%m-%d')}", verbose)
            if tmax is not None:
                self.geodf = self.geodf[self.geodf[self.time_col_name] <= pd.to_datetime(tmax)]
                self._log(f"Filtered dataset to tmax={pd.to_datetime(tmax).strftime('%Y-%m-%d')}", verbose)
            self._log(f"Dataset filtered to {len(self.geodf)} rows", verbose)


    def _is_verbose(self, verbose=None) -> bool:
        return self.verbose if verbose is None else verbose

    def _log(self, msg: str, verbose=None) -> None:
        if self._is_verbose(verbose):
            self.print_info(msg)

    def build(self, predict=False, verbose=None):
         
        if isinstance(self.geodf, geopd.GeoDataFrame):
            self._log("Building design matrices from GeoDataFrame", verbose)
            
            if predict == False:
                self.design_matrices = self._build_geodataframe(self.lhs_termlist, self.rhs_termlist, verbose=verbose)
            else: 
                self.design_matrices = self._build_geodataframe([], self.rhs_termlist, verbose=verbose)

        else:
            raise ValueError("Input dataset must be a GeoDataFrame")   
        
        self._log("Design matrices built successfully", verbose)
        return self.design_matrices

    def __call__(self, verbose=None):
        if not hasattr(self, "design_matrices"):
            self._log("Design matrices not found, calling build()", verbose)
            return self.build(verbose=verbose)
        self._log("Returning cached design matrices", verbose)
        return self.design_matrices

    def _check_geodataframe(self, geodf, dtype, verbose=None):
        self._log("Checking input GeoDataFrame", verbose)

        flag, msg, self.geometry_id = self._checkSpatialDataset(geodf, verbose=verbose)
        if not flag:
            raise ValueError(msg)
        self._log(f"Spatial check passed using geometry id column '{self.geometry_id}'", verbose)

        flag, msg, self.time_col_name = self._checkTimeDataset(geodf, verbose=verbose)
        if not flag:
            raise ValueError(msg)
        self._log(f"Time column detected: '{self.time_col_name}'", verbose)

        # check the time stamps format and convert from string to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(geodf[self.time_col_name]):
            #raise TypeError(f"{self.time_col_name} must be a datetime column")
            self._log(f"Converting time column '{self.time_col_name}' to datetime", verbose)
            geodf[self.time_col_name] = pd.to_datetime(geodf[self.time_col_name], errors="coerce")
            if geodf[self.time_col_name].isna().any():
                raise ValueError(f"Time column '{self.time_col_name}' contains non-datetime values that could not be converted")
            self._log(f"Time column '{self.time_col_name}' converted to datetime successfully", verbose)
        
        self._log(
            f"Column detected: {', '.join(self.covariate_names) if self.covariate_names else 'intercept only'}",
            verbose,
        )
        
        # Check the time column consistency and get the delta and unit for the time dimension
        flag, self.delta, self.unit = self._checkTimeColumn(
            geodf,
            self.response_name,
            self.geometry_id,
            self.time_col_name,
            verbose=verbose,
        )
        if not flag:
            raise ValueError(msg)
        self._log(f"Time consistency check passed: delta {self.delta}, unit {self.unit}", verbose)

        flag, msg, self.geometry = self._check_geometry(self.geodf, verbose=verbose)
        if not flag:
            raise ValueError(msg)
        self._log(f"Geometry type detected: {', '.join(self.geometry.astype(str))}", verbose)

        numeric_col = self._numeric_column(self.geodf, verbose=verbose)
        self.geodf[numeric_col] = (
            self.geodf[numeric_col]
            .apply(lambda col: pd.to_numeric(col, errors="coerce"))
            .astype(dtype)
        )
        self._log(f"Converted numeric columns to dtype={dtype}", verbose)

        return True

    def _build_geodataframe(self, lhs_termlist, rhs_termlist, verbose=None):
        self._log("Creating spatial-temporal design matrices", verbose)


        geodf, points, y, _y_design_info, Xbeta, _x_design_info, N, T, timestamps = (
            self._computedesignMatrix_geodataframe(
                lhs_termlist, 
                rhs_termlist, 
                self.geodf,
                self.geometry_id,
                self.time_col_name,
                self.response_column,
                verbose=verbose,
            )
        )

        self._log(f"Computed design matrices with N={N}, P={Xbeta.shape[1]}, T={T}", verbose)

        # TODO: convert points to a more compact representation if needed, e.g. by using a KD-tree or similar structure for spatial indexing
        # TODO: consider to check if the geometry is already in a compact form (e.g. centroids) and skip this step if so
        # self.points = self._getPoints(geodf, self.geometry_id, verbose=verbose)

        return DesignMatrices(
            y=y,
            y_design_info=_y_design_info,
            y_name = _y_design_info.design_info.column_names if _y_design_info else None,
            y_expr = self.response_expressions,
            X=Xbeta,
            X_design_info=_x_design_info,
            x_names = _x_design_info.design_info.column_names if _x_design_info else None,
            x_exprs = self.covariate_expressions,
            points=points,
            formula=self.formula,
            crs=self.crs,
            box = self.box,
            geometry=self.geometry,
            timestamps=timestamps,
            delta = self.delta,
            unit = self.unit,
            time_col_name=self.time_col_name,
            dtype=self.dtype,
            # backend="jax" if JAX_AVAILABLE else "numpy", 
        )

    def _computedesignMatrix_geodataframe(
        self,
        lhs_termlist,
        rhs_termlist,
        geodf,
        geometry_id,
        time_col_name,
        response_column,
        verbose=None,
    ):
        self._log("Computing design matrix from GeoDataFrame", verbose)
        
        if lhs_termlist is None and rhs_termlist is None:
            msg = "Formula must be provided to compute design matrices"
            raise ValueError(msg)
        
        if rhs_termlist is None:
            msg = "Covariate terms must be provided in the formula to compute design matrices"
            raise ValueError(msg)


        geodf = geodf.sort_values([time_col_name, geometry_id])
        self._log("Dataset sorted by time and geometry id", verbose)

        geodf = geodf.drop_duplicates(subset=[geometry_id, time_col_name])
        self._log("Dropped duplicate space-time rows", verbose)
        

        # len() >0 estimation
        # len() =0 prediction 
        if len(lhs_termlist) > 0:
            #stp = (
            #    geodf.groupby(geometry_id, observed=True)[response_name]
            #    .count()
            #    ).reset_index()

            #observed_sites = stp.loc[stp[response_name] > 0, geometry_id].tolist()
            observed_sites = geodf.loc[geodf[response_column].notna(), geometry_id].unique().tolist()
        else:
            observed_sites = geodf[geometry_id].unique().tolist()

        geodf = geodf[geodf[geometry_id].isin(observed_sites)]
        self._log(f"Kept {len(observed_sites)} observed spatial locations", verbose)

        timestep = np.sort(np.unique(geodf[time_col_name]))
        T = timestep.shape[0]
        self._log(f"Found {T} unique timestamps", verbose)

        points = geodf.geometry.unique()
        N = points.shape[0]
        self._log(f"Found {N} spatial locations", verbose)

        self._log("Generating design matrices...", verbose)
        
        from patsy import NAAction
        class NAPassthrough(NAAction):
            def __init__(self):
                super().__init__(NA_types=[])  # tell patsy not to treat anything as NA
        
        if len(lhs_termlist) > 0:

            ytemp, Xtemp = dmatrices(
                self.formula,
                data=geodf,
                NA_action=NAPassthrough(),
                return_type="matrix",
            )
            self._log(f"y name: {ytemp.design_info.column_names[0]}", verbose)
            self._log(f"Design matrix shapes: ytemp={ytemp.shape}", verbose)
            
            ytemp[np.isinf(ytemp)] = np.nan
            self._log("Restored NaN values in response matrix", verbose)
            
            y = ytemp.reshape(T, N).T
            self._log(f"Reshaped y to {y.shape}", verbose)

        else:
            
            Xtemp = dmatrix(
                ModelDesc(lhs_termlist, rhs_termlist),
                data=geodf,
                NA_action=NAPassthrough(),
                return_type="matrix",
            )
            y = None
            self._log("No response variable specified, only design matrix X will be computed", verbose)
    
        self._log(f"X names: {', '.join(Xtemp.design_info.column_names)}", verbose)
        self._log(f"Design matrix shapes: Xtemp={Xtemp.shape}", verbose)

        
        Xbeta = np.zeros((N, Xtemp.shape[1], T), dtype=self.dtype)
        for i in range(Xtemp.shape[1]):
            Xbeta[:, i, :] = Xtemp[:, i].reshape(T, 1, N).T.squeeze(axis=1)

        self._log(f"Reshaped X to {Xbeta.shape}", verbose)
        self._log("Design matrices computed successfully", verbose)

        return (
            geodf,
            points,
            y,
            ytemp.design_info if y is not None else None,
            Xbeta,
            Xtemp.design_info,
            N,
            T,
            timestep,
        )

    def _checkFormula(self, formula: str, verbose=None):
        self._log(f"Checking formula: {formula}", verbose)

        if not isinstance(formula, str) or not formula.strip():
            return False, "Formula must be a non-empty string", None, [], [], []

        try:
            m = ModelDesc.from_formula(formula)
        except Exception as exc:
            return False, f"Invalid formula: {exc}", None, [], [], []

        # parse the response variable
        if len(m.lhs_termlist) > 0:    
            # return False, "Formula must include a response variable on the left-hand side", None, [], [], []
            y_names, y_exprs, y_column = self._extract_formula_metadata(m.lhs_termlist)

            if len(y_names) == 0:
                return (False, 
                "Could not identify the response variable from the formula", None, [], [], [])

            if len(y_names) > 1:
                return (
                    False,
                    "Only one response variable is supported in the formula",
                    None,
                    [],
                    [],
                    [],
                )
            else:
                y_column = y_column[0]


            y_names = y_names[0]
            self._log(f"Response name: {y_names}", verbose)

            self._log(
            f"Response expression(s): {', '.join(y_exprs) if y_exprs else y_names}",
            verbose)
        else:
            y_names, y_exprs, y_column = None, None, None
        

        # parse the covariates
        x_names, x_exprs, x_columns = self._extract_formula_metadata(m.rhs_termlist)

        
        self._log(
            f"Covariate name(s): {', '.join(x_names) if x_names else 'intercept only'}",
            verbose,
        )
        self._log(
            f"Covariate expression(s): {', '.join(x_exprs) if x_exprs else 'intercept only'}",
            verbose,
        )

        return True, "", m.lhs_termlist, m.rhs_termlist, y_names, y_exprs, y_column, x_names, x_exprs, x_columns
       

    def _extract_formula_metadata(self, termlist):
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


    def _checkTimeColumn(self, geodf, response_name, geometry_id, time_col_name, verbose=None):
        self._log(f"Checking time column consistency for '{time_col_name}'", verbose)

        flag, delta, unit = self.check_regular_timestamps(
            geodf[time_col_name],
            verbose=verbose,
        )
        if not flag:
            msg = "Timestamps are not regularly spaced"
            raise ValueError(msg)

        self._log(f"Time column check completed: delta={delta}, unit={unit}", verbose)
        return flag, delta, unit

    def check_regular_timestamps(self, timestamps, verbose=None):
        """
        Check if timestamps are regularly spaced and return (delta, unit).

        Notes
        -----
        - Uses UNIQUE timestamps (important for panel data with repeated times per geometry).
        - Handles parsing errors robustly.
        - Supports fixed deltas (sec/min/hour/day) and calendar frequencies
          (month/year) when inferable.
        """
        self._log("Checking regular spacing of timestamps", verbose)

        # 1) Parse safely
        ts = pd.Series(timestamps)
        ts = pd.to_datetime(ts, errors="coerce", utc=True)

        if ts.isna().any():
            n_bad = int(ts.isna().sum())
            self._log(f"Found {n_bad} invalid timestamps", verbose)
            return False, None, None

        # 2) Use unique sorted times (avoid duplicated rows across geometries)
        ts_unique = pd.DatetimeIndex(ts.drop_duplicates().sort_values())


        if ts_unique.size == 1:
            self._log("Only one unique timestamp found", verbose)
            return True, 0, "None"

        if ts_unique.size < 2:
            self._log("Need at least 2 unique timestamps", verbose)
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
                self._log(f"Regular calendar frequency detected: {inferred}", verbose)
                return True, mult, "month"

            # Year-like frequencies
            if base in {"A", "AS", "Y", "YS", "YE", "BA", "BAS"}:
                self._log(f"Regular calendar frequency detected: {inferred}", verbose)
                return True, mult, "year"

        # 4) Fallback: fixed timedeltas
        diffs = pd.Series(ts_unique).diff().dropna()
        if diffs.empty:
            self._log("Could not compute timestamp differences", verbose)
            return False, None, None

        first = diffs.iloc[0]
        if not (diffs == first).all():
            self._log("Timestamp spacing is not constant", verbose)
            return False, None, None

        seconds = first.total_seconds()
        if seconds <= 0:
            self._log("Non-positive timestamp delta detected", verbose)
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

    def _checkTimeDataset(self, geodf, verbose=None):
        self._log("Searching for time column", verbose)

        msg = ""
        flag = True
        time_col_name = None

        if "Time" not in geodf:
            time_col = [
                col
                for col in geodf.columns
                if pd.api.types.is_datetime64_any_dtype(geodf[col])
            ]

            if len(time_col) == 0:
                msg += "The 'Time' column not found \n"
                flag = False
                self._log("No valid time column found", verbose)
            else:
                time_col_name = time_col[0]
                msg += "'Time' column found: {col} \n".format(col=time_col)
                msg += "Keeped 'Time' column: {col} \n".format(col=time_col_name)
                self._log(f"Using datetime column '{time_col_name}' as time column", verbose)
        else:
            time_col_name = "Time"
            self._log("Using 'Time' column", verbose)

        return flag, msg, time_col_name

    def _check_geometry(self, geodf, verbose=None):
        self._log("Checking geometry type", verbose)

        geom_type = pd.unique(self.geodf.geometry.geom_type)

        msg = ""
        flag = True
        if len(geom_type) > 1:
            msg = "Should be only one geometry type"
            flag = False
            self._log("Multiple geometry types found", verbose)
        else:
            self._log(f"Geometry type check passed: {geom_type[0]}", verbose)

        return flag, msg, geom_type

    def _numeric_column(self, geodf, verbose=None):
        numeric_col = [
            col for col in geodf.columns if pd.api.types.is_numeric_dtype(geodf[col])
        ]
        self._log(f"Found numeric columns: {numeric_col}", verbose)
        return numeric_col

    def _getPoints(self, geodf, geometry_id, verbose=None):
        self._log("Computing centroid coordinates for geometry points", verbose)

        if "geometry" not in geodf:
            raise ValueError("Geometry column not found in the dataset")

        gdf_metric = geodf.to_crs(geodf.estimate_utm_crs())
        centroids = gdf_metric.geometry.drop_duplicates().centroid.to_crs(geodf.crs)
        points = np.column_stack([centroids.x, centroids.y])

        self._log(f"Computed {points.shape[0]} point coordinates", verbose)
        return points

    def _computeDistance(self, points, pt=None, distance="euclidean", verbose=None):
        self._log(f"Computing distance matrix using '{distance}' metric", verbose)

        if pt is None:
            dist = cdist(points, points, distance)
            self._log(f"Computed square distance matrix with shape {dist.shape}", verbose)
            return dist
        else:
            dist = cdist(points, pt, distance)
            self._log(f"Computed cross-distance matrix with shape {dist.shape}", verbose)
            return dist

    def _checkSpatialDataset(self, geodf, verbose=None):
        self._log("Checking spatial dataset", verbose)

        msg = ""
        flag = True

        if not isinstance(geodf, geopd.geodataframe.GeoDataFrame):
            msg += "Type of dataset must be geopandas.geodataframe see: (lint to doc) \n"
            flag = False
            self._log("Dataset is not a GeoDataFrame", verbose)

        if geodf.crs is None:
            msg += "Dataset CRS not found: (lint to doc) \n"
            flag = False
            self._log("Dataset CRS is missing", verbose)

        mask = geodf.is_valid
        if not mask.all():
            msg += "Check the rows geometry: (.is_valid) \n"
            flag = False
            self._log("Invalid geometries found", verbose)

        if "geometry" not in geodf:
            msg += "Rename the column with the geometry 'geometry' \n"
            flag = False
            self._log("Geometry column not found", verbose)

        geometry_id = "geometry_id"
        ct = pd.Categorical(geodf["geometry"], categories=geodf.geometry.unique())
        geodf[geometry_id] = ct.codes

        mask = np.unique(geodf.geom_type)
        if mask.shape != (1,):
            msg += (
                "Just one spatial geometry is supported. Currently found geometries "
                "{maks} \n".format(maks=mask)
            )
            flag = False
            self._log(f"Multiple geometry types found: {mask}", verbose)
        else:
            self._log(f"Spatial dataset check passed with geometry type {mask[0]}", verbose)

        return flag, msg, geometry_id

    def print_info(self, msg):
        dt = datetime.fromtimestamp(time.time(), tz=timezone.utc)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")
