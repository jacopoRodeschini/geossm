"""
Data preparation module for GEOSSM.

This module provides functionality to prepare spatial-temporal datasets
for modeling, including design matrix construction and dataset validation.

DesignMatrices class encapsulates the design matrices and related metadata.

DataPreparation class handles the preparation of the DesignMatrices object.

"""
import numpy as np
from scipy.spatial.distance import cdist
from patsy import ModelDesc, dmatrices
import geopandas as geopd
import pandas as pd


class DesignMatrices:

    def __init__(self, y: np.array, y_design_info, X: np.array, X_design_info, formula: str,
                 csr, geometry, timestamps, time_col_name='Time'):
        """
        Design Matrices for spatial-temporal modeling.
        Construct while inittialized.
        Supposed to be return from data_preparation class.
        """
        self.y = y  # target variables, np.array [N x T]
        self.X = X  # covariates, np.array [N x P x T]
        self.y_design_info =  y_design_info # design info for y (patsy)
        self.X_design_info = X_design_info  # design info for X (patsy)
        self.formula = formula  # formula string
        self.crs = csr  # coordinate reference system
        self.geometry = geometry  # obervations geometries
        self.geometry_id = 'geometry_id'  # geometry id column name
        self.time_col_name = time_col_name  # time column name
        self.timestamps = timestamps  # observations timestamps
        # number of points, number of covariates, number of timestamps
        self.N, self.b, self.T = X.shape
        self.terms = ModelDesc.from_formula(formula)
    
    def __str__(self):
        description = """Design matrices object
----------------------------------
y: {y_shape}
y name : {y_name}       

X: {x_shape}                    
X name: {x_name}
----------------------------------
""".format(y_shape=self.y.shape, y_name=self.y_design_info.column_names,
                x_shape=self.X.shape, x_name=self.X_design_info.column_names)
        return description


class data_preparation:

    def __init__(self, geodf: geopd.GeoDataFrame, formula: str, dtype=np.float64):
        """
        Prepare the spatial-temporal dataset for modeling.
        Parameters
        ----------
        geodf : geopandas.GeoDataFrame
            Input spatial-temporal dataset with geometry and time columns.
        formula : str
            Patsy formula string for design matrix construction.
        Returns
        -------
        DesignMatrices
            An object containing the design matrices and related metadata.

        """

        self.geodf = geodf
        self.formula = formula

        if self._check(self.geodf, dtype):
            self.design_matrices = self._build()

    def __call__(self):
        return self.get()

    def get(self):
        return self.design_matrices

    def _check(self, geodf, dtype):
        # Do preliminary check of the dataset
        flag, msg, self.geometry_id = self._checkSpatialDataset(geodf)
        if not flag:
            raise ValueError(msg)

        flag, msg, self.time_col_name = self._checkTimeDataset(geodf)
        if not flag:
            raise ValueError(msg)

        # check formula (NOW NOT IMPLEMENTED) [return true]
        flag, msg, self.response_name = self._checkFormula(self.formula)
        if not flag:
            raise ValueError(msg)

        # Check the time column (timestmap / delta / complete - NAN)
        flag, msg = self._checkTimeColumn(
            geodf, self.response_name, self.geometry_id, self.time_col_name)

        if not flag:
            raise ValueError(msg)

        # check the geometry
        flag, msg, self.geometry = self._check_geometry(self.geodf)
        if not flag:
            raise ValueError(msg)

        # convert to dytype object
        numeric_col = self._numeric_column(self.geodf)
        self.geodf[numeric_col] = self.geodf[numeric_col].apply(
            lambda col: pd.to_numeric(col, errors='coerce')).astype(dtype)

        return True

    def _check_geometry(self, geodf):

        geom_type = pd.unique(self.geodf.geometry.geom_type)

        msg = ""
        flag = True
        if len(geom_type) > 1:
            msg = "Shoud be onlt one geometru type"
            flag = False

        return flag, msg, geom_type

    def _numeric_column(self, geodf):
        # check numeric value to dtype
        numeric_col = [
            col for col in geodf.columns if pd.api.types.is_numeric_dtype(geodf[col])]

        return numeric_col

    def _build(self):

        # Create new dataset & compute the designed matrix (spatiotemporal matrix)

        self.geodf, self.idPoint, self.y, self._y_design_info, self.X, self._x_design_info, self.N, self.T, self.timestamps = self._computedesignMatrix(
            self.geodf, self.geometry_id, self.time_col_name, self.response_name, self.formula)

        # Objcet geometry metrics attributes
        self.box = self.geodf.total_bounds
        self.geometry_type = self.geodf.geom_type
        self.crs = self.geodf.crs

        # Design matrix
        self.intercept = True  # default

        # Get the points of the response variable and distace
        self.points = self._getPoints(self.geodf, self.geometry_id)

        return DesignMatrices(self.y, self._y_design_info, self.X, self._x_design_info, self.formula ,self.crs, self.geometry, self.timestamps)

    def _computedesignMatrix(self, geodf, geometry_id, time_col_name, response_name, formula=None, terms=None):

        flag = True
        msg = ""

        if ((formula is None) and (terms is None)) or ((formula is not None) and (terms is not None)):
            fag = False
            msg += "Formula or terms must be provided"

        # sort the dataset by time
        geodf = geodf.sort_values([time_col_name, geometry_id])

        # take just the unique row
        geodf = geodf.drop_duplicates(subset=[geometry_id, time_col_name])

        # Create new dataset starting from formula
        if formula is not None:

            time = np.unique(geodf[time_col_name])
            T = time.shape[0]

            # TODO: not all point measure the variable (remove this point)
            # Check the ID statin with measure always nan (and merge same sites)

            group_cols = [geometry_id]
            stp = geodf.groupby(group_cols, observed=True).agg({
                response_name: lambda x: np.nansum(x)
            }).reset_index()

            idS = stp[stp[response_name] > 0].index

            geodf = geodf[geodf.geometry_id.isin(idS)]

            # check the number of point available
            point = geodf.geometry.unique()
            N = point.shape[0]

            # Convert Nan to inf (this becouse the dmatrices remove nan -> we have to keep this values)
            geodf.loc[geodf[response_name].isna(), response_name] = np.inf

            ytemp, Xtemp = dmatrices(
                formula, data=geodf, NA_action='raise', return_type='matrix')

            # replace inf with nan
            ytemp[np.isinf(ytemp)] = np.nan

            # just reshape:
            # response variable: [N x T]
            # Covariate variable: [N x P x T]
            y = ytemp.reshape(T, N).T

            Xbeta = np.zeros((N, Xtemp.shape[1], T))

            for i in range(0, Xtemp.shape[1]):
                Xbeta[:, i, :] = Xtemp[:, i].reshape(T, 1, N).T.squeeze(axis=1)

            return geodf, point, y, ytemp.design_info, Xbeta, Xtemp.design_info, N, T, time

    def _checkFormula(self, formula: str):

        # TODO: check the formula parser
        flag = True
        msg = ""

        m = ModelDesc.from_formula(formula)

        return True, msg, m.lhs_termlist[0].name()

    def _checkTimeColumn(self, geodf, response_name, geometry_id, time_col_name):

        # Check foreach geometry_id the timeseries lenght, start/end date, the delta time.
        flag = True
        msg = ""

        return flag, msg

    def _checkTimeDataset(self, geodf):
        msg = ""
        flag = True
        time_col_name = None

        # check if time is in dataset
        if not 'Time' in geodf:

            # check other columns with datatime dtype
            time_col = [
                col for col in geodf.columns if pd.api.types.is_datetime64_any_dtype(geodf[col])]

            if len(time_col) == 0:
                msg += "The 'Time' column not found \n"
                flag = False

            else:
                time_col_name = time_col[0]
                msg += "'Time' column found: {col} \n".format(col=time_col)
                msg += "Keeped 'Time' column: {col} \n".format(
                    col=time_col_name)

        else:
            time_col_name = 'Time'

        return flag, msg, time_col_name

    def _getPoints(self, geodf, geometry_id):

        # pojected geometry to_crs() for accurete results
        uni, idx = np.unique(geodf[geometry_id], return_index=True)

        # Extract the centroid by the geometry
        centroid = geodf.iloc[idx].geometry.centroid

        # Create a point vector (x, y)
        return np.stack((centroid.x, centroid.y), axis=1)

    def _computeDistance(self, points, pt=None, distance='euclidean'):

        if pt is None:

            # Compute distance matrix points itself
            return cdist(points, points, distance)
        else:
            # Compute distance between matrix and points
            return cdist(points, pt, distance)

    def _checkSpatialDataset(self, geodf):

        msg = ""
        flag = True

        # Check the type of the dataset obj
        if type(geodf) != geopd.geodataframe.GeoDataFrame:
            msg += "Type of dataset must be geopandas.geodataframe see: (lint to doc) \n"
            flag = False

        # Check the crs[already done]
        if geodf.crs == None:
            msg += "Dataset CRS not found: (lint to doc) \n"
            flag = False

        # Check valid geometry [All line must be valid]
        mask = geodf.is_valid
        if not mask.all():
            msg += "Check the rows geometry: (.is_valid) \n"
            flag = False

        # Rename geometry
        if not ('geometry' in geodf):
            msg += "Rename the column with the geometry 'geometry' \n"
            flag = False

        geodf.set_geometry("geometry")

        # Create unique code (categorical) to idendify a point (from geometry)
        geometry_id = 'geometry_id'
        ct = pd.Categorical(geodf['geometry'],
                            categories=geodf.geometry.unique())
        geodf[geometry_id] = ct.codes

        # Check the geometry type (now just single resolution)
        mask = np.unique(geodf.geom_type)
        if not mask.shape == (1,):
            msg += "Just one spatial geometry is supported. Currently found geometries {maks} \n".format(
                maks=mask)
            flag = False

        return flag, msg, geometry_id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        description = """Dataset object 
----------------------------------
formula: {formula}
Time  column: {time_name}
Space column: {space_name}

# Space
- Crs: {crs_name}
- Geometry type: {ptType}
- Number of points: {N} (centroid)
- Box: {box}

# Time
- Number of timestamp: {T} 
- Timestamp: min={tmin}, max={tmax}

# Design matrix 
y: {y_shape}
y name : {y_name}

X: {x_shape}
X name: {x_name}
----------------------------------
""".format(formula=self.formula, time_name=self.time_col_name, space_name="geometry",
           crs_name=self.crs.name, ptType=np.unique(self.geometry), N=self.N, box=self.box,
           T=self.T, tmin=self.timestamps.min(), tmax=self.timestamps.max(),
           y_shape=self.y.shape,
           y_name=self._y_design_info.column_names,
           x_shape=self.X.shape,
           x_name=self._x_design_info.column_names
           )

        return description
