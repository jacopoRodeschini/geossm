#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:23:19 2024

@author: jacopo
"""

import numpy as np

from scipy.spatial.distance import cdist
from patsy import ModelDesc, dmatrices, build_design_matrices
import pandas as pd

import geopandas


class grid:
    def __init__(self, dataset, formula):

        # Do preliminary check of the dataset
        flag, msg = self._checkSpatialDataset(dataset)
        if not flag:
            raise ValueError(msg)

        flag, msg, self.time_col_name = self._checkTimeDataset(dataset)
        if not flag:
            raise ValueError(msg)

        # check formula (NOW NOT IMPLEMENTED) [return true]
        flag, msg, self.response_name = self._checkFormula(formula)
        if not flag:
            raise ValueError(msg)

        self.formula = formula

        # Create unique code (categorical) to idendify a point (from geometry)
        self.geometry_id = 'geometry_id'
        ct = pd.Categorical(dataset['geometry'],
                            categories=dataset.geometry.unique())
        dataset[self.geometry_id] = ct.codes

        # Check the time column (timestmap / delta / complete - NAN)
        flag, msg = self._checkTimeColumn(
            dataset, self.response_name, self.geometry_id, self.time_col_name)

        if not flag:
            raise ValueError(msg)

        # Create new dataset & compute the designed matrix
        self.df, self.idPoint, self.y, self._y_design_info, self.X, self._x_design_info, self.N, self.T = self._computedesignMatrix(
            dataset, self.geometry_id, self.time_col_name, self.response_name, formula)

        # Objcet geometry metrics attributes
        self.box = self.df.total_bounds
        self.geometry_type = self.df.geom_type
        self.crs = self.df.crs

        # Design matrix
        self.eigvals = 0.00
        self.condition_number = 0.00
        self.intercept = True  # default

        # Get the points of the response variable and distace
        self.points, self.distance = self._getPoints(
            self.df, self.geometry_id, self.time_col_name)

    def __str__(self):
        return """Grid object ----------------------
formula: {formula}
Time name: {time_name}

# Space
- Crs: {crs_name}
- Geometry type: {ptType}
- Geometry name: {geo_name} 
- Number of points: {N} (centroid)
- Box: {box}
- Dist: min={dmin}, median={dmed}, max={dmax}

# Time
- Number of timestamp: {T} 
- Timestamp: min={tmin}, max={tmax}

# Design matrix 
y: {y_shape}
y name : {y_name}

X: {x_shape}
X name: {x_name}
----------------------------------
""".format(formula=self.formula, time_name=self.time_col_name,
           crs_name=self.crs.name, ptType=np.unique(self.geometry_type), geo_name=self.df.geometry.name, N=self.N, box=self.box,
           dmin=np.round(self.distance.min(), 2),
           dmed=np.round(np.median(self.distance), 2),
           dmax=np.round(self.distance.max(), 2),
           T=self.T, tmin=self.df[self.time_col_name].min(), tmax=self.df[self.time_col_name].max(),
           y_shape=self.y.shape,
           y_name=self._y_design_info.column_names,
           x_shape=self.X.shape,
           x_name=self._x_design_info.column_names
           )

    def _computedesignMatrix(self, df, geometry_id, time_col_name, response_name, formula=None, terms=None):

        flag = True
        msg = ""

        if ((formula is None) and (terms is None)) or ((formula is not None) and (terms is not None)):
            fag = False
            msg += "Formula or terms must be provided"

        # sort the dataset by time
        # df[geometry_id] = df[geometry_id].astype('category')
        # sdf[time_col_name] = df[time_col_name].astype('category')
        df = df.sort_values([time_col_name, geometry_id])

        # take just the unique row
        df = df.drop_duplicates(subset=[geometry_id, time_col_name])

        # Create new dataset starting from formula
        if formula is not None:

            time = np.unique(df[time_col_name])
            T = time.shape[0]

            # TODO: not all point measure the variable (remove this point)
            # Check the ID statin with measure always nan (and merge same sites)

            group_cols = [geometry_id]
            stp = df.groupby(group_cols, observed=True).agg({
                response_name: lambda x: np.nansum(x)
            }).reset_index()

            idS = stp[stp[response_name] > 0].index

            df = df[df.geometry_id.isin(idS)]

            # check the number of point available
            point = df.geometry.unique()
            N = point.shape[0]

            # Convert Nan to inf (this becouse the dmatrices remove nan -> we have to keep this values)
            df.loc[df[response_name].isna(), response_name] = np.inf

            ytemp, Xtemp = dmatrices(
                formula, data=df, NA_action='raise', return_type='matrix')

            # replace inf with nan
            ytemp[np.isinf(ytemp)] = np.nan

            # just reshape:
            # response variable: [N x T]
            # Covariate variable: [N x P x T]
            y = ytemp.reshape(T, N).T

            Xbeta = np.zeros((N, Xtemp.shape[1], T))

            for i in range(0, Xtemp.shape[1]):
                Xbeta[:, i, :] = Xtemp[:, i].reshape(T, 1, N).T.squeeze(axis=1)

            return df, point, y, ytemp.design_info, Xbeta, Xtemp.design_info, N, T

    def _checkFormula(self, formula):

        # TODO: check the fromula parser
        flag = True
        msg = ""

        m = ModelDesc.from_formula(formula)

        return True, msg, m.lhs_termlist[0].name()

    def _checkTimeColumn(self, df, response_name, geometry_id, time_col_name):

        # Check foreach geometry_id the timeseries lenght, start/end date, the delta time.
        flag = True
        msg = ""

        return flag, msg

    def _checkTimeDataset(self, df):
        msg = ""
        flag = True
        time_col_name = None

        # check if time is in dataset
        if not 'Time' in df:

            # check other columns with datatime dtype
            time_col = [
                col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

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

    def _getPoints(self, df, geometry_id, time_col_name):

        # pojected geometry to_crs() for accurete results
        uni, idx = np.unique(df[geometry_id], return_index=True)

        # Extract the centroid by the geometry
        centroid = df.iloc[idx].geometry.centroid

        # Create a point vector (x, y)
        pts = np.stack((centroid.x, centroid.y), axis=1)

        # Compute the distance
        dist = self._computeDistance(pts)

        return pts, dist

    def _computeDistance(self, points, pt=None, distance='euclidean'):

        if pt is None:

            # Compute distance matrix points itself
            return cdist(points, points, distance)
        else:
            # Compute distance between matrix and points
            return cdist(points, pt, distance)

    def _checkSpatialDataset(self, df):

        msg = ""
        flag = True

        # Check the type of the dataset obj
        if type(df) != geopandas.geodataframe.GeoDataFrame:
            msg += "Type of dataset must be geopandas.geodataframe see: (lint to doc) \n"
            flag = False

        # Check the crs[already done]
        if df.crs == None:
            msg += "Dataset CRS not found: (lint to doc) \n"
            flag = False

        # Check valid geometry [All line must be valid]
        mask = df.is_valid
        if not mask.all():
            msg += "Check the rows geometry: (.is_valid) \n"
            flag = False

        # Rename geometry
        if not ('geometry' in df):
            msg += "Rename the column with the geometry 'geometry' \n"
            flag = False

        df.set_geometry("geometry")

        # Check the geometry type (now just single resolution)
        mask = np.unique(df.geom_type)
        if not mask.shape == (1,):
            msg += "Just one spatial geometry is supported. Currently found geometries {maks} \n".format(
                maks=mask)
            flag = False

        return flag, msg

    def write(self, filename, mode='a'):
        # write the class into a file
        s = "OBSERVATION GRID \n"
        s += self.__str__()

        with open(filename, mode) as f:
            f.write(s)

    def getObservarion(self):
        return self.y

    def getPoints(self):
        return self.points

    def getDistance(self):
        return self.distance

    def getDesignMatrix(self):
        if (self.y is not None) and (self.X is not None):
            return self.y, self._y_info.column_names, self.X, self._x_info.column_names
        elif (self.y is None):
            return self.X, self._x_info.column_names
        else:
            print("Error")
            return

    def isValid(self):
        # Boolean test to check if the grid obj is valid

        # Check if the distance matrx is computed
        if self.distance is None:
            return False

    @ property
    def dataset(self):
        return self._dataset

    def __repr__(self):
        return self.__str__()

    def __eq__(self, o):
        return True


"""
class gridList:
    def __init__(self):
        self.list = []
        
    def append(self, grid):
        
        if not type(grid) grid:
            return "The grid neet to be a gridd object"
        
        self.list.append(grid)

"""
