import numpy as np
from scipy.spatial.distance import cdist
from patsy import ModelDesc, dmatrices
import geopandas as geopd
import pandas as pd

class DesignMatrices:
    def __init__(self, y, X, formula, crs, geometry, timestamps, 
                 y_design_info, X_design_info, df, time_col_name, response_name, 
                 geometry_type, box, distance):
        self.y = y
        self.X = X
        self.formula = formula
        self.crs = crs
        self.geometry = geometry
        self.timestamps = timestamps
        self._y_design_info = y_design_info
        self._x_design_info = X_design_info
        self.df = df
        self.time_col_name = time_col_name
        self.response_name = response_name
        self.geometry_type = geometry_type
        self.box = box
        self.distance = distance
        self.N, self.b, self.T = X.shape

    def __str__(self):
        description = f"""Design Matrices object ----------------------
formula: {self.formula}
Time name: {self.time_col_name}

# Space
- Crs: {self.crs.name}
- Geometry type: {np.unique(self.geometry_type)}
- Geometry name: {self.df.geometry.name} 
- Number of points: {self.N} (centroid)
- Box: {self.box}
- Dist: min={np.round(self.distance.min(), 2)}, median={np.round(np.median(self.distance), 2)}, max={np.round(self.distance.max(), 2)}

# Time
- Number of timestamp: {self.T} 
- Timestamp: min={self.df[self.time_col_name].min()}, max={self.df[self.time_col_name].max()}

# Design matrix 
y: {self.y.shape}
y name : {self._y_design_info.column_names}

X: {self.X.shape}
X name: {self._x_design_info.column_names}
----------------------------------
"""
        return description

    def __repr__(self):
        return self.__str__()

class DataPreparation:
    def __init__(self, geodf: geopd.GeoDataFrame, formula: str):
        self.geodf = geodf
        self.formula = formula
        self._validate_inputs()
        self._build_design_matrices()

    def _validate_inputs(self):
        flag, msg = self._check_spatial_dataset(self.geodf)
        if not flag:
            raise ValueError(msg)
        flag, msg, self.time_col_name = self._check_time_dataset(self.geodf)
        if not flag:
            raise ValueError(msg)
        flag, msg, self.response_name = self._check_formula(self.formula)
        if not flag:
            raise ValueError(msg)
        self.geometry_id = 'geometry_id'
        ct = pd.Categorical(self.geodf['geometry'], categories=self.geodf.geometry.unique())
        self.geodf[self.geometry_id] = ct.codes
        flag, msg = self._check_time_column(self.geodf, self.response_name, self.geometry_id, self.time_col_name)
        if not flag:
            raise ValueError(msg)

    def _build_design_matrices(self):
        df, points, y, y_design_info, X, X_design_info, N, T = self._compute_design_matrix(
            self.geodf, self.geometry_id, self.time_col_name, self.response_name, self.formula)
        box = df.total_bounds
        geometry_type = df.geom_type
        crs = df.crs
        pts, distance = self._get_points(df, self.geometry_id)
        self.design_matrices = DesignMatrices(
            y, X, self.formula, crs, pts, np.unique(df[self.time_col_name]),
            y_design_info, X_design_info, df, self.time_col_name, self.response_name,
            geometry_type, box, distance
        )

    def _compute_design_matrix(self, df, geometry_id, time_col_name, response_name, formula):
        df = df.sort_values([time_col_name, geometry_id])
        df = df.drop_duplicates(subset=[geometry_id, time_col_name])
        time = np.unique(df[time_col_name])
        T = time.shape[0]
        group_cols = [geometry_id]
        stp = df.groupby(group_cols, observed=True).agg({
            response_name: lambda x: np.nansum(x)
        }).reset_index()
        idS = stp[stp[response_name] > 0].index
        df = df[df.geometry_id.isin(idS)]
        points = df.geometry.unique()
        N = points.shape[0]
        df.loc[df[response_name].isna(), response_name] = np.inf
        ytemp, Xtemp = dmatrices(formula, data=df, NA_action='raise', return_type='matrix')
        ytemp[np.isinf(ytemp)] = np.nan
        y = ytemp.reshape(T, N).T
        Xbeta = np.zeros((N, Xtemp.shape[1], T))
        for i in range(Xtemp.shape[1]):
            Xbeta[:, i, :] = Xtemp[:, i].reshape(T, 1, N).T.squeeze(axis=1)
        return df, points, y, ytemp.design_info, Xbeta, Xtemp.design_info, N, T

    def _check_formula(self, formula):
        m = ModelDesc.from_formula(formula)
        return True, "", m.lhs_termlist[0].name()

    def _check_time_column(self, df, response_name, geometry_id, time_col_name):
        return True, ""

    def _check_time_dataset(self, df):
        msg = ""
        flag = True
        time_col_name = None
        if 'Time' not in df:
            time_col = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if len(time_col) == 0:
                msg += "The 'Time' column not found \n"
                flag = False
            else:
                time_col_name = time_col[0]
                msg += f"'Time' column found: {time_col}\nKeeped 'Time' column: {time_col_name}\n"
        else:
            time_col_name = 'Time'
        return flag, msg, time_col_name

    def _get_points(self, df, geometry_id):
        uni, idx = np.unique(df[geometry_id], return_index=True)
        centroid = df.iloc[idx].geometry.centroid
        pts = np.stack((centroid.x, centroid.y), axis=1)
        dist = self._compute_distance(pts)
        return pts, dist

    def _compute_distance(self, points, pt=None, distance='euclidean'):
        if pt is None:
            return cdist(points, points, distance)
        else:
            return cdist(points, pt, distance)

    def _check_spatial_dataset(self, df):
        msg = ""
        flag = True
        if type(df) != geopd.geodataframe.GeoDataFrame:
            msg += "Type of dataset must be geopandas.geodataframe\n"
            flag = False
        if df.crs is None:
            msg += "Dataset CRS not found\n"
            flag = False
        mask = df.is_valid
        if not mask.all():
            msg += "Check the rows geometry: (.is_valid)\n"
            flag = False
        if 'geometry' not in df:
            msg += "Rename the column with the geometry 'geometry'\n"
            flag = False
        df.set_geometry("geometry")
        mask = np.unique(df.geom_type)
        if not mask.shape == (1,):
            msg += f"Just one spatial geometry is supported. Currently found geometries {mask}\n"
            flag = False
        return flag, msg
