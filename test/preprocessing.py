from shapely.geometry import Point
import pandas as pd
import geopandas as geopd
from data_preparation import data_preparation as dp
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('geossm/src/data_preparation'))


# Create a simple GeoDataFrame for testing
data = {
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6],
    'target': [7, 8, 9],
    'Time': [pd.to_datetime('2023-01-01') for _ in range(3)],
    'geometry': [Point(1., 2.), Point(2., 3.), Point(3., 4.)]
}


gdf = geopd.GeoDataFrame(data, crs="EPSG:4326")

formula = 'target ~ feature1 + feature2'

# %% Create the dataset 


dataset = dp(gdf, formula)()

# %% Test the following assertions


assert dataset.y.shape == (3, 1)
assert dataset.X.shape == (3, 3, 1)  # Including intercept
assert dataset.formula == formula
assert dataset.crs == "EPSG:4326"
assert len(dataset.geometry) == 1
