import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preparation import data_preparation as dp
import geopandas as geopd
import pandas as pd
from shapely.geometry import Point

# Create a simple GeoDataFrame for testing
data = {
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6],
    'target': [7, 8, 9],
    'Time': pd.date_range('2023-01-01', periods=3),
    'geometry': [Point(1, 2), Point(2, 3), Point(3, 4)]
}
gdf = geopd.GeoDataFrame(data, crs="EPSG:4326")

formula = 'target ~ feature1 + feature2'

design_matrices = dp(gdf, formula)

assert design_matrices.y.shape == (3,)
assert design_matrices.X.shape == (3, 3)  # Including intercept
assert design_matrices.formula == formula
assert design_matrices.crs == "EPSG:4326"
assert len(design_matrices.geometry) == 3
