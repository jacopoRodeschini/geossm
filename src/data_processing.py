#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jacopo - Tommaso
@title: 
"""

import numpy as np
from patsy import ModelDesc, dmatrices, build_design_matrices

import geopandas as geopd
import warning

# %%


class DesingMatrices:

    def __init__(self, y: np.array, X: np.array, formula: str,
                 csr, geometry, timestamps):

        # main attribute
        self.y = y
        self.X = X
        self.formula = formula
        self.geometry = geometry
        self.timestamps = timestamps

        self.N, self.b, self.T = X.shape

    def _parse_formula(self,):

        terms = ModelDesc.from_formula(formula)

        return y_name.lhs_termlist[0].name(), X_name.rhs_termlist.name()


# %%

class data_preparation:

    def __init__(self, geodf: geopd.DataFrame, fomula: str):

        # pipeline steps
        # ...

        return DesingMatrices(y, X, formula, crs, geometry, timestamps)

    def _check(self, ):

    def _build(self, )


# %%
"""
class _DesingMatrices_geossm(DesingMatrices):
    
    def __init__(args*):
        super.__init__(self, args*)
"""
