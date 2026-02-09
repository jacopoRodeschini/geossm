"""
Adapter scaffolding making the project's StateSpaceModel usable with
statsmodels' MLEModel API.
"""

import numpy as np
from scipy.spatial import ConvexHull

try:
    from statsmodels.tsa.statespace.mlemodel import MLEModel
except Exception:
    # Minimal fallback base class to allow importing this module when
    # statsmodels is not installed. This fallback does not implement
    # any optimization or fit behavior.
    class MLEModel(object):
        def __init__(self, endog=None, exog=None, **kwargs):
            self.endog = endog
            self.exog = exog

# inmport the state space model
from ssm import StateSpaceModel
from covmodel import spdeAppoxCov
import geopandas

# %% Low Rank State-Space Model adapter to statsmodels MLEModel API


class LRStateSpaceModel:

    def __init__(self, df, formulas, domain=None):

        self.df = df
        self.formulas = formulas

        # Compute the design matrices
        self.nvar, self.points, self.gridList, self.ndim, self.pdim, self.block_p, self.T = self._buildObservationGrid(
            df, formulas)

        self.yTrain, self.Xbeta_train = self._buildDesignMatrix(self)

        # Check the domain
        flag, msg = self._checkDomain(domain)
        if flag:
            raise ValueError(msg)
        else:
            self.domain = self._setDomain(domain)

    def setup(self, meshio):
        # create the covariance model
        cov_matern = spdeAppoxCov(
            [domain], latlon=False, nu=1, var=1, rescale=1)

    def _setdomasin(self, polygon):
        if polygon is None:
            polygon = [ConvexHull(pts) for pts in self.points]

        return polygon

    def _checkDomain(self, domain):

        flag = False
        msg = ""

        if domain is not None:
            if not isinstance(domain, (list, tuple)):
                raise TypeError("domain must be a list of Polygon objects")
            for poly in domain:
                if not isinstance(poly, Polygon):
                    flag = True
                    msg = f"Each domain element must be a shapely Polygon, got {type(poly).__name__}"

        return flag, msg

    def _buildObservationGrid(self, df, formulas):

        nvar = len(formula)  # numer of the response variable
        gridList = [grid(df, f) for f in formulas]

        T = [gr.T for gr in gridList]
        points = [gr.points for gr in gridList]

        # get dimnesion of each grid
        ndim = [grid.N for grid in gridList]
        block = np.hstack((0, np.cumsum(ndim)))

        return nvar, points, gridList, ndim, block[-1], block, T

    def _buildDesignMatrix(self, ):

        Ylist_original = [grid.y for grid in self.gridList]

        # applay the log transofrmation (natural log) [positive prediction]
        Ylist = Ylist_original
        # Ylist = [] # Ylist_original
        # for yi in Ylist_original:
        #     yi[yi <= 0.5] = np.nan
        #     Ylist.append(np.log(yi))

        # X - Fixed effect design matrix -> 3D block diag - [N x beta x T]
        XBeta_list = [grid.X for grid in self.gridList]

        # points_train = [pt[index, :] for pt, index in zip(points, itrain)]
        # points_test = [pt[index, :] for pt, index in zip(points, itest)]

        # Y_train_list = [yi[index, :] for yi, index in zip(Ylist, itrain)]
        # Xbeta_train_list = [xi[index, :, :] for xi, index in zip(Xlist, itrain)]

        # Y_test_list = [yi[index, :] for yi, index in zip(Ylist, itest)]
        # Xbeta_test_list = [xi[index, :, :] for xi, index in zip(Xlist, itest)]

        yTrain = jnp.vstack(Ylist)
        Xbeta_train = block_diag_3D(Xbeta_list)

        # Y_test = np.vstack(Y_test_list)
        # Xbeta_test = block_diag_3D(Xbeta_test_list)

    return yTrain, Xbeta_train
