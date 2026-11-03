"""
GeoSSM: Geospatial State-Space Modeling Tools
==============================================

A Python package for geospatial state-space modeling with support for
spatio-temporal data analysis.

Main modules:
- covmodel: Covariance models and SPDE approximations
- datasets: Dataset loading utilities
- ssm: State-space models
- stmodel: Spatio-temporal models (Low-Rank State-Space Models)

Example:
    >>> from geossm import LRStateSpaceModel
    >>> from geossm.datasets import load_dataset
"""

from ._version import __version__

# Core utilities
from .data_preparation import DesignMatrices, DesignMatricesBuilder
from .utils import block_diag_3D, write, get_hardware

# Submodules
from . import covmodel
from . import datasets
from . import ssm
from . import stmodel

# Import main classes for convenience
from .covmodel import SPDEApproxCov, FEMSolver
from .datasets import load_dataset, list_datasets
from .ssm import StateSpaceModel, StateSpaceResults
from .stmodel import (
    LRStateSpaceModel,
    Param,
    FitOptions,
    ModelParams,
    LRStateSpaceResults,
    LRPredictionResults,
)

# Public API
__all__ = [
    # Version
    "__version__",
    # Core utilities
    "data_preparation",
    "DesignMatrices",
    "DesignMatricesBuilder",
    "block_diag_3D",
    "write",
    "get_hardware",
    # Submodules
    "covmodel",
    "datasets",
    "ssm",
    "stmodel",
    # Covariance models
    "SPDEApproxCov",
    "FEMSolver",
    # Datasets
    "load_dataset",
    "list_datasets",
    # State-space models
    "StateSpaceModel",
    "StateSpaceResults",
    # Spatio-temporal models
    "LRStateSpaceModel",
    "Param",
    "FitOptions",
    "ModelParams",
    "LRStateSpaceResults",
    "LRPredictionResults",
]
