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

# Version handling
try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("geossm")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    # Python < 3.8
    __version__ = "1.1.1"

# Core utilities
from .data_preparation import DesignMatrices, data_preparation
from .utils import block_diag_3D, write, getHardware

# Submodules
from . import covmodel
from . import datasets
from . import ssm
from . import stmodel

# Import main classes for convenience
from .covmodel import spdeAppoxCov, FEMSolver
from .datasets import load_dataset, list_datasets
from .ssm import StateSpaceModel, StateSpaceResults
from .stmodel import (
    LRStateSpaceModel,
    Param,
    FitOptions,
    ModelParams,
    LRStateSpaceResults,
)

# Public API
__all__ = [
    # Version
    "__version__",
    # Core utilities
    "data_preparation",
    "DesignMatrices",
    "block_diag_3D",
    "write",
    "getHardware",
    # Submodules
    "covmodel",
    "datasets",
    "ssm",
    "stmodel",
    # Covariance models
    "spdeAppoxCov",
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
]
