# Single source of the package version: pyproject.toml reads it at build time
# (`[tool.setuptools.dynamic]`) and `geossm.__version__` re-exports it, so the
# two cannot drift (F29) even when installed metadata is stale (editable installs).
__version__ = "2.1.0"
