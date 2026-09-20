# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import datetime
import importlib.metadata

# -- Path setup --------------------------------------------------------------
#
# `geossm` uses a `src/` layout. Rather than manipulating `sys.path` (fragile,
# and it silently hides packaging/import errors that would also affect users),
# the docs build expects the package to be installed - e.g. via
# `pip install -e ".[docs]"` - in the environment running Sphinx, exactly as
# recommended for local development in CONTRIBUTING.md. If the import below
# fails, install the package first instead of adding path hacks here.
try:
    import geossm  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'geossm' package could not be imported. Install it (and its "
        "documentation dependencies) before building the docs, e.g.:\n\n"
        "    pip install -e '.[docs]'\n"
    ) from exc

# -- Project information ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "geossm"
author = "Jacopo Rodeschini"
copyright = f"{datetime.date.today().year}, {author}"

try:
    release = importlib.metadata.version("geossm")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    release = "0.0.0"
# Short X.Y version used in e.g. the sidebar.
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
}

# The master toctree document.
root_doc = "index"

# Warn about all references where the target cannot be found.
nitpicky = False

# `gstools` (an upstream dependency of `geossm.covmodel.spdeAppoxCov`, via
# `gstools.covmodel.Matern`) appends its own `CovModel.__doc__` onto every
# subclass's docstring at class-definition time (see
# `gstools.covmodel.base.CovModel.__init_subclass__`), including ":any:"
# roles (e.g. `KM_SCALE`, `hankel.SymmetricFourierTransform`) that are not
# resolvable without gstools' own Sphinx inventory. That text is kept -- it
# documents `spdeAppoxCov`'s inherited constructor arguments -- but the
# resulting unresolvable-target warnings are expected and suppressed here.
suppress_warnings = ["ref.any"]

# -- Autodoc / Autosummary -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
# `:inherited-members:` is intentionally NOT a global default here: for
# `LRStateSpaceModel`/`LRStateSpaceResults` (subclasses of geossm's own
# `StateSpaceModel`/`StateSpaceResults`) it is opted into per-class in
# `_templates/autosummary/class.rst`, since those inherited methods are
# part of the documented public workflow. Left off elsewhere, it would
# otherwise also pull in `spdeAppoxCov`'s ~90 inherited members from
# `gstools.covmodel.Matern`/`CovModel` (rotation/anisotropy/pykrige
# plumbing that isn't part of geossm's own API).
# Keep signatures readable: show type hints in the description, not stacked
# into an already-long `def name(...)` line.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
# `geossm` is built around dataclasses and Matern subclassing; keep the
# resolved MRO instead of hiding it.
autodoc_inherit_docstrings = True
autodoc_class_signature = "mixed"

# Third-party / heavy scientific & mesh dependencies that are only needed if
# every optional runtime extra (mesh generation, JAX GPU backends, ...) is
# installed. Mocking them keeps the docs buildable in slimmer environments
# (e.g. a docs-only CI job) without pulling in the full native toolchain.
autodoc_mock_imports = []

# -- Napoleon (NumPy-style docstrings) -----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -- Intersphinx ----------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
# Network access is not guaranteed in every build environment (e.g. offline
# CI runners); a missing inventory should not fail the build.
intersphinx_disabled_reftypes = ["*"]
intersphinx_timeout = 5

# -- Options for HTML output ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"{project} {version}"

html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "logo-dark.svg",
    "dark_logo": "logo-white.svg",
    "source_repository": "https://github.com/jacopoRodeschini/geossm",
    "source_branch": "develop",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jacopoRodeschini/geossm",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49"
                "-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
                "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 "
                "1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 "
                "3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 "
                '8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

# -- Options for autosummary/autodoc: custom templates -------------------------

autosummary_context = {}
