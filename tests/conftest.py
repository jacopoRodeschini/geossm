"""Shared pytest configuration and fixtures for the geossm test suite.

Design rules for the suite (see docs/design_audit/design_findings.md, F28):

* Tests run on CPU and are deterministic; GPU-only tests skip themselves when
  no GPU is present (they are reported as skipped, never as passed).
* No test may leak global JAX configuration: `jax_enable_x64` is restored
  after every test (the package itself flips it, finding F03).
* Confirmed-but-unfixed defects are pinned with ``known_bug`` (a strict
  xfail): once a defect is fixed the test XPASSes, the run fails, and the
  marker must be removed in the same change.
"""

import os

# Must be set before JAX initialises its GPU backend: otherwise the first
# model construction preallocates 75 % of GPU memory, even for CPU-only tests
# (finding F04). setdefault keeps any value the caller exported explicitly.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests._reference import make_problem  # noqa: E402

# Local research script with a machine-specific data path and a GPU assert;
# it is not a test (kept in the folder, excluded from collection).
collect_ignore = ["test_mesh_pen.py"]

CPU_DEVICE = jax.devices("cpu")[0]


@pytest.fixture(autouse=True)
def _restore_jax_x64():
    """Restore the global x64 flag after each test (guards against F03 and
    against tests that enable it without restoring)."""
    before = jax.config.jax_enable_x64
    yield
    if jax.config.jax_enable_x64 != before:
        jax.config.update("jax_enable_x64", before)


@pytest.fixture(autouse=True)
def _default_cpu_device():
    """Run arrays/kernels on CPU unless a test explicitly selects a device."""
    with jax.default_device(CPU_DEVICE):
        yield


@pytest.fixture
def x64():
    """Enable float64 for the duration of one test (restored afterwards)."""
    jax.config.update("jax_enable_x64", True)
    yield


@pytest.fixture
def problem():
    """Small well-conditioned SSM problem without missing data (NumPy float64)."""
    return make_problem(seed=0, p=6, q=3, T=10, b=2)


@pytest.fixture
def missing_mask(problem):
    """Deterministic missing-data pattern: scattered entries, one row
    partially missing, and one time step entirely missing."""
    mask = np.zeros_like(problem.y, dtype=bool)
    mask[[0, 1, 4], [1, 3, 6]] = True
    mask[2, :4] = True
    mask[:, 8] = True
    return mask


@pytest.fixture
def problem_missing(problem, missing_mask):
    return problem.with_missing(missing_mask)
