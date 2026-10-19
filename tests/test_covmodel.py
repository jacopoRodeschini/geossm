"""`FEMSolver` finite-element matrices/basis and `spdeAppoxCov` precision."""

import pickle

import numpy as np
import pytest
import scipy.sparse as sp
from shapely.geometry import Polygon

from geossm.covmodel import FEMSolver, spdeAppoxCov
from tests._known_bugs import known_bug
from tests._synthetic import make_square_mesh

LO, HI = -3.0, 13.0
INNER = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


@pytest.fixture(scope="module")
def mesh():
    return make_square_mesh(LO, HI, mesh_size=1.5)


@pytest.fixture(scope="module")
def fem(mesh):
    return FEMSolver(mesh, domain=[INNER], verbose=False, stats=False)


@pytest.fixture
def cov(mesh):
    return spdeAppoxCov(latlon=False, rescale=0.8, verbose=False).setup(mesh, domain=[INNER], stats=False)


class TestFEMSolver:
    def test_lumped_mass_is_positive_diagonal_and_sums_to_area(self, fem):
        C = fem.mass
        assert sp.issparse(C) and C.shape == (fem.nvertex, fem.nvertex)
        assert (C - sp.diags(C.diagonal())).count_nonzero() == 0
        assert C.diagonal().min() > 0
        assert C.diagonal().sum() == pytest.approx((HI - LO) ** 2, rel=1e-10)

    def test_stiffness_symmetric_psd_with_constant_nullspace(self, fem):
        G = fem.stiff.toarray()
        np.testing.assert_allclose(G, G.T, atol=1e-12)
        np.testing.assert_allclose(G.sum(axis=1), 0.0, atol=1e-10)
        assert np.linalg.eigvalsh(G).min() > -1e-10

    def test_stiffness_energy_of_linear_function(self, fem):
        # For u = a x + b y, u' G u = |grad u|^2 * area exactly (P1 elements).
        v = fem.vertex[:, :2]
        u = 2.0 * v[:, 0] - 0.5 * v[:, 1]
        assert u @ fem.stiff @ u == pytest.approx((2.0**2 + 0.5**2) * (HI - LO) ** 2, rel=1e-8)

    def test_inner_vertices_are_inside_domain(self, fem):
        v = fem.vertex[fem.inner, :2]
        assert 0 < fem.n_inner_points < fem.nvertex
        assert np.all((v >= 0) & (v <= 10))

    def test_basis_partition_of_unity_and_linear_reproduction(self, fem):
        rng = np.random.default_rng(0)
        pts = rng.uniform(LO + 0.1, HI - 0.1, size=(50, 2))
        count, not_found, B = fem.getBasis(pts)
        assert count == 50 and len(not_found) == 0
        np.testing.assert_allclose(np.asarray(B.sum(axis=1)).ravel(), 1.0, atol=1e-10)
        f = lambda xy: 1.0 + 2.0 * xy[:, 0] - 3.0 * xy[:, 1]  # noqa: E731
        np.testing.assert_allclose(B @ f(fem.vertex[:, :2]), f(pts), atol=1e-8)

    def test_covers_domain(self, fem):
        assert fem.covers_domain()


class TestSpdePrecision:
    def test_precision_symmetric_positive_definite(self, cov):
        Q = cov.precision().toarray()
        np.testing.assert_allclose(Q, Q.T, atol=1e-12)
        np.linalg.cholesky(Q)

    def test_precision_formula(self, cov):
        # Q = sigma2k * K C^-1 K with K = kappa^2 C + G, sigma2k = 1 / (4 pi kappa^2)
        C, G, k = cov.fem_solver.mass, cov.fem_solver.stiff, 0.8
        K = (k**2 * C + G).toarray()
        expected = K @ np.diag(1 / C.diagonal()) @ K / (4 * np.pi * k**2)
        np.testing.assert_allclose(cov.precision().toarray(), expected, rtol=1e-10, atol=1e-14)

    def test_marginal_precision_is_schur_complement(self, cov):
        Q = cov.precision().toarray()
        inner = cov.fem_solver.inner
        Qm = cov.precision(marginal=True)
        assert Qm.shape == (inner.sum(), inner.sum())
        # inverse of the marginal precision = inner block of the full covariance
        np.testing.assert_allclose(np.linalg.inv(Qm), np.linalg.inv(Q)[np.ix_(inner, inner)], rtol=1e-6)

    def test_rescale_argument_changes_precision(self, cov):
        a = cov.precision(rescale=0.5).toarray()
        b = cov.precision(rescale=1.5).toarray()
        assert not np.allclose(a, b)

    @known_bug("F05", "precision() hard-codes nu=1; nu only changes the reported range")
    def test_nu_is_honoured_or_rejected(self, mesh):
        try:
            c1 = spdeAppoxCov(latlon=False, nu=1, verbose=False).setup(mesh, domain=[INNER], stats=False)
            c2 = spdeAppoxCov(latlon=False, nu=2.5, verbose=False).setup(mesh, domain=[INNER], stats=False)
        except ValueError:
            return  # rejecting unsupported smoothness is an acceptable fix
        assert not np.allclose(c1.precision().toarray(), c2.precision().toarray())

    @known_bug("F06", "unpickling re-runs setup() without the domain (all vertices inner)")
    def test_pickle_roundtrip_preserves_model(self, cov):
        restored = pickle.loads(pickle.dumps(cov))
        np.testing.assert_array_equal(restored.fem_solver.inner, cov.fem_solver.inner)
        np.testing.assert_allclose(restored.precision(marginal=True), cov.precision(marginal=True))
