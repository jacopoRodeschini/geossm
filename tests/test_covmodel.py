"""`FEMSolver` finite-element matrices/basis and `SPDEApproxCov` precision."""

import pickle

import numpy as np
import pytest
import scipy.sparse as sp
from shapely.geometry import Polygon

from geossm.covmodel import FEMSolver, SPDEApproxCov
from tests._known_bugs import known_bug
from tests._printing import assert_printing_is_side_effect_free
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
    return SPDEApproxCov(latlon=False, rescale=0.8, verbose=False).setup(mesh, domain=[INNER], stats=False)


def test_printing_has_no_side_effects(fem, cov):
    assert_printing_is_side_effect_free(fem)
    assert_printing_is_side_effect_free(cov)


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
        count, not_found, B = fem.eval_basis(pts)
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

    def test_precision_with_rescale_does_not_change_the_object(self, cov):
        # Section D: precision(rescale=x) used to set cov.rescale = x on an
        # object shared with fitted results.
        r0 = cov.rescale
        Q = cov.precision(rescale=2.0 * r0, marginal=True)
        assert cov.rescale == r0
        cov.rescale = 2.0 * r0
        np.testing.assert_array_equal(Q, cov.precision(marginal=True))
        cov.rescale = r0

    @pytest.mark.parametrize("kw", [{"nu": 2.5}, {"nu": 0.5}, {"var": 2.0}], ids=["nu2.5", "nu0.5", "var2"])
    def test_unsupported_nu_and_var_are_rejected(self, kw):
        # F05 / Section C: precision() is the nu = 1, unit-variance SPDE; other
        # values used to be silently ignored.
        with pytest.raises(ValueError, match="nu and var must be 1"):
            SPDEApproxCov(latlon=False, verbose=False, **kw)

    @pytest.mark.parametrize("latlon", [False, True], ids=["planar", "latlon"])
    def test_inherited_matern_describes_the_spde_field(self, latlon):
        # The inherited gstools Matern methods describe the nu = 1 field whose
        # SPDE precision() approximates, with kappa = rescale (Section C, C3).
        from scipy.special import kv
        cov = SPDEApproxCov(latlon=latlon, rescale=2.0, verbose=False)
        h = np.array([0.1, 0.5, 1.0, 2.0])
        r = np.deg2rad(h) * cov.geo_scale if latlon else h
        np.testing.assert_allclose(cov.covariance(r), 2.0 * h * kv(1, 2.0 * h), rtol=1e-6)

    def test_pickle_roundtrip_preserves_model(self, cov):
        # F06: unpickling re-ran setup() without the domain (all vertices inner).
        restored = pickle.loads(pickle.dumps(cov))
        np.testing.assert_array_equal(restored.fem_solver.inner, cov.fem_solver.inner)
        np.testing.assert_allclose(restored.precision(marginal=True), cov.precision(marginal=True))


class TestLatentInterface:
    """Interface between SPDEApproxCov and the spatio-temporal model (Section B,
    P3): what the model used to read from `cov.fem_solver` directly."""

    def test_latent_basis_matches_the_fem_basis_on_inner_nodes(self, cov):
        rng = np.random.default_rng(0)
        pts = np.vstack([rng.uniform(0, 10, size=(30, 2)), [[50.0, 50.0]]])  # last site: outside the mesh
        B = cov.latent_basis(pts)
        _, _, H = cov.fem_solver.eval_basis(pts)
        H_inner = H[:, cov.fem_solver.inner].toarray()
        # previous model code: multiply each row by the inverse of its sum
        sums = H_inner.sum(axis=1)
        inv = np.divide(1.0, sums, out=np.zeros_like(sums), where=sums != 0)
        expected = inv[:, None] * H_inner
        assert sp.issparse(B) and B.shape == (len(pts), cov.n_latent)
        np.testing.assert_array_equal(B.toarray(), expected)
        # Rows sum to 1 where the site has inner support, else 0: outside the
        # mesh (last site), or in an element whose nodes are all outer, as for
        # sites just inside the domain boundary (F51).
        has_inner = H_inner.sum(axis=1) > 0
        np.testing.assert_allclose(B.toarray().sum(axis=1), has_inner.astype(float))
        assert not has_inner[-1] and has_inner[:-1].sum() >= 25

    def test_fem_arrays_and_sizes(self, cov):
        mass, stiff, inner = cov.fem_arrays()
        fem = cov.fem_solver
        assert mass is fem.mass and stiff is fem.stiff
        assert inner.dtype == bool and np.array_equal(inner, fem.inner)
        assert cov.n_latent == int(inner.sum()) and isinstance(cov.n_latent, int)
        assert cov.box == fem.box

    def test_interface_requires_setup(self):
        cov = SPDEApproxCov(latlon=False, verbose=False)
        for get in (lambda: cov.n_latent, lambda: cov.box, cov.fem_arrays, lambda: cov.latent_basis([[0.0, 0.0]])):
            with pytest.raises(RuntimeError, match="setup"):
                get()


def test_model_uses_only_the_latent_interface():
    # LRStateSpaceModel no longer reaches into FEMSolver (Section B, P3).
    import inspect
    from geossm.stmodel import lrssm
    assert "fem_solver" not in inspect.getsource(lrssm)


class TestPresentation:
    """Section B, P8: one log format, plotting outside the numerical class,
    mesh statistics computed on demand."""

    def test_mesh_statistics_are_computed_on_first_access(self, mesh):
        fem = FEMSolver(mesh, domain=[INNER], verbose=False)
        assert fem._angles is None and fem._areas is None  # not computed at construction
        angles, areas = fem.compute_stats()
        np.testing.assert_array_equal(fem.angles, angles)
        np.testing.assert_array_equal(fem.areas, areas)
        eager = FEMSolver(mesh, domain=[INNER], verbose=False, stats=True)
        assert eager._angles is not None
        np.testing.assert_array_equal(eager.angles, angles)
        assert "N/A" not in str(FEMSolver(mesh, domain=[INNER], verbose=False).summary())

    def test_plot_mesh_draws_with_the_plotting_function(self, fem):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from geossm.covmodel._plotting import plot_mesh
        ax = fem.plot_mesh(title="mesh")
        assert ax.get_title() == "mesh" and len(ax.lines) + len(ax.collections) > 0
        ax2 = plot_mesh(fem, title="mesh")
        assert len(ax2.lines) == len(ax.lines) and len(ax2.collections) == len(ax.collections)
        plt.close("all")

    def test_print_info_format_is_shared(self, fem, cov, capsys):
        import re
        from geossm import DesignMatricesBuilder
        from geossm.stmodel import LRStateSpaceModel
        printers = [fem.print_info, cov.print_info, DesignMatricesBuilder.print_info,
                    lambda m: LRStateSpaceModel.print_info(None, m)]
        for p in printers:
            p("hello")
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 4
        assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - hello", ln) for ln in lines)
