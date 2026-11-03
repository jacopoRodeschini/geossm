"""Mesh builders `buildMesh2d` and `buildMeshGrid2d` (Section Q): vertex budget,
domain coverage, anchoring, argument checks, and use by `SPDEApproxCov`."""

import warnings

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

from geossm.covmodel import SPDEApproxCov, buildMesh2d, buildMeshGrid2d

DOMAIN = box(0, 0, 10, 10)


@pytest.fixture(scope="module")
def points():
    """Clustered sites in DOMAIN: dense in the lower-left quarter."""
    rng = np.random.default_rng(0)
    dense = rng.uniform(0.5, 4.5, size=(60, 2))
    sparse = rng.uniform(0.5, 9.5, size=(40, 2))
    return np.vstack([dense, sparse])


def _triangles(mesh):
    return mesh.cells_dict["triangle"]


def _inside(mesh, polygon):
    return int(shapely.contains_xy(polygon, mesh.points[:, 0], mesh.points[:, 1]).sum())


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # mesh-quality warnings are tested separately
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def mesh_info(points):
    return _quiet(buildMesh2d, points, n_vertices=60, domain=DOMAIN, return_info=True)


class TestBuildMesh2d:
    def test_vertex_budget_inside_the_domain(self, mesh_info):
        mesh, _, info = mesh_info
        assert info["target_n"] == 60 and info["n_inside"] == _inside(mesh, DOMAIN)
        assert abs(info["n_inside"] - 60) <= max(1, 0.02 * 60) or len(info["history"]) == 10

    def test_lowrank_sets_the_budget_from_the_number_of_points(self, points):
        _, _, info = _quiet(buildMesh2d, points, lowrank=0.5, domain=DOMAIN, return_info=True)
        assert info["target_n"] == round(0.5 * len(points))

    def test_mesh_covers_the_domain_with_a_buffer(self, mesh_info):
        mesh, hull, _ = mesh_info
        mesh_hull = shapely.MultiPoint(mesh.points[:, :2]).convex_hull
        assert mesh_hull.covers(DOMAIN) and hull.covers(DOMAIN) and hull.area > DOMAIN.area

    def test_valid_triangulation(self, mesh_info):
        mesh, _, _ = mesh_info
        tri = _triangles(mesh)
        assert tri.min() >= 0 and tri.max() < len(mesh.points)
        p = mesh.points[:, :2]
        a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        assert area.min() > 0
        hull = shapely.MultiPoint(p).convex_hull
        assert area.sum() == pytest.approx(hull.area, rel=1e-6)  # no overlaps or holes

    def test_denser_where_the_points_are(self, mesh_info):
        # the size field follows the intensity of the points
        mesh, _, _ = mesh_info
        assert _inside(mesh, box(0, 0, 5, 5)) > _inside(mesh, box(5, 5, 10, 10))

    def test_anchors_are_mesh_vertices(self, points):
        mesh, _, info = _quiet(buildMesh2d, points, n_vertices=60, domain=DOMAIN, anchor_spacing=0.8,
                               return_info=True)
        anchors = points[info["anchors"]]
        assert len(anchors) > 0
        d = np.min(np.linalg.norm(mesh.points[None, :, :2] - anchors[:, None, :], axis=2), axis=1)
        assert d.max() < 1e-8

    def test_multipolygon_domain_counts_vertices_in_its_parts(self, points):
        domain = MultiPolygon([box(0, 0, 4, 10), box(6, 0, 10, 10)])
        mesh, _, info = _quiet(buildMesh2d, points, n_vertices=50, domain=domain, return_info=True)
        assert info["n_inside"] == _inside(mesh, domain)

    @pytest.mark.parametrize("kw, match", [
        (dict(), "exactly one of"),
        (dict(lowrank=0.5, n_vertices=10), "exactly one of"),
        (dict(lowrank=1.5), "lowrank"),
        (dict(n_vertices=20, eps=0.0), "eps"),
        (dict(n_vertices=20, gradation=0.0), "gradation"),
        (dict(n_vertices=20, anchor_spacing=-1.0), "anchor_spacing"),
    ])
    def test_invalid_arguments_raise(self, points, kw, match):
        with pytest.raises(ValueError, match=match):
            buildMesh2d(points, domain=DOMAIN, **kw)

    def test_unreachable_angle_target_warns(self, points):
        with pytest.warns(UserWarning, match="minimum interior angle"):
            buildMesh2d(points, n_vertices=40, domain=DOMAIN, min_angle=59.9)

    def test_mesh_sets_up_the_spde(self, mesh_info, points):
        mesh, _, _ = mesh_info
        cov = SPDEApproxCov(latlon=False).setup(mesh, domain=[DOMAIN])
        assert cov.n_latent == _inside(mesh, DOMAIN)
        H = cov.latent_basis(points)
        np.testing.assert_allclose(np.asarray(H.sum(axis=1)).ravel(), 1.0, atol=1e-10)


class TestBuildMeshGrid2d:
    def test_explicit_resolution(self):
        mesh, hull = buildMeshGrid2d(domain=DOMAIN, nx=12, ny=12)
        assert len(mesh.points) <= 12 * 12
        assert shapely.MultiPoint(mesh.points[:, :2]).convex_hull.covers(DOMAIN)
        assert hull.covers(DOMAIN)

    def test_min_degree(self):
        mesh, _ = buildMeshGrid2d(domain=Polygon([(0, 0), (10, 0), (0, 10)]), nx=15, ny=15, min_degree=3)
        tri = _triangles(mesh)
        edges = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
        degree = np.bincount(edges.ravel(), minlength=len(mesh.points))
        assert degree.min() >= 3 and set(np.unique(tri)) == set(range(len(mesh.points)))

    def test_lowrank_returns_the_closest_lattice(self, points):
        # The lattice counts reachable here are ..., 34, 47, 62, ...: the
        # target 60 cannot be met within tol. The closest (62) is returned,
        # with a warning (the last one tried, 47, was returned before).
        with pytest.warns(UserWarning, match="closest regular lattice has 62"):
            mesh, _ = buildMeshGrid2d(points=points, domain=DOMAIN, lowrank=0.6)
        assert len(mesh.points) == 62

    def test_lowrank_within_tolerance_is_silent(self, points):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mesh, _ = buildMeshGrid2d(points=points, domain=DOMAIN, lowrank=0.47)
        assert abs(len(mesh.points) - 47) <= 1

    @pytest.mark.parametrize("kw, match", [
        (dict(), "points"),
        (dict(domain=DOMAIN), "nx"),
        (dict(domain=DOMAIN, nx=10), "together"),
        (dict(domain=DOMAIN, lowrank=0.5), "points"),
    ])
    def test_invalid_arguments_raise(self, kw, match):
        with pytest.raises(ValueError, match=match):
            buildMeshGrid2d(**kw)
