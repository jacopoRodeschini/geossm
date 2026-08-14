import warnings

import gmsh
import meshio
import numpy as np
import pygmsh
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon


# % Utility functions

def _mesh_min_angle(mesh):
    """Smallest interior angle (degrees) over all triangles of a meshio mesh."""
    tri = mesh.cells_dict.get("triangle")
    if tri is None or len(tri) == 0:
        return np.nan

    pts = mesh.points[:, :2]
    p0, p1, p2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]

    angles = []
    for a, b, c in ((p0, p1, p2), (p1, p2, p0), (p2, p0, p1)):
        u, v = b - a, c - a
        cos_t = np.einsum("ij,ij->i", u, v) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

    return float(np.min(angles))


def buildMesh2d(
    points,
    boundary=None,
    max_edge=None,
    min_edge=None,
    offset=None,
    cutoff=None,
    min_angle=21.0,
    lowrank=None,
    density_neighbors=8,
    tol=0.02,
    max_iter=25,
):
    """
    Build a 2D triangular (gmsh/pygmsh) mesh around a set of observed
    locations, in the spirit of R-INLA's ``inla.mesh.2d()``.

    The mesh covers ``boundary`` (or the convex hull of ``points`` if not
    given), extended outward by ``offset`` to limit boundary effects, with
    triangle edges bounded by ``max_edge``/``min_edge``.

    Parameters
    ----------
    points : (n, 2) array_like
        Observed locations. Drive the default domain and, when `lowrank`
        is set, the local mesh density. Analogous to INLA's `loc`.
    boundary : shapely.geometry.Polygon, optional
        Domain the mesh must cover. Defaults to the convex hull of `points`.
    max_edge : float, optional
        Largest allowed triangle edge length. Defaults to 1/15 of the
        domain's bounding-box diagonal.
    min_edge : float, optional
        Smallest allowed triangle edge length. Defaults to `max_edge / 10`.
    offset : float, optional
        Buffer added around `boundary` so the mesh extends past the data.
        Defaults to `max_edge`.
    cutoff : float, optional
        Points closer together than `cutoff` are merged before building the
        density field. Defaults to `max_edge / 5`.
    min_angle : float, default 21.0
        Target minimum interior angle (degrees); drives extra mesh
        optimization passes. Mirrors INLA's `min.angle`. This is a soft
        target: no unstructured mesher guarantees it exactly.
    lowrank : float, optional
        Value in (0, 1]. When given, the local element size is rescaled by
        the density of `points` (finer where points are dense, coarser
        where they are sparse), so that the resulting mesh has approximately
        ``round(lowrank * len(points))`` vertices. Local density is a
        k-nearest-neighbor estimate (see `density_neighbors`) ranked by
        percentile, rather than a single-bandwidth KDE, so that separate
        clusters of comparable local density (e.g. several cities) are all
        refined even if one cluster has far more points overall.
    density_neighbors : int, default 8
        Number of neighbors used for the local density estimate that drives
        `lowrank`. Ignored if `lowrank` is not given.
    tol : float, default 0.02
        Relative tolerance on the vertex-count target used to stop the
        `lowrank` search.
    max_iter : int, default 25
        Maximum number of mesh (re)generations used by the `lowrank` search.

    Returns
    -------
    mesh : meshio.Mesh
    domain : shapely.geometry.Polygon
        The (buffered) domain the mesh was built over -- `boundary` (or its
        default) extended by `offset`.
    """
    points = np.asarray(points, dtype=float)[:, :2]
    n_input = len(points)

    if boundary is None:
        boundary = MultiPoint(points).convex_hull

    bbox = boundary.bounds
    diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))

    if max_edge is None:
        max_edge = diag / 15
    if min_edge is None:
        min_edge = max_edge / 10
    if cutoff is None:
        cutoff = max_edge / 5
    if offset is None:
        offset = max_edge

    # merge near-duplicate points within `cutoff`, as inla.mesh.2d does
    if cutoff > 0 and len(points) > 1:
        keep = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            if not keep[i]:
                continue
            d = np.linalg.norm(points[i] - points[i + 1:], axis=1)
            keep[i + 1:][d < cutoff] = False
        points = points[keep]

    domain = boundary.buffer(offset)
    if not isinstance(domain, Polygon):
        raise ValueError("boundary.buffer(offset) did not yield a single polygon.")
    coords = np.array(domain.simplify(offset * 0.25).exterior.coords[:-1])

    tree = None
    target_n = None
    k = None
    sorted_local_dens = None
    if lowrank is not None:
        if not (0 < lowrank <= 1):
            raise ValueError("lowrank must be in (0, 1].")
        k = max(1, min(density_neighbors, len(points) - 1))
        tree = cKDTree(points)
        # local density at each point, from the distance to its k-th
        # *other* point (hence k + 1 neighbors, dropping the self-match)
        dist, _ = tree.query(points, k=k + 1)
        dk = dist[:, k]
        local_dens = k / (np.pi * dk**2 + 1e-12)
        sorted_local_dens = np.sort(local_dens)
        target_n = max(3, round(lowrank * n_input))

    def _percentile_weight(x, y):
        # k-th nearest-neighbor local density at (x, y), converted to its
        # percentile rank among the points' own local densities: a cluster
        # that is *locally* as tight as the busiest cluster gets w close to
        # 1 even if it has far fewer points overall (unlike a fixed-
        # bandwidth KDE, whose single global bandwidth is dominated by
        # whichever cluster has the most points).
        dist, _ = tree.query([x, y], k=k)
        dk = dist if k == 1 else dist[-1]
        dens = k / (np.pi * dk**2 + 1e-12)
        return np.searchsorted(sorted_local_dens, dens) / len(sorted_local_dens)

    def _size_callback(scale):
        def callback(dim, tag, x, y, z, lc):
            w = _percentile_weight(x, y)  # 0 sparse .. 1 dense
            size = max_edge - w * (max_edge - min_edge)  # in [min_edge, max_edge]
            return float(size * scale)

        return callback

    def _generate(scale=1.0, opt_rounds=2):
        with pygmsh.occ.Geometry() as geom:
            surf = geom.add_polygon(coords, mesh_size=max_edge)
            geom.add_physical(surf, label="surface_domain")
            gmsh.model.occ.synchronize()

            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

            lo_bound = min_edge * min(scale, 1.0)
            hi_bound = max_edge * max(scale, 1.0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lo_bound)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hi_bound)

            if tree is not None:
                gmsh.model.mesh.setSizeCallback(_size_callback(scale))

            gmsh.model.mesh.generate(2)
            for _ in range(opt_rounds):
                gmsh.model.mesh.optimize("Laplace2D")
                gmsh.model.mesh.optimize("Netgen")

            mesh = geom.generate_mesh()

        return mesh

    if tree is None:
        return _generate(), domain

    # Bisection on a global size-scale factor to hit the target vertex count:
    # smaller scale -> smaller elements everywhere -> more vertices.
    lo, hi = 0.1, 10.0
    scale = lo
    mesh = _generate(scale=lo, opt_rounds=1)
    n_lo = len(mesh.points)

    if n_lo > target_n:
        for _ in range(max_iter):
            scale = 0.5 * (lo + hi)
            mesh = _generate(scale=scale, opt_rounds=1)
            n = len(mesh.points)
            if abs(n - target_n) <= max(1, tol * target_n):
                break
            if n > target_n:
                lo = scale
            else:
                hi = scale

    # final pass with more optimization rounds to push the min angle up
    mesh = _generate(scale=scale, opt_rounds=5)

    angle = _mesh_min_angle(mesh)
    if angle < min_angle:
        warnings.warn(
            f"buildMesh2d: reached {len(mesh.points)} vertices "
            f"(target {target_n}) but the minimum interior angle is "
            f"{angle:.1f} deg < min_angle={min_angle} deg. Consider a "
            "larger `lowrank`, or widen the min_edge/max_edge range."
        )

    return mesh, domain


def _prune_low_degree(vertices, triangles, min_degree, max_iter=50):
    """Drop triangles touching any vertex with fewer than `min_degree`
    distinct mesh edges, iterating since removal can lower a neighbor's
    degree in turn."""
    for _ in range(max_iter):
        edges = np.vstack(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        degree = np.bincount(edges.ravel(), minlength=len(vertices))
        bad = degree < min_degree
        if not bad.any():
            break
        triangles = triangles[~bad[triangles].any(axis=1)]
        if len(triangles) == 0:
            raise ValueError(
                "buildMeshGrid2d: the mesh vanished while enforcing "
                "min_degree; use a larger nx/ny (or lowrank), or a "
                "smaller min_degree."
            )
    return triangles


def buildMeshGrid2d(
    points=None,
    boundary=None,
    offset=None,
    nx=None,
    ny=None,
    lowrank=None,
    min_degree=3,
    tol=0.02,
    max_iter=25,
):
    """
    Build a regular (structured) 2D triangular mesh: a lattice of `nx` by
    `ny` vertices spanning the bounding box of `boundary` (or the convex
    hull of `points` if `boundary` is not given), extended by `offset`.
    Grid cells lying entirely outside the (buffered) domain are dropped, so
    the mesh follows the shape of `boundary` rather than just its bounding
    box. Any vertex left with fewer than `min_degree` mesh edges by that
    pruning (stray corners along the boundary "staircase") is removed
    together with its triangles, repeated until the whole mesh satisfies
    `min_degree`; vertices left unused are dropped from the result.

    The resolution can be set directly with `nx`/`ny`, or -- when `points`
    and `lowrank` are given instead -- solved for automatically so the
    final mesh has approximately ``round(lowrank * len(points))``
    vertices, mirroring `buildMesh2d`'s `lowrank` argument. Note the
    lattice stays regular: `lowrank` only controls its overall resolution
    here, not a spatially-varying density.

    Parameters
    ----------
    points : (n, 2) array_like, optional
        Observed locations. Used to derive `boundary` (as their convex
        hull) when `boundary` is not given, and as the basis for `lowrank`.
    boundary : shapely.geometry.Polygon, optional
        Domain the mesh must cover.
    offset : float, optional
        Buffer added around `boundary` so the lattice extends past the
        data. Defaults to 1/15 of the domain's bounding-box diagonal.
    nx, ny : int, optional
        Number of vertices along the x- and y-axis of the underlying
        rectangular lattice (before pruning). Give both together, or give
        `lowrank` instead.
    lowrank : float, optional
        Value in (0, 1]. When given (with `nx`/`ny` left unset), the
        resolution is found automatically so the mesh has approximately
        ``round(lowrank * len(points))`` vertices. Requires `points`.
    min_degree : int, default 3
        Minimum number of mesh edges required at every vertex.
    tol : float, default 0.02
        Relative tolerance on the vertex-count target used by the
        `lowrank` search.
    max_iter : int, default 25
        Maximum number of mesh (re)generations used by the `lowrank`
        search.

    Returns
    -------
    mesh : meshio.Mesh
    domain : shapely.geometry.Polygon
        The (buffered) domain the mesh was built over -- `boundary` (or its
        default) extended by `offset`.
    """
    if boundary is None:
        if points is None:
            raise ValueError("either `points` or `boundary` must be given.")
        points = np.asarray(points, dtype=float)[:, :2]
        boundary = MultiPoint(points).convex_hull
    elif points is not None:
        points = np.asarray(points, dtype=float)[:, :2]

    if offset is None:
        bbox = boundary.bounds
        diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        offset = diag / 15

    domain = boundary.buffer(offset) if offset else boundary
    if not isinstance(domain, Polygon):
        raise ValueError("boundary.buffer(offset) did not yield a single polygon.")

    minx, miny, maxx, maxy = domain.bounds
    aspect = (maxx - minx) / (maxy - miny)

    def _build(nx, ny):
        xs = np.linspace(minx, maxx, nx)
        ys = np.linspace(miny, maxy, ny)
        grid = np.column_stack([np.tile(xs, ny), np.repeat(ys, nx)])

        # two triangles per grid cell, both wound counter-clockwise
        idx = np.arange(nx * ny).reshape(ny, nx)
        a = idx[:-1, :-1].ravel()
        b = idx[:-1, 1:].ravel()
        c = idx[1:, :-1].ravel()
        d = idx[1:, 1:].ravel()
        triangles = np.vstack(
            [np.column_stack([a, b, d]), np.column_stack([a, d, c])]
        )

        # keep only cells whose centroid lies within the (buffered) domain
        centroids = grid[triangles].mean(axis=1)
        inside = shapely.contains_xy(domain, centroids[:, 0], centroids[:, 1])
        triangles = triangles[inside]

        triangles = _prune_low_degree(grid, triangles, min_degree)

        # drop vertices that are no longer referenced and remap indices
        used, triangles = np.unique(triangles, return_inverse=True)
        triangles = triangles.reshape(-1, 3)
        vertices = grid[used]

        return meshio.Mesh(
            points=np.column_stack([vertices, np.zeros(len(vertices))]),
            cells=[("triangle", triangles)],
        )

    if nx is not None or ny is not None:
        if nx is None or ny is None:
            raise ValueError("`nx` and `ny` must be given together.")
        return _build(nx, ny), domain

    if lowrank is None:
        raise ValueError("either `nx`/`ny` or `lowrank` must be given.")
    if points is None:
        raise ValueError("`points` is required when `lowrank` is given.")
    if not (0 < lowrank <= 1):
        raise ValueError("lowrank must be in (0, 1].")

    target_n = max(3, round(lowrank * len(points)))

    def _safe_build(res):
        # res ~ nx * ny before domain/degree pruning
        nx_ = max(2, round(np.sqrt(res * aspect)))
        ny_ = max(2, round(np.sqrt(res / aspect)))
        try:
            return _build(nx_, ny_)
        except ValueError:
            return None

    # Bisection on a lattice-resolution scalar to hit the target vertex
    # count: larger resolution -> more vertices (monotonic up to rounding
    # and boundary/degree pruning noise, which `tol` absorbs).
    lo, hi = 4.0, max(4.0, float(target_n))
    mesh = _safe_build(hi)
    tries = 0
    while (mesh is None or len(mesh.points) < target_n) and tries < 15:
        hi *= 2
        mesh = _safe_build(hi)
        tries += 1
    if mesh is None:
        raise ValueError(
            "buildMeshGrid2d: could not reach `lowrank` with a mesh "
            "satisfying `min_degree`; try a larger lowrank or a smaller "
            "min_degree."
        )

    best = mesh
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mesh = _safe_build(mid)
        if mesh is None:
            lo = mid
            continue
        n = len(mesh.points)
        best = mesh
        if abs(n - target_n) <= max(1, tol * target_n):
            break
        if n > target_n:
            hi = mid
        else:
            lo = mid

    return best, domain