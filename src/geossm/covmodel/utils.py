import warnings

import gmsh
import meshio
import numpy as np
import pygmsh
import shapely
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union


# % Utility functions

def _flatten_polygons(geom):
    """Flatten a Polygon / MultiPolygon / (possibly nested) list of these
    into a plain list of Polygons."""
    if isinstance(geom, (list, tuple)):
        polys = []
        for g in geom:
            polys.extend(_flatten_polygons(g))
        return polys
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, Polygon):
        return [geom]
    raise TypeError(
        "boundary must be a Polygon, a MultiPolygon, or a (possibly "
        f"nested) list of these; got {type(geom)}."
    )


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


def _prepare_domain(points, boundary, max_edge, min_edge, offset, cutoff):
    """Shared setup for buildMesh2d/buildMesh2d_pen: normalizes `boundary`
    (Polygon, MultiPolygon, or a possibly nested list of these) into the
    scientific-interest domain and the convex hull used to build the mesh,
    fills in max_edge/min_edge/offset/cutoff defaults, merges near-
    duplicate points, and buffers the domain.

    Returns
    -------
    points : (m, 2) ndarray
        `points` after merging near-duplicates within `cutoff`.
    n_input : int
        Number of points *before* that merge (what `lowrank` is relative to).
    interest_domain : shapely.geometry.base.BaseGeometry
        Union of `boundary`'s parts (or the convex hull of `points`, if
        `boundary` is not given), *before* it is widened to its convex hull.
    domain : shapely.geometry.Polygon
        The convex hull of `interest_domain`, buffered by `offset`: the
        actual extent the mesh is built over.
    coords : (k, 2) ndarray
        `domain`'s exterior ring, simplified, for gmsh's polygon input.
    max_edge, min_edge, offset, cutoff : float
        The (possibly defaulted) values actually used.
    """
    points = np.asarray(points, dtype=float)[:, :2]
    n_input = len(points)

    if boundary is None:
        interest_domain = MultiPoint(points).convex_hull
        boundary = interest_domain
    else:
        interest_domain = unary_union(_flatten_polygons(boundary))
        boundary = interest_domain.convex_hull

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

    return points, n_input, interest_domain, domain, coords, max_edge, min_edge, offset, cutoff


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

    The mesh covers the convex hull of ``boundary`` (or of ``points`` if
    `boundary` is not given), extended outward by ``offset`` to limit
    boundary effects, with triangle edges bounded by ``max_edge``/``min_edge``.

    Parameters
    ----------
    points : (n, 2) array_like
        Observed locations. Drive the default domain and, when `lowrank`
        is set, the local mesh density. Analogous to INLA's `loc`.
    boundary : Polygon, MultiPolygon, or (possibly nested) list of these, optional
        The scientific-interest domain, e.g. the same composition of
        polygons passed to `spdeAppoxCov`/`FEMSolver`. It does not need to
        be convex or a single piece (it can be a disconnected set of
        regions, as for a country plus its islands): all parts are unioned
        together, and the mesh is built over the convex hull of that union,
        extended by `offset`. Defaults to the convex hull of `points`.
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
        where they are sparse), so that the mesh has approximately
        ``round(lowrank * len(points))`` vertices *inside the scientific-
        interest domain* -- i.e. inside the union of `boundary` before it is
        widened to its convex hull and `offset` (or inside the convex hull
        of `points`, if `boundary` is not given). Vertices in the outer
        buffer region are not counted and are free to be as sparse as the
        size field makes them. Local density is a k-nearest-neighbor
        estimate (see `density_neighbors`) ranked by percentile, rather than
        a single-bandwidth KDE, so that separate clusters of comparable
        local density (e.g. several cities) are all refined even if one
        cluster has far more points overall.
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
    points, n_input, interest_domain, domain, coords, max_edge, min_edge, offset, cutoff = (
        _prepare_domain(points, boundary, max_edge, min_edge, offset, cutoff)
    )

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

    def _n_inside(mesh):
        pts = mesh.points
        return int(shapely.contains_xy(interest_domain, pts[:, 0], pts[:, 1]).sum())

    # Bisection on a global size-scale factor to hit the target vertex count
    # *inside the interest domain* (vertices in the outer offset buffer
    # don't count): smaller scale -> smaller elements everywhere -> more
    # vertices.
    lo, hi = 0.1, 10.0
    scale = lo
    mesh = _generate(scale=lo, opt_rounds=1)
    n_lo = _n_inside(mesh)

    if n_lo > target_n:
        for _ in range(max_iter):
            scale = 0.5 * (lo + hi)
            mesh = _generate(scale=scale, opt_rounds=1)
            n = _n_inside(mesh)
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
            f"buildMesh2d: reached {_n_inside(mesh)} vertices inside the "
            f"interest domain (target {target_n}; {len(mesh.points)} total "
            "including the outer buffer) but the minimum interior angle is "
            f"{angle:.1f} deg < min_angle={min_angle} deg. Consider a "
            "larger `lowrank`, or widen the min_edge/max_edge range."
        )

    return mesh, domain


def buildMesh2d_pen(
    points,
    lowrank,
    boundary=None,
    max_edge=None,
    min_edge=None,
    offset=None,
    cutoff=None,
    min_angle=21.0,
    snap_to_points=True,
    tol=0.05,
    max_iter=20,
    seed=None,
):
    """
    Build a 2D triangular mesh whose interior vertices *are* a density-
    matched subset of `points`, rather than free vertices placed by a
    smooth size field (as `buildMesh2d` does).

    Where `buildMesh2d` steers a Delaunay mesher's local element size to
    approximate the density of `points`, this function instead: (1) picks
    landmark locations via k-means on `points` -- since k-means codebook
    density tracks source density, this directly matches mesh-vertex
    density to point density; (2) by default snaps each landmark to its
    nearest *not-yet-used* point (a greedy closest-pair matching), so
    landmarks coincide exactly with observed locations instead of merely
    being near them, letting the mesh capture the observed variability
    directly at those nodes; (3) embeds the landmarks into the gmsh mesh
    as fixed points -- gmsh's optimizer moves every other node but never
    relocates or removes an embedded point, so the overlap from (2)
    survives mesh generation and smoothing exactly. Inside the interest
    domain, gmsh fills in the rest from the landmarks' own local spacing
    (`Mesh.MeshSizeFromPoints`) alone -- not a flat `max_edge` cap, which
    would force even the sparsest landmark-free pockets down to `max_edge`
    and badly overshoot the vertex budget for a small `lowrank`. `max_edge`
    is instead only imposed as a hard cap *outside* the interest domain,
    chiefly the outer offset buffer, which has no landmarks at all to size
    itself from. Either way, `max_edge` (and the landmarks' own local
    spacing) is what keeps interior angles away from zero: an abrupt jump
    from a tight cluster of landmarks straight to a coarse neighboring
    region would force sliver triangles, so gmsh grades a few extra
    (non-landmark) vertices in between as needed.

    Because that grading adds a few vertices beyond the landmarks
    themselves, asking k-means for exactly ``round(lowrank * len(points))``
    landmarks can still overshoot the vertex-count target slightly. So,
    mirroring `buildMesh2d`'s bisection over a size-field scale, this
    function bisects over the *number of landmarks requested* instead,
    regenerating the mesh each time, until the total vertex count inside
    the interest domain (landmarks plus grading) is within `tol` of the
    target -- see `buildMesh2d`'s `lowrank` for what "interest domain"
    means here.

    Landmarks closer together than `cutoff` are merged, which trades off
    against `min_angle`: forcing two near-duplicate data points to both
    remain exact, fixed vertices is often what produces a sliver triangle
    that no amount of optimization can fix, since fixed points cannot be
    moved. Widen `cutoff` (or reduce `lowrank`) if the `min_angle` warning
    fires often.

    Parameters
    ----------
    points : (n, 2) array_like
        Observed locations.
    lowrank : float
        Value in (0, 1]. Target number of vertices inside the interest
        domain, as a fraction of `len(points)` -- see above.
    boundary : Polygon, MultiPolygon, or (possibly nested) list of these, optional
        The scientific-interest domain; see `buildMesh2d`. Defaults to the
        convex hull of `points`.
    max_edge : float, optional
        Largest allowed triangle edge length, enforced as a hard cap in the
        outer buffer (outside the interest domain) and as an upper bound on
        each landmark's own local-spacing size; not otherwise enforced
        inside the interest domain, so it does not fight `lowrank` -- see
        above. Defaults to 1/15 of the domain's bounding-box diagonal.
    min_edge : float, optional
        Smallest allowed triangle edge length. Defaults to `max_edge / 10`.
    offset : float, optional
        Buffer added around `boundary` so the mesh extends past the data.
        Defaults to `max_edge`.
    cutoff : float, optional
        Minimum allowed separation between points, and independently
        between landmarks (see above). Defaults to `max_edge / 5`.
    min_angle : float, default 21.0
        Target minimum interior angle (degrees). This is a soft target
        graded around the fixed landmarks; it is not enforced on the
        landmarks' own placement, which is data-driven, not
        quality-driven -- see above.
    snap_to_points : bool, default True
        If True (recommended), landmarks coincide exactly with observed
        points (greedy nearest-pair matching from k-means centroids to
        `points`). If False, landmarks are left at the k-means centroids
        themselves -- close to the data but generally not exactly on it,
        which can give a smoother, better-quality triangulation.
    tol : float, default 0.05
        Relative tolerance on the vertex-count target used to stop the
        landmark-count search. Landmark placement (via k-means) and mesh
        grading are both a little noisy as the requested count changes by
        one, so this defaults looser than `buildMesh2d`'s `tol`.
    max_iter : int, default 20
        Maximum number of mesh (re)generations used by the search.
    seed : int, optional
        Seed for the k-means initialization. Also fixes it internally
        across the search's repeated k-means calls (even if left `None`,
        in which case a seed is drawn once and reused for this call only),
        so that requesting fewer landmarks is what changes the mesh, not
        fresh k-means randomness.

    Returns
    -------
    mesh : meshio.Mesh
    domain : shapely.geometry.Polygon
        The (buffered) domain the mesh was built over -- `boundary` (or its
        default) extended by `offset`.
    """
    points, n_input, interest_domain, domain, coords, max_edge, min_edge, offset, cutoff = (
        _prepare_domain(points, boundary, max_edge, min_edge, offset, cutoff)
    )

    if not (0 < lowrank <= 1):
        raise ValueError("lowrank must be in (0, 1].")
    target_n = min(len(points), max(3, round(lowrank * n_input)))

    # fixed for the duration of this call, so repeated k-means calls below
    # are directly comparable (see `seed` docs above)
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))

    def _place_landmarks(n_landmarks):
        n_landmarks = max(1, min(n_landmarks, len(points)))
        with warnings.catch_warnings():
            # scipy warns if k-means collapses onto fewer than
            # `n_landmarks` distinct clusters (e.g. many duplicate/near-
            # duplicate points); the cutoff-merge below reports that
            # outcome on its own.
            warnings.simplefilter("ignore", UserWarning)
            centroids, _ = kmeans2(points, k=n_landmarks, minit="++", seed=seed)

        if snap_to_points:
            # greedy closest-pair matching from centroids to points: not
            # the globally optimal assignment, but simple and effective,
            # and it guarantees each point is used as a landmark at most
            # once
            dist = cdist(centroids, points)
            order = np.argsort(dist, axis=None)
            n_pts = len(points)
            centroid_to_point = np.full(n_landmarks, -1)
            point_used = np.zeros(n_pts, dtype=bool)
            n_assigned = 0
            for flat_idx in order:
                ci, pi = divmod(int(flat_idx), n_pts)
                if centroid_to_point[ci] == -1 and not point_used[pi]:
                    centroid_to_point[ci] = pi
                    point_used[pi] = True
                    n_assigned += 1
                    if n_assigned == n_landmarks:
                        break
            landmarks = points[centroid_to_point]
        else:
            landmarks = centroids

        # enforce a minimum landmark separation: two landmarks closer than
        # `cutoff` would force a sliver triangle that fixed (embedded)
        # points can never be optimized away
        if cutoff > 0 and len(landmarks) > 1:
            keep = np.ones(len(landmarks), dtype=bool)
            for i in range(len(landmarks)):
                if not keep[i]:
                    continue
                d = np.linalg.norm(landmarks[i] - landmarks[i + 1:], axis=1)
                keep[i + 1:][d < cutoff] = False
            landmarks = landmarks[keep]

        return landmarks

    def _generate(landmarks, opt_rounds):
        # Each landmark's own mesh size is its distance to its nearest
        # other landmark (clipped to [min_edge, max_edge]), not a flat
        # `max_edge`: with `Mesh.MeshSizeFromPoints` this lets gmsh grade
        # element size from each point's *own* local spacing, so a sparse
        # region isn't padded with extra fill just because `max_edge` is
        # tuned for a denser region elsewhere.
        if len(landmarks) > 1:
            ltree = cKDTree(landmarks)
            nn_dist, _ = ltree.query(landmarks, k=2)
            landmark_lc = np.clip(nn_dist[:, 1], min_edge, max_edge)
        else:
            landmark_lc = np.array([max_edge])

        with pygmsh.occ.Geometry() as geom:
            surf = geom.add_polygon(coords, mesh_size=max_edge)
            geom.add_physical(surf, label="surface_domain")

            embedded_tags = [
                gmsh.model.occ.addPoint(x, y, 0, lc)
                for (x, y), lc in zip(landmarks, landmark_lc)
            ]
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.embed(0, embedded_tags, 2, surf.dim_tag[1])

            gmsh.option.setNumber("Mesh.Algorithm", 6)
            # let element size follow the (embedded) landmarks' own
            # spacing, rather than a hand-built field as in buildMesh2d
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_edge)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0e22)
            # `max_edge` is a hard cap only *outside* the interest domain
            # (chiefly the outer offset buffer, which has no landmarks at
            # all to size itself from). A flat cap applied everywhere would
            # also force the interest domain's sparser areas down to
            # `max_edge`, which for a small `lowrank` is typically much
            # finer than the landmarks alone need -- exactly what
            # overshoots the vertex-count target. Inside the interest
            # domain, sizing is left to `landmark_lc`/MeshSizeFromPoints
            # above (the large sentinel here is a no-op, combined via min).
            def _outer_cap(dim, tag, x, y, z, lc):
                if shapely.contains_xy(interest_domain, x, y):
                    return 1.0e22
                return max_edge

            gmsh.model.mesh.setSizeCallback(_outer_cap)

            gmsh.model.mesh.generate(2)
            for _ in range(opt_rounds):
                gmsh.model.mesh.optimize("Laplace2D")
                gmsh.model.mesh.optimize("Netgen")

            mesh = geom.generate_mesh()

        return mesh

    def _n_inside(mesh):
        pts = mesh.points
        return int(shapely.contains_xy(interest_domain, pts[:, 0], pts[:, 1]).sum())

    # Bisect the *number of requested landmarks* (not a continuous scale,
    # since placement itself is discrete/explicit here) so the final mesh
    # -- landmarks plus whatever grading gmsh adds -- has approximately
    # target_n vertices inside the interest domain. More landmarks always
    # means at least as much grading, so this is monotonic enough to
    # bisect despite the k-means/cutoff noise between successive integers.
    lo, hi = 3, target_n
    landmarks = _place_landmarks(hi)
    mesh = _generate(landmarks, opt_rounds=1)
    n_hi = _n_inside(mesh)

    best_landmarks, best_diff = landmarks, abs(n_hi - target_n)

    if n_hi > target_n:
        for _ in range(max_iter):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            landmarks = _place_landmarks(mid)
            mesh = _generate(landmarks, opt_rounds=1)
            n = _n_inside(mesh)
            diff = abs(n - target_n)
            if diff < best_diff:
                best_landmarks, best_diff = landmarks, diff
            if diff <= max(1, tol * target_n):
                break
            if n > target_n:
                hi = mid
            else:
                lo = mid

    # final pass with more optimization rounds to push the min angle up
    mesh = _generate(best_landmarks, opt_rounds=5)
    landmarks = best_landmarks

    n_inside = _n_inside(mesh)
    angle = _mesh_min_angle(mesh)
    if angle < min_angle:
        warnings.warn(
            f"buildMesh2d_pen: {len(landmarks)} landmark vertices placed "
            f"(requested from a {target_n}-vertex target), {n_inside} mesh "
            "vertices inside the interest domain, but the minimum interior "
            f"angle is {angle:.1f} deg < min_angle={min_angle} deg. "
            "Landmarks are fixed at (or near) data locations and cannot be "
            "moved to fix this; consider a larger `cutoff`, a smaller "
            "`lowrank`, or `snap_to_points=False`."
        )
    if best_diff > max(1, tol * target_n):
        warnings.warn(
            f"buildMesh2d_pen: reached {n_inside} vertices inside the "
            f"interest domain, outside the requested tolerance of target "
            f"{target_n} (tol={tol}). Consider a larger `max_iter`, or a "
            "coarser `max_edge` relative to the typical spacing between "
            "points."
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