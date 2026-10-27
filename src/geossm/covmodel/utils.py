import warnings

import gmsh
import meshio
import numpy as np
import pygmsh
import shapely
from scipy.cluster.vq import kmeans2
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree, Delaunay, QhullError
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
        "domain must be a Polygon, a MultiPolygon, or a (possibly "
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


def _prepare_domain(points, domain, max_edge, min_edge, offset, cutoff):
    """Shared setup for buildMesh2d/buildMesh2d_density: normalizes `domain`
    (Polygon, MultiPolygon, or a possibly nested list of these) into the
    scientific-interest domain and the convex hull used to build the mesh,
    fills in max_edge/min_edge/offset/cutoff defaults, merges near-
    duplicate points, and buffers the domain's hull.

    Returns
    -------
    points : (m, 2) ndarray
        `points` after merging near-duplicates within `cutoff`.
    n_input : int
        Number of points *before* that merge (what `lowrank` is relative to).
    interest_domain : shapely.geometry.base.BaseGeometry
        Union of `domain`'s parts (or the convex hull of `points`, if
        `domain` is not given), *before* it is widened to its convex hull.
    convex_hull : shapely.geometry.Polygon
        The convex hull of `interest_domain`, buffered by `offset`: the
        actual extent the mesh is built over -- i.e. where the SPDE/FEM
        problem is defined, as distinct from `interest_domain` (the
        scientific-interest domain).
    coords : (k, 2) ndarray
        `convex_hull`'s exterior ring, simplified, for gmsh's polygon input.
    max_edge, min_edge, offset, cutoff : float
        The (possibly defaulted) values actually used.
    """
    points = np.asarray(points, dtype=float)[:, :2]
    n_input = len(points)

    if domain is None:
        interest_domain = MultiPoint(points).convex_hull
        hull = interest_domain
    else:
        interest_domain = unary_union(_flatten_polygons(domain))
        hull = interest_domain.convex_hull

    bbox = hull.bounds
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

    convex_hull = hull.buffer(offset)
    if not isinstance(convex_hull, Polygon):
        raise ValueError("domain's convex hull, buffered by offset, did not yield a single polygon.")
    coords = np.array(convex_hull.simplify(offset * 0.25).exterior.coords[:-1])

    return points, n_input, interest_domain, convex_hull, coords, max_edge, min_edge, offset, cutoff


# Vertices per unit of int h^{-2}: an equilateral triangulation of edge h has
# area sqrt(3)/2 h^2 per vertex (6 triangles of area sqrt(3)/4 h^2 around a
# vertex, each shared by 3 vertices), hence N ~ (2/sqrt(3)) int_D h^{-2} ds.
_VERTEX_DENSITY_2D = 2.0 / np.sqrt(3.0)


def _gradient_limit(h, delta, gradation):
    """Largest g-Lipschitz field below `h` on a regular grid of spacing
    `delta`: h_g(x) = min_y { h(y) + g |x - y| } (Persson, 2006), with |x - y|
    the 8-neighbor grid-graph distance (overestimates the Euclidean one by at
    most ~8%). Computed exactly as a single-source shortest path from a
    virtual source linked to every node y with weight h(y). `h` may be +inf
    (no own constraint); every node then inherits one from its neighbors."""
    ny, nx = h.shape
    n = nx * ny
    idx = np.arange(n).reshape(ny, nx)

    rows, cols, wts = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        j0, j1 = max(0, -dj), nx - max(0, dj)
        a = idx[: ny - di, j0:j1].ravel()
        b = idx[di:, j0 + dj : j1 + dj].ravel()
        w = gradation * delta * np.hypot(di, dj)
        rows += [a, b]
        cols += [b, a]
        wts += [np.full(len(a), w), np.full(len(a), w)]

    finite = np.flatnonzero(np.isfinite(h.ravel()))
    rows.append(np.full(len(finite), n))
    cols.append(finite)
    wts.append(h.ravel()[finite])

    graph = coo_matrix(
        (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n + 1, n + 1),
    ).tocsr()
    dist = dijkstra(graph, directed=True, indices=n)
    return dist[:n].reshape(ny, nx)


def _size_conforming_subset(points, h_pts, alpha, eligible):
    """Greedy subset of `points` that is well spaced w.r.t. the size field:
    any two chosen points a, a' satisfy |a - a'| >= alpha (h(a) + h(a')) / 2,
    and every eligible point not chosen lies closer than that to a chosen one
    (maximality). Points are visited densest first (smallest h; ties by input
    order), as in variable-radius Poisson-disk sampling. Returns indices."""
    order = np.lexsort((np.arange(len(points)), h_pts))
    tree = cKDTree(points)
    blocked = ~eligible
    h_cap = h_pts.max()
    chosen = []
    for i in order:
        if blocked[i]:
            continue
        chosen.append(i)
        nb = np.asarray(tree.query_ball_point(points[i], alpha * 0.5 * (h_pts[i] + h_cap)))
        d = np.linalg.norm(points[nb] - points[i], axis=1)
        blocked[nb[d < alpha * 0.5 * (h_pts[i] + h_pts[nb])]] = True
    return np.array(chosen, dtype=int)


def _gmsh_mesh_from_size_grid(coords, xs, ys, h, embed=None, algorithm=6):
    """Frontal-Delaunay triangulation of the polygon `coords` whose local
    element size follows the grid field `h` (sampled at nodes `xs` x `ys`),
    passed to gmsh as a background PostView field so it is evaluated in C++
    rather than through a Python callback. Points in `embed` ((k, 2), strictly
    inside the polygon) become fixed mesh vertices. `algorithm` is gmsh's
    2D algorithm: 6 (Frontal-Delaunay, best angles) or 5 (Delaunay, which
    samples steep size fields more reliably)."""
    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("buildMesh2d_density_new")
    view = None
    try:
        geo = gmsh.model.geo
        ptags = [geo.addPoint(x, y, 0.0) for x, y in coords]
        ltags = [geo.addLine(ptags[i], ptags[(i + 1) % len(ptags)]) for i in range(len(ptags))]
        surf = geo.addPlaneSurface([geo.addCurveLoop(ltags)])
        etags = [geo.addPoint(x, y, 0.0) for x, y in (embed if embed is not None else [])]
        geo.synchronize()
        if etags:
            gmsh.model.mesh.embed(0, etags, 2, surf)

        # background triangulation of the size grid (two triangles per cell),
        # in gmsh's list format: x1 x2 x3 y1 y2 y3 z1 z2 z3 v1 v2 v3
        X, Y = np.meshgrid(xs, ys)
        idx = np.arange(X.size).reshape(X.shape)
        a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        tri = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
        xf, yf, hf = X.ravel(), Y.ravel(), h.ravel()
        data = np.hstack([xf[tri], yf[tri], np.zeros(tri.shape), hf[tri]])

        view = gmsh.view.add("size")
        gmsh.view.addListData(view, "ST", len(tri), data.ravel())
        field = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.model.mesh.generate(2)

        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        _, tri_nodes = gmsh.model.mesh.getElementsByType(2)
    finally:
        if view is not None:
            gmsh.view.remove(view)
        gmsh.model.remove()
        if initialized_here:
            gmsh.finalize()

    # gmsh node tags -> contiguous indices, dropping nodes no triangle uses
    lookup = np.full(int(node_tags.max()) + 1, -1)
    lookup[node_tags.astype(int)] = np.arange(len(node_tags))
    triangles = lookup[tri_nodes.astype(int)].reshape(-1, 3)
    used, triangles = np.unique(triangles, return_inverse=True)
    triangles = triangles.reshape(-1, 3)
    vertices = xyz.reshape(-1, 3)[used]
    vertices[:, 2] = 0.0

    # counter-clockwise orientation (gmsh follows the boundary ring's, which
    # shapely may give clockwise)
    p0, p1, p2 = (vertices[triangles[:, k], :2] for k in range(3))
    cw = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]) < 0
    triangles[cw] = triangles[cw][:, ::-1]
    return meshio.Mesh(points=vertices, cells=[("triangle", triangles)])


def buildMesh2d(
    points,
    lowrank=None,
    n_vertices=None,
    domain=None,
    intensity=None,
    bandwidth=None,
    eps=0.1,
    gradation=0.25,
    offset=None,
    outer_ratio=2.0,
    grid_size=None,
    anchor_spacing=None,
    min_angle=21.0,
    tol=0.02,
    max_iter=10,
    return_info=False,
):
    r"""
    Build a 2D triangular mesh with a vertex budget `R`, whose local element
    size follows the intensity of `points` through the Persson-Strang sizing
    function

    .. math::

        h(s) = \kappa \, \rho_\varepsilon(s)^{-1/2}, \qquad
        \rho_\varepsilon = (1 - \varepsilon)\, \hat f + \varepsilon / |D|,

    where :math:`\hat f = \hat\lambda / \hat\Lambda(D)` is the normalized
    intensity of `points` on the interest domain `D` (so
    :math:`\rho_\varepsilon = \lambda_\varepsilon / \Lambda_\varepsilon(D)`).

    Steps:

    1. *Intensity.* :math:`\hat\lambda` is a Gaussian kernel estimate with
       Diggle's edge correction, :math:`\hat\lambda(s) = \sum_i k_b(s - s_i)
       / \int_D k_b(s - u)\,du`, computed by convolution on a background grid
       (see `grid_size`), or any user `intensity`.
    2. *Sizing.* Since a near-equilateral mesh with size field `h` has
       :math:`N \approx (2/\sqrt3)\int_D h^{-2}` vertices in `D`, taking
       :math:`\kappa = \sqrt{2 / (\sqrt3 R)}` recovers exactly
       :math:`h = (c R \lambda_\varepsilon / \Lambda_\varepsilon(D))^{-1/2}`,
       :math:`c = \sqrt3/2`.
    3. *Gradation.* `h` is replaced by the largest `gradation`-Lipschitz field
       below it, :math:`h_g(x) = \min_y \{\kappa h(y) + g |x - y|\}`, so the
       ratio of neighboring element sizes is at most about :math:`1 + g`
       (smooth transitions, hence shape-regular elements). Outside `D` (the
       `offset` buffer) `h` is defined by this extension alone, capped at
       `outer_ratio` times its largest value in `D`.
    4. *Budget.* :math:`\kappa` is found by the fixed-point iteration
       :math:`\kappa \leftarrow \kappa \sqrt{N(\kappa) / R}`, first on the
       predicted count :math:`(2/\sqrt3)\int_D h_g^{-2}`, then on the vertex
       count of the actual gmsh mesh, until it is within `tol` of `R`.
    5. *Meshing.* gmsh's Frontal-Delaunay mesher on the buffered convex hull,
       with :math:`h_g` as a background size field. If it returns fewer than
       half the vertices the field predicts (it can miss small fine regions
       of a steep field, e.g. with a large `gradation`), gmsh's Delaunay
       mesher is used instead, which samples the field more reliably.
    6. *Anchoring* (only with `anchor_spacing` = :math:`\alpha`). Before
       meshing, a subset :math:`A` of `points` is chosen greedily, densest
       first, so that :math:`|a - a'| \ge \alpha (h_g(a) + h_g(a'))/2` for
       :math:`a, a' \in A` and every other point is closer than that to some
       anchor. :math:`A` is embedded as fixed vertices and gmsh fills the rest
       of the domain from the same size field. Because :math:`A` is spaced
       according to :math:`h_g`, the size field, budget calibration and
       gradation are unchanged, while every point :math:`s` is either a vertex
       or within :math:`\alpha h_g(s) / (1 - \alpha g / 2)` of one.

    Consistency: because :math:`\rho_\varepsilon \ge \varepsilon/|D|`,
    :math:`\sup_D h \le \sqrt{2|D| / (\sqrt3\,\varepsilon R)} \to 0` as
    :math:`R \to \infty` (gradation only lowers `h`), while
    :math:`\sup h / \inf h \le \sqrt{\sup\rho_\varepsilon / \inf\rho_\varepsilon}`
    stays bounded independently of `R`.

    Parameters
    ----------
    points : (n, 2) array_like
        Observed locations.
    lowrank : float, optional
        Value in (0, 1]. Sets the budget to ``R = round(lowrank * len(points))``
        vertices inside `D`, as in `buildMesh2d_density`.
    n_vertices : int, optional
        The budget `R` directly. Give exactly one of `lowrank` and
        `n_vertices`. Unlike `lowrank`, it may exceed ``len(points)``.
    domain : Polygon, MultiPolygon, or (possibly nested) list of these, optional
        The interest domain `D`; see `buildMesh2d`. Defaults to the convex
        hull of `points`. The vertex budget counts vertices inside `D` only.
    intensity : callable, optional
        ``intensity(xy) -> (m,)`` non-negative values at the ``(m, 2)``
        locations `xy`, replacing the kernel estimate (e.g. a fitted
        point-process intensity). Only proportionality matters.
    bandwidth : float, optional
        Gaussian kernel bandwidth `b`. Defaults to Scott's rule,
        ``n**(-1/6) * sqrt((var_x + var_y) / 2)``. Ignored with `intensity`.
    eps : float, default 0.1
        Uniform mixing weight in (0, 1]: the fraction of the budget spread
        evenly over `D`. Must be positive for consistency (``h_max -> 0``);
        it also bounds the size ratio between sparse and dense regions.
    gradation : float, default 0.25
        Lipschitz constant `g` of the size field (Persson's gradient limit).
        Smaller values give smoother transitions at the cost of a weaker
        density contrast for a fixed budget.
    offset : float, optional
        Buffer added around `D`'s convex hull. Defaults to 1/15 of the
        domain's bounding-box diagonal.
    outer_ratio : float, default 2.0
        Cap on element size in the buffer, relative to the largest one in `D`.
    grid_size : int, optional
        Background-grid resolution (cells along the longer side) for the
        intensity and size fields. Defaults to at least 3 cells across the
        smallest element (between 200 and 1500); a warning is raised if an
        explicit value is too coarse to resolve it.
    anchor_spacing : float, optional
        If given (typically 0.6-1.0), co-locate mesh vertices with a
        size-conforming subset of `points` (step 6). Smaller values anchor
        more points and bound the point-to-vertex distance more tightly, at
        the cost of anchors closer together than the size field asks for
        (worse angles). ``None`` (default) leaves vertex placement to gmsh.
    min_angle : float, default 21.0
        Soft target for the minimum interior angle (degrees); a warning is
        raised if the mesh does not reach it.
    tol : float, default 0.02
        Relative tolerance on the vertex-count target.
    max_iter : int, default 10
        Maximum number of gmsh mesh generations used to calibrate `kappa`.
    return_info : bool, default False
        Also return a dict of diagnostics (size and intensity fields on the
        background grid, `kappa`, calibration history, minimum angle).

    Returns
    -------
    mesh : meshio.Mesh
    convex_hull : shapely.geometry.Polygon
        The (buffered) convex hull the mesh was built over.
    info : dict
        Only if `return_info` is True.
    """
    if (lowrank is None) == (n_vertices is None):
        raise ValueError("give exactly one of `lowrank` and `n_vertices`.")
    if not (0 < eps <= 1):
        raise ValueError("eps must be in (0, 1].")
    if gradation <= 0:
        raise ValueError("gradation must be positive.")

    points, n_input, interest_domain, convex_hull, coords, _, _, offset, _ = (
        _prepare_domain(points, domain, None, None, offset, 0.0)
    )
    if lowrank is not None:
        if not (0 < lowrank <= 1):
            raise ValueError("lowrank must be in (0, 1].")
        target_n = max(3, round(lowrank * n_input))
    else:
        target_n = max(3, int(n_vertices))

    minx, miny, maxx, maxy = convex_hull.bounds
    extent = max(maxx - minx, maxy - miny)
    shapely.prepare(interest_domain)
    if intensity is None and bandwidth is None:
        bandwidth = len(points) ** (-1 / 6) * np.sqrt(points.var(axis=0).mean())

    def _calibrated_fields(n_cells):
        # Background grid (nodes) covering the buffered hull with one spare
        # cell on each side, so gmsh never queries the size field outside it.
        delta = extent / n_cells
        xs = minx - delta + delta * np.arange(int(np.ceil((maxx - minx) / delta)) + 3)
        ys = miny - delta + delta * np.arange(int(np.ceil((maxy - miny) / delta)) + 3)
        X, Y = np.meshgrid(xs, ys)
        mask = shapely.contains_xy(interest_domain, X, Y)
        if mask.sum() < 10:
            raise ValueError("domain is too small for the background grid; increase grid_size.")
        dA = delta**2
        area_D = mask.sum() * dA

        # (1) intensity on D, normalized to a density f (int_D f = 1)
        if intensity is not None:
            lam = np.zeros(X.shape)
            lam[mask] = np.asarray(intensity(np.column_stack([X[mask], Y[mask]])), dtype=float)
            if np.any(lam < 0) or not np.all(np.isfinite(lam)):
                raise ValueError("`intensity` must return finite, non-negative values.")
        else:
            ix = np.rint((points[:, 0] - xs[0]) / delta).astype(int)
            iy = np.rint((points[:, 1] - ys[0]) / delta).astype(int)
            ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
            counts = np.zeros(X.shape)
            np.add.at(counts, (iy[ok], ix[ok]), 1.0)
            sigma = bandwidth / delta
            smoothed = gaussian_filter(counts, sigma, mode="constant")
            # Diggle edge correction e(s) = int_D k_b(s - u) du; floored so
            # thin slivers of D don't blow up the estimate
            edge = gaussian_filter(mask.astype(float), sigma, mode="constant")
            lam = np.where(mask, smoothed / np.maximum(edge, 0.05), 0.0)
        total = lam[mask].sum() * dA
        if total <= 0:
            raise ValueError("the intensity vanishes on the whole domain.")
        f = lam / total

        # (2) reference size field h0 = rho_eps^{-1/2} on D, unconstrained
        # outside
        rho = (1 - eps) * f + eps / area_D
        h0 = np.full(X.shape, np.inf)
        h0[mask] = rho[mask] ** -0.5

        def size_field(kappa):
            # (3) gradation, then the cap on the outer buffer
            hg = _gradient_limit(kappa * h0, delta, gradation)
            return np.minimum(hg, outer_ratio * hg[mask].max())

        # (4a) calibrate kappa on the predicted count; without gradation
        # this is exactly kappa = sqrt(c / R), gradation only adds vertices
        kappa = np.sqrt(_VERTEX_DENSITY_2D / target_n)
        for _ in range(20):
            h = size_field(kappa)
            n_pred = _VERTEX_DENSITY_2D * np.sum(h[mask] ** -2.0) * dA
            if abs(n_pred - target_n) <= 0.25 * tol * target_n:
                break
            kappa *= np.sqrt(n_pred / target_n)
        return xs, ys, mask, delta, dA, f, size_field, kappa, h

    if grid_size is None:
        # coarse pass to locate the smallest element, then a grid with at
        # least 3 cells across it
        _, _, mask, _, _, _, _, _, h = _calibrated_fields(200)
        grid_size = int(np.clip(np.ceil(3 * extent / h[mask].min()), 200, 1500))
    xs, ys, mask, delta, dA, f, _size_field, kappa, h = _calibrated_fields(grid_size)

    if h[mask].min() < 2 * delta:
        warnings.warn(
            f"buildMesh2d_density_new: the smallest element size "
            f"({h[mask].min():.3g}) is below two background-grid cells "
            f"({2 * delta:.3g}); increase `grid_size` to resolve the size field."
        )

    def _n_inside(mesh):
        pts = mesh.points
        return int(shapely.contains_xy(interest_domain, pts[:, 0], pts[:, 1]).sum())

    if anchor_spacing is not None:
        if anchor_spacing <= 0:
            raise ValueError("anchor_spacing must be positive.")
        # anchors must sit strictly inside the meshed polygon, far enough
        # from its boundary vertices not to force slivers there
        mesh_polygon = Polygon(coords)
        inside_polygon = shapely.contains_xy(mesh_polygon, points[:, 0], points[:, 1])
        boundary_dist = shapely.distance(mesh_polygon.exterior, shapely.points(points))

    def _anchors(h):
        if anchor_spacing is None:
            return None
        # points outside the grid are outside the meshed polygon too, hence
        # never eligible; the fill value only keeps them out of the way
        h_pts = RegularGridInterpolator(
            (ys, xs), h, bounds_error=False, fill_value=float(h.max())
        )(points[:, ::-1])
        eligible = inside_polygon & (boundary_dist >= anchor_spacing * h_pts)
        return _size_conforming_subset(points, h_pts, anchor_spacing, eligible)

    # (4b)+(5)+(6) mesh, then correct kappa on the actual vertex count
    history = []
    best = None
    algorithm = 6  # Frontal-Delaunay: best angles
    for _ in range(max_iter):
        anchors = _anchors(h)
        embed = None if anchors is None else points[anchors]
        mesh = _gmsh_mesh_from_size_grid(coords, xs, ys, h, embed=embed, algorithm=algorithm)
        n = _n_inside(mesh)
        n_pred = _VERTEX_DENSITY_2D * np.sum(h[mask] ** -2.0) * dA
        if algorithm == 6 and n < 0.5 * n_pred:
            # Frontal-Delaunay can miss small fine regions of a steep field
            # (e.g. large `gradation`) and return far fewer vertices than the
            # field asks for, making N(kappa) discontinuous; Delaunay samples
            # the field at every circumcenter and does not, so switch to it
            algorithm = 5
            mesh = _gmsh_mesh_from_size_grid(coords, xs, ys, h, embed=embed, algorithm=algorithm)
            n = _n_inside(mesh)
        history.append((float(kappa), n))
        if best is None or abs(n - target_n) < abs(best[1] - target_n):
            best = (mesh, n, kappa, h, anchors)
        if abs(n - target_n) <= max(1, tol * target_n):
            break
        kappa *= np.sqrt(n / target_n)
        h = _size_field(kappa)

    mesh, n_inside, kappa, h, anchors = best
    if abs(n_inside - target_n) > max(1, tol * target_n):
        warnings.warn(
            f"buildMesh2d_density_new: reached {n_inside} vertices inside the "
            f"interest domain (target {target_n}, tol={tol}); consider a "
            "larger `max_iter`."
        )
    angle = _mesh_min_angle(mesh)
    if angle < min_angle:
        warnings.warn(
            f"buildMesh2d_density_new: minimum interior angle is {angle:.1f} "
            f"deg < min_angle={min_angle} deg. Consider a smaller `gradation` "
            "or a larger `grid_size`."
        )

    if not return_info:
        return mesh, convex_hull
    info = {
        "target_n": target_n,
        "n_inside": n_inside,
        "n_total": len(mesh.points),
        "kappa": float(kappa),
        "history": history,
        "min_angle": angle,
        "bandwidth": bandwidth,
        "grid_x": xs,
        "grid_y": ys,
        "domain_mask": mask,
        "density": f,
        "size_field": h,
        "h_max_D": float(h[mask].max()),
        "h_min_D": float(h[mask].min()),
        # distance from each of `points` to its nearest mesh vertex
        "point_distance": cKDTree(mesh.points[:, :2]).query(points)[0],
        "anchors": anchors,
        "algorithm": algorithm,
    }
    return mesh, convex_hull, info


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
    domain=None,
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
    `ny` vertices spanning the bounding box of `domain` (or the convex
    hull of `points` if `domain` is not given), extended by `offset`.
    Grid cells lying entirely outside the (buffered) convex hull are
    dropped, so the mesh follows the shape of `domain` rather than just its
    bounding box. Any vertex left with fewer than `min_degree` mesh edges by
    that pruning (stray corners along the boundary "staircase") is removed
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
        Observed locations. Used to derive `domain` (as their convex
        hull) when `domain` is not given, and as the basis for `lowrank`.
    domain : shapely.geometry.Polygon, optional
        The scientific-interest domain the mesh must cover.
    offset : float, optional
        Buffer added around `domain` so the lattice extends past the
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
    convex_hull : shapely.geometry.Polygon
        The (buffered) convex hull the mesh was built over -- `domain` (or
        its default) extended by `offset`.
    """
    if domain is None:
        if points is None:
            raise ValueError("either `points` or `domain` must be given.")
        points = np.asarray(points, dtype=float)[:, :2]
        domain = MultiPoint(points).convex_hull
    elif points is not None:
        points = np.asarray(points, dtype=float)[:, :2]

    if offset is None:
        bbox = domain.bounds
        diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        offset = diag / 15

    convex_hull = domain.buffer(offset) if offset else domain
    if not isinstance(convex_hull, Polygon):
        raise ValueError("domain.buffer(offset) did not yield a single polygon.")

    minx, miny, maxx, maxy = convex_hull.bounds
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

        # keep only cells whose centroid lies within the (buffered) convex hull
        centroids = grid[triangles].mean(axis=1)
        inside = shapely.contains_xy(convex_hull, centroids[:, 0], centroids[:, 1])
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
        return _build(nx, ny), convex_hull

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

    return best, convex_hull