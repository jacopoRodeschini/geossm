#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Parallel search over `buildMesh2d` settings.

Given observation sites, an interest domain and a vertex budget (`lowrank`),
build one mesh per combination of settings (eps, gradation, bandwidth,
anchor_spacing, ...) in parallel and return one row of quality metrics per
mesh, so the mesh best suited to a case study can be picked from a table.

Use from Python:

    from mesh_settings_search import search_mesh_settings, rank_settings, build_from_row
    table = search_mesh_settings(points, domain, lowrank=0.5, n_jobs=16)
    ranked = rank_settings(table, min_angle=21, tol=0.05)
    mesh, hull = build_from_row(points, domain, ranked.iloc[0])

Workers are started with "spawn" (gmsh and JAX do not survive a fork), which
re-imports the calling script in every worker. So, in a script, call the
search under ``if __name__ == "__main__":``; Jupyter and VS Code ``# %%``
cells work as they are. An `intensity` callback must be a function defined
in an importable module (not a lambda or a function defined in the
notebook); build any heavy state lazily inside that module.

From the command line, on the GRINS PM10 stations:

    python examples/mesh_settings_search.py --lowrank 0.5 --jobs 16 --out mesh_search.csv

Columns of the returned table (distances are in the units of `points`; "h" is
the gradient-limited size field h_g at that location):

    settings ......... eps, gradation, bandwidth_factor, bandwidth, anchor_spacing
    status ........... ok / warning / error, and the warning or error message
    target, n_inside, n_total, n_over_target
                       budget R, vertices inside the domain, all vertices
                       (inside + outer buffer), and n_inside / R
    n_sites, colocated, colocated_frac
                       sites inside the domain, how many of them are mesh
                       vertices, and that share
    dist_mean, dist_q10, dist_q25, dist_median, dist_q75, dist_q90, dist_max
                       distance from each site in the domain to its nearest
                       mesh vertex
    dist_h_mean, dist_h_median, dist_h_q90
                       the same distance divided by h at the site
    angle_min, angle_q01, angle_q05, angle_median
                       smallest interior angle of each triangle (degrees),
                       summarised over the mesh
    frac_angle_lt20, frac_angle_lt30
                       share of triangles whose smallest angle is below
                       20 / 30 degrees
    angle_max, frac_obtuse
                       largest angle in the mesh, and share of triangles with
                       an angle above 90 degrees (each one can add a positive
                       off-diagonal entry to the FEM stiffness matrix)
    nbr_ratio_max, nbr_ratio_q99
                       size ratio sqrt(area_i / area_j) between triangles that
                       share an edge: how smooth the grading is
    edge_h_median, edge_h_q10, edge_h_q90
                       edge length / h at its midpoint, inside the domain:
                       1 means the mesh follows the size field
    h_min, h_max, h_contrast
                       range of the size field in the domain, and h_max / h_min
    area_ratio         largest / smallest triangle area
    mass_ratio         largest / smallest lumped FEM mass (only with fem=True)
    iters, algorithm, seconds
                       gmsh runs used to meet the budget, gmsh algorithm
                       (6 Frontal-Delaunay, 5 Delaunay fallback), wall time
"""

import argparse
import itertools
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

# A small grid around the defaults; every key is a `buildMesh2d`
# argument, except `bandwidth_factor`, which multiplies Scott's bandwidth.
DEFAULT_GRID = {
    "eps": [0.05, 0.1, 0.2],
    "gradation": [0.1, 0.25, 0.5],
    "bandwidth_factor": [0.2, 0.5, 1.0],
    "anchor_spacing": [None, 1.0, 0.8, 0.6],
}


# %% Worker (runs in a separate process)

def _init_worker():
    # one BLAS/OpenMP thread per process, and keep JAX (imported by geossm)
    # off the GPU in the workers
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _scott_bandwidth(points):
    return len(points) ** (-1 / 6) * np.sqrt(points.var(axis=0).mean())


def _interest_domain(points, domain):
    import shapely
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    from geossm.covmodel.utils import _flatten_polygons

    union = MultiPoint(points).convex_hull if domain is None else unary_union(_flatten_polygons(domain))
    shapely.prepare(union)
    return union


def _triangle_geometry(P, T):
    p = [P[T[:, k]] for k in range(3)]
    area = 0.5 * np.abs((p[1][:, 0] - p[0][:, 0]) * (p[2][:, 1] - p[0][:, 1])
                        - (p[1][:, 1] - p[0][:, 1]) * (p[2][:, 0] - p[0][:, 0]))
    angles = []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u, v = p[b] - p[a], p[c] - p[a]
        cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    angles = np.column_stack(angles)
    return area, angles.min(axis=1), angles.max(axis=1)


def _neighbour_ratios(T, area):
    edges = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    tri = np.tile(np.arange(len(T)), 3)
    _, inv = np.unique(edges, axis=0, return_inverse=True)
    order = np.argsort(inv.ravel(), kind="stable")
    inv, tri = inv.ravel()[order], tri[order]
    shared = inv[1:] == inv[:-1]
    r = np.sqrt(area[tri[1:][shared]] / area[tri[:-1][shared]])
    return np.maximum(r, 1.0 / r)


def mesh_metrics(mesh, info, points, union, domain=None, fem=False):
    """Quality metrics of one mesh built by `buildMesh2d` with
    `return_info=True`; see the module docstring for their meaning."""
    import shapely
    from scipy.interpolate import RegularGridInterpolator

    P, T = mesh.points[:, :2], mesh.cells_dict["triangle"]
    area, tri_min, tri_max = _triangle_geometry(P, T)
    ratios = _neighbour_ratios(T, area)
    h_at = RegularGridInterpolator((info["grid_y"], info["grid_x"]), info["size_field"],
                                   bounds_error=False, fill_value=np.nan)

    # sites inside the interest domain
    in_d = shapely.contains_xy(union, points[:, 0], points[:, 1])
    d = info["point_distance"][in_d]
    h_site = h_at(points[in_d][:, ::-1])
    d_h = d / h_site

    # edges inside the interest domain, relative to the size field
    e = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    mid = 0.5 * (P[e[:, 0]] + P[e[:, 1]])
    keep = shapely.contains_xy(union, mid[:, 0], mid[:, 1])
    edge_h = np.linalg.norm(P[e[keep, 0]] - P[e[keep, 1]], axis=1) / h_at(mid[keep][:, ::-1])

    mass_ratio = np.nan
    if fem:
        try:
            from geossm.covmodel import FEMSolver

            solver = FEMSolver(mesh, domain=domain, verbose=False, stats=False)
            lumped = np.asarray(solver._mass.sum(axis=1)).ravel()
            mass_ratio = float(lumped.max() / lumped.min())
        except Exception:
            pass

    q = lambda x, p: float(np.nanpercentile(x, p)) if len(x) else np.nan  # noqa: E731
    colocated = int((d < 1e-9).sum())
    return {
        "target": info["target_n"],
        "n_inside": info["n_inside"],
        "n_total": info["n_total"],
        "n_over_target": info["n_inside"] / info["target_n"],
        "n_sites": int(in_d.sum()),
        "colocated": colocated,
        "colocated_frac": colocated / max(1, int(in_d.sum())),
        "dist_mean": float(d.mean()) if len(d) else np.nan,
        "dist_q10": q(d, 10), "dist_q25": q(d, 25), "dist_median": q(d, 50),
        "dist_q75": q(d, 75), "dist_q90": q(d, 90),
        "dist_max": float(d.max()) if len(d) else np.nan,
        "dist_h_mean": float(np.nanmean(d_h)) if len(d) else np.nan,
        "dist_h_median": q(d_h, 50), "dist_h_q90": q(d_h, 90),
        "angle_min": float(tri_min.min()),
        "angle_q01": q(tri_min, 1), "angle_q05": q(tri_min, 5), "angle_median": q(tri_min, 50),
        "frac_angle_lt20": float((tri_min < 20).mean()),
        "frac_angle_lt30": float((tri_min < 30).mean()),
        "angle_max": float(tri_max.max()),
        "frac_obtuse": float((tri_max > 90).mean()),
        "nbr_ratio_max": float(ratios.max()), "nbr_ratio_q99": q(ratios, 99),
        "edge_h_median": q(edge_h, 50), "edge_h_q10": q(edge_h, 10), "edge_h_q90": q(edge_h, 90),
        "h_min": info["h_min_D"], "h_max": info["h_max_D"],
        "h_contrast": info["h_max_D"] / info["h_min_D"],
        "area_ratio": float(area.max() / area.min()),
        "mass_ratio": mass_ratio,
        "iters": len(info["history"]),
        "algorithm": info["algorithm"],
    }


def _resolve_settings(points, settings):
    """Turn `bandwidth_factor` into an absolute `bandwidth`."""
    kw = dict(settings)
    factor = kw.pop("bandwidth_factor", None)
    if factor is not None and kw.get("bandwidth") is None and kw.get("intensity") is None:
        kw["bandwidth"] = factor * _scott_bandwidth(points)
    return kw


def _evaluate(job):
    job_id, points, domain, budget, settings, fem = job
    from geossm.covmodel import buildMesh2d

    row = {"job": job_id, **{k: v for k, v in settings.items() if k != "intensity"}}
    kw = _resolve_settings(points, settings)
    row["bandwidth"] = kw.get("bandwidth", np.nan)
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mesh, _, info = buildMesh2d(points, domain=domain, return_info=True,
                                        **budget, **kw)
        union = _interest_domain(points, domain)
        row.update(mesh_metrics(mesh, info, points, union, domain=domain, fem=fem))
        if kw.get("bandwidth") is None:  # Scott's default, or None with a callback
            row["bandwidth"] = info["bandwidth"]
        msgs = [str(w.message).split(": ", 1)[-1] for w in caught
                if "buildMesh2d" in str(w.message)]
        row["status"] = "warning" if msgs else "ok"
        row["message"] = " | ".join(msgs)
    except Exception as exc:  # report, never stop the search
        row["status"] = "error"
        row["message"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - t0
    return row


# %% Public API

def expand_grid(grid):
    """Cartesian product of a {name: [values]} dict, as a list of dicts."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def search_mesh_settings(points, domain=None, lowrank=None, n_vertices=None, grid=None,
                         settings=None, fixed=None, n_jobs=None, fem=False, verbose=True):
    """Build one `buildMesh2d` mesh per setting, in parallel, and
    return a DataFrame with one row of quality metrics per mesh.

    Parameters
    ----------
    points : (n, 2) array_like
        Observation sites.
    domain : Polygon, MultiPolygon, or list of these, optional
        Interest domain, as for `buildMesh2d`.
    lowrank, n_vertices : float / int
        The vertex budget (give exactly one), shared by all settings.
    grid : dict of lists, optional
        Settings to combine (Cartesian product). Keys are
        `buildMesh2d` arguments, plus `bandwidth_factor`
        (multiple of Scott's bandwidth). Defaults to `DEFAULT_GRID`.
    settings : list of dicts, optional
        Explicit settings instead of (or in addition to) `grid`.
    fixed : dict, optional
        Arguments applied to every run (e.g. ``{"grid_size": 600}``).
        An `intensity` callback must be a function defined in an
        importable module (not a lambda): see the module docstring.
    n_jobs : int, optional
        Worker processes. Defaults to ``min(len(settings), cpu_count - 1)``;
        1 runs sequentially in this process.
    fem : bool, default False
        Also assemble the FEM mass matrix (`mass_ratio` column); slower.
    """
    points = np.asarray(points, dtype=float)[:, :2]
    if (lowrank is None) == (n_vertices is None):
        raise ValueError("give exactly one of `lowrank` and `n_vertices`.")
    budget = {"lowrank": lowrank} if lowrank is not None else {"n_vertices": n_vertices}

    runs = list(settings or [])
    if grid is not None or not runs:
        runs += expand_grid(grid or DEFAULT_GRID)
    runs = [{**(fixed or {}), **s} for s in runs]
    jobs = [(i, points, domain, budget, s, fem) for i, s in enumerate(runs)]

    n_jobs = n_jobs or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    t0 = time.perf_counter()
    rows = []
    if n_jobs == 1:
        rows = [_evaluate(job) for job in jobs]
    else:
        # spawn, not fork: gmsh and JAX do not survive a fork reliably
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(n_jobs, mp_context=ctx, initializer=_init_worker) as pool:
            futures = [pool.submit(_evaluate, job) for job in jobs]
            for k, fut in enumerate(as_completed(futures), 1):
                try:
                    rows.append(fut.result())
                except BrokenProcessPool as exc:
                    raise RuntimeError(
                        "a worker process could not start: call search_mesh_settings under "
                        "`if __name__ == '__main__':` in scripts, make any `intensity` "
                        "callback an importable module-level function, or use n_jobs=1."
                    ) from exc
                if verbose and (k % max(1, len(jobs) // 10) == 0 or k == len(jobs)):
                    print(f"  {k}/{len(jobs)} meshes done ({time.perf_counter() - t0:.0f}s)")
    if verbose:
        print(f"{len(jobs)} settings evaluated in {time.perf_counter() - t0:.1f}s on {n_jobs} process(es)")
    return pd.DataFrame(rows).sort_values("job").reset_index(drop=True)


def rank_settings(table, min_angle=21.0, tol=0.05, max_nbr_ratio=None, by="dist_median"):
    """Flag and order the settings: `suitable` rows meet the angle bound and
    the budget within `tol` (and, optionally, a neighbour-ratio bound); they
    come first, ordered by `by` (default: median site-to-vertex distance),
    then by the minimum angle (descending)."""
    t = table[table.status != "error"].copy()
    ok = (t.angle_min >= min_angle) & ((t.n_over_target - 1).abs() <= tol)
    if max_nbr_ratio is not None:
        ok &= t.nbr_ratio_max <= max_nbr_ratio
    t["suitable"] = ok
    return t.sort_values(["suitable", by, "angle_min"], ascending=[False, True, False])


def build_from_row(points, domain, row, lowrank=None, n_vertices=None, **extra):
    """Rebuild the mesh of one table row (the mesher is deterministic)."""
    from geossm.covmodel import buildMesh2d

    if lowrank is None and n_vertices is None:
        n_vertices = int(row["target"])
    budget = {"lowrank": lowrank} if lowrank is not None else {"n_vertices": n_vertices}
    names = ("eps", "gradation", "bandwidth", "anchor_spacing", "offset", "outer_ratio", "grid_size")
    kw = {k: row[k] for k in names if k in row and not pd.isna(row[k])}
    kw.setdefault("anchor_spacing", None)
    return buildMesh2d(np.asarray(points, float)[:, :2], domain=domain,
                       **budget, **kw, **extra)


SUMMARY_COLUMNS = [
    "eps", "gradation", "bandwidth_factor", "anchor_spacing", "status",
    "n_inside", "target", "colocated_frac", "dist_median", "dist_q90", "dist_h_median",
    "angle_min", "angle_q05", "frac_angle_lt30", "frac_obtuse", "nbr_ratio_max",
    "edge_h_median", "h_contrast", "seconds",
]


# %% Command line: GRINS PM10 stations over Italy

def _load_grins(shapefile):
    import geopandas as gpd
    from shapely import ops
    from shapely.geometry import MultiPolygon

    import geossm.datasets as gd

    grins, _ = gd.load_dataset("aqclim_points")
    regions = gpd.read_file(shapefile).to_crs(4326)
    regions["geometry"] = regions["geometry"].apply(
        lambda g: max(g.geoms, key=lambda p: p.area) if isinstance(g, MultiPolygon) else g
    )
    italy = ops.unary_union(regions.geometry)
    stations = grins[~np.isnan(grins["AQ_mean_PM10"])]["AirQualityStation"].unique()
    sub = grins[grins["AirQualityStation"].isin(stations)]
    points = np.array([g.coords[0] for g in sub.geometry.unique()])
    return points, list(getattr(italy, "geoms", [italy]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lowrank", type=float, default=0.5)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--fem", action="store_true", help="also assemble FEM mass matrices")
    parser.add_argument("--min-angle", type=float, default=21.0)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--out", default="mesh_settings_search.csv")
    parser.add_argument(
        "--shapefile",
        default="/home/jrodeschini/geopy_casestudy/geopy_grins/EApplication/Ita_region/Reg01012024_g_WGS84.shp",
    )
    args = parser.parse_args()

    points, domain = _load_grins(args.shapefile)
    print(f"{len(points)} stations, lowrank={args.lowrank}")
    table = search_mesh_settings(points, domain, lowrank=args.lowrank, n_jobs=args.jobs, fem=args.fem)
    table.to_csv(args.out, index=False)

    ranked = rank_settings(table, min_angle=args.min_angle, tol=args.tol)
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.float_format", "{:.3g}".format):
        print(f"\n{int(ranked.suitable.sum())} of {len(table)} settings meet "
              f"min angle >= {args.min_angle} and budget within {100 * args.tol:.0f}%. Top 15:")
        print(ranked[["suitable"] + SUMMARY_COLUMNS].head(15).to_string(index=False))
        errors = table[table.status == "error"]
        if len(errors):
            print(f"\n{len(errors)} settings failed:")
            print(errors[["eps", "gradation", "bandwidth_factor", "anchor_spacing", "message"]].to_string(index=False))
    print(f"\nfull table ({table.shape[1]} columns) written to {args.out}")
