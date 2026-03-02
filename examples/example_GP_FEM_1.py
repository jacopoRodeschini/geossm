import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from shapely.geometry import Polygon
from gstools.covmodel import Matern
import pygmsh
import gmsh
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from tqdm import tqdm
import geossm

# print("Version: ", geossm.__version__)
print("Load from: ", geossm.__file__)

# %% Import the Matern model based on the SPDE approach R^2

if geossm.__file__:
    from geossm.covmodel.covmodels import spdeAppoxCov

# %% Create a convex domain [0,1]^2
n = 400
domain = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

# generate points inside that domain
points = np.random.uniform(0, 1, size=(n, 2))

# plot the generate poinsts
fix, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 1], "x", label="Random Points")
ax.plot(domain.exterior.xy[0], domain.exterior.xy[1], "r-", label="Pacman Boundary")
ax.set_title("Random Points in Pacman Domain")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.legend()
plt.show()

# %% Generate the observed dataset

theta = 0.5  # range
rescale = np.sqrt(8 * 1) / theta

covf = Matern(dim=2, var=1, rescale=rescale, nugget=0, latlon=False, nu=1)

hdist = cdist(points, points)

Sigma = covf.covariance(hdist)
Sigma_chol = np.linalg.cholesky(Sigma + 1e-10 * np.eye(len(points)))

s2 = 0.1
yobs = Sigma_chol @ np.random.normal(size=len(points)) + np.random.normal(
    0, s2, size=len(points)
)

# plot the observed data
fix, ax = plt.subplots()
s = ax.scatter(points[:, 0], points[:, 1], c=yobs, cmap="viridis")
ax.plot(domain.exterior.xy[0], domain.exterior.xy[1], "r-", label="Pacman Boundary")
ax.set_title("Simulated Observations in Pacman Domain")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.colorbar(s, label="Observed Value")
plt.show()


# %% [Utils] Build mesh gmsh
def buildMesh(poly, lc, points, lc_buffer=None, lc_points=1e22, embed=False):
    with pygmsh.occ.Geometry() as geom:

        if lc_buffer is None:
            lc_buffer = lc

        coords = np.array(
            poly.buffer(lc_buffer).simplify(lc_buffer).exterior.coords[:-1]
        )
        domain = geom.add_polygon(coords, mesh_size=lc_buffer * 0.1)

        # 2. Add physical group for the domain surface (good practice)
        geom.add_physical(domain, label="surface_domain")

        # Add points for the boundary
        embedded_tags = []
        for p in points:
            t = gmsh.model.occ.addPoint(p[0], p[1], 0, lc_points)
            embedded_tags.append(t)

        gmsh.model.occ.synchronize()  # Synchronize OCC entities before using them in fields

        # fix the points
        if embed:
            gmsh.model.mesh.embed(0, embedded_tags, 2, domain._id)

        # gmsh.option.setNumber("Mesh.Algorithm", 6)

        # Allow triangles to be very large
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        # Only limit the absolute minimum to prevent crashes
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 2)

        # 5. Generate
        gmsh.model.mesh.generate(2)
        # gmsh.option.setNumber("Mesh.Smoothing", 10)

        # # This allows the optimizer to move nodes more freely
        # gmsh.option.setNumber("Mesh.Optimize", 1)
        # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        mesh = geom.generate_mesh()

    return mesh


# %% [Utils] Optimisation function
def minfun(par, H, spdeCov, yObs):

    n = len(yObs)
    rescale = np.exp(par)

    Q = spdeCov.precision(rescale=rescale)

    inx = spdeCov.fem_solver.inner
    Q_11 = Q[inx, :][:, inx]
    Q_12 = Q[inx, :][:, ~inx]
    Q_22 = Q[~inx, :][:, ~inx]

    # Covarianza marginale
    Q_mar = Q_11 - Q_12 @ np.linalg.inv(Q_22.toarray()) @ Q_12.T
    Sigma_mar = np.linalg.inv(Q_mar)

    # use SWM formula
    M = H @ Sigma_mar @ H.T + 0.1 * np.eye(n)
    invM = np.linalg.inv(M)

    # Calculate the log-pdf using the multivariate normal formula
    logdet = np.linalg.slogdet(M)[1]
    logpdf = logdet + yObs @ invM @ yObs.T

    #  print(rescale, logpdf)

    return 0.5 * logpdf


# %% Create the covariance funtion

# create the mesh
mesh_io = buildMesh(domain, lc=0.1, points=points, lc_buffer=1)

# Create the covariance funcion
cov_matern = spdeAppoxCov([domain], latlon=False, nu=1, var=1, rescale=1)

# set the mesh
cov_matern = cov_matern.setup(mesh_io)
print(cov_matern)

# plot the mesh
fix, ax = plt.subplots()
s = ax.scatter(points[:, 0], points[:, 1], c=yobs, cmap="viridis")
ax.plot(domain.exterior.xy[0], domain.exterior.xy[1], "r-", label="Pacman Boundary")
cov_matern.fem_solver.plot_mesh(ax=ax)
ax.set_title("Simulated Observations in Pacman Domain")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.colorbar(s, label="Observed Value")
plt.show()

# compute the H on the new points (while the mesh is fixed)
count, notfind, H = cov_matern.fem_solver._compute_basis(points)
H = H[:, cov_matern.fem_solver.inner]

# minimise the - log_likelihood
# %% Estiamte single process

res = sc.optimize.minimize(
    minfun, args=(H, cov_matern, yobs), x0=np.log(5), method="Nelder-Mead"
)


# %% The estimate task


def estimate(n=100):

    # Create the random points within the domain
    points = np.random.uniform(0, 1, size=(n, 2))

    covf = Matern(dim=2, var=1, rescale=rescale, nugget=0, latlon=False, nu=1)
    hdist = cdist(points, points)

    Sigma = covf.covariance(hdist)
    Sigma_chol = np.linalg.cholesky(Sigma + 1e-10 * np.eye(len(points)))

    s2 = 0.1
    yobs = Sigma_chol @ np.random.normal(size=len(points)) + np.random.normal(
        0, s2, size=len(points)
    )

    # create the mesh
    mesh_io = buildMesh(domain, lc=0.15, points=points, lc_buffer=0.5)

    # Create the covariance funcion
    cov_matern = spdeAppoxCov([domain], latlon=False, nu=1, var=1, rescale=1)
    cov_matern = cov_matern.setup(mesh_io)

    # compute the H on the new points (while the mesh is fixed)
    count, notfind, H = cov_matern.fem_solver._compute_basis(points)
    H = H[:, cov_matern.fem_solver.inner]

    res = sc.optimize.minimize(
        minfun, args=(H, cov_matern, yobs), x0=np.log(0.3), method="Nelder-Mead"
    )

    return np.exp(res.x)


# %% Run The MC simulation
boot = 10

# Parallel execution [faster]
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(estimate)(n=100) for _ in tqdm(range(boot))
)

bias = rescale - np.mean(results)
rmse = np.sqrt(np.mean((rescale - results) ** 2))
