#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:35:01 2024

@author: jacopo
@ title: create the spatial covariance class to model the spatial covariance for
the spatial process
"""
# from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import mfem.ser as mfem
# from mfem.common.vtk import VTKOutput

import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy as sc
import gstools as gs
from shapely.geometry import LineString, Point, Polygon, MultiPoint

from gstools.covmodel import Matern
import warnings

from scipy.spatial import ConvexHull


# %%

import platform
import psutil
import datetime

from scipy.sparse.linalg import splu

# %% Compute the log-probability density function (PDF) from the precision matrix Q


def logpdfQ(y, Q):
    """
    Compute the log-probability density function (PDF) of a multivariate normal 
    distribution defined by the precision matrix Q.

    Parameters:
    ----------
    y : numpy.ndarray
        A vector of observations of shape (n,).
    Q : numpy.ndarray
        Precision matrix (inverse of the covariance matrix) of shape (n, n).

    Returns:
    -------
    float
        The log-probability density value.

    Notes:
    -----
    The formula used is:
        logp(y; Q) = -1/2 * [ n * log(2π) - log(det(Q)) + y^T Q y ]
    where:
        - n is the size of Q,
        - det(Q) is the determinant of Q,
        - y^T Q y is the quadratic form involving the precision matrix.
    """
    n = Q.shape[0]  # Dimension of Q (assumed square)
    # logdetQ = np.linalg.slogdet(Q)[1]  # Compute log(abs(det(Q)))
    logdetQ = logdetSparse(Q)

    # Calculate the log-pdf using the multivariate normal formula
    logp = -n * np.log(2 * np.pi) + logdetQ - y @ Q @ y

    return 0.5 * logp


# %% Compute the sparse log-determinant of a matrix Q

def logdetSparse(Q):
    """
    Compute the log-determinant of a sparse matrix Q using LU decomposition.

    Parameters:
    ----------
    Q : scipy.sparse matrix
        Sparse matrix of shape (n, n).

    Returns:
    -------
    float
        The log-determinant of Q.

    Notes:
    -----
    For sparse matrices, direct computation of the determinant is infeasible.
    Instead, this function uses LU decomposition to extract the diagonal elements
    of L and U matrices:
        log(det(Q)) = sum(log(abs(diag(L)))) + sum(log(abs(diag(U))))
    """
    # Perform LU decomposition of the sparse matrix
    lu = splu(Q)

    # Extract diagonal elements from L and U
    diagL = lu.L.diagonal().astype(np.complex128)
    diagU = lu.U.diagonal().astype(np.complex128)

    # Compute the log-determinant
    logdet = np.log(np.abs(diagL)).sum() + np.log(np.abs(diagU)).sum()

    return logdet


# %% Compute the Kullback-Leibler divergence between two distributions

def KLdivergence2Q(Sigma1, Q2):
    """
    Compute the Kullback-Leibler (KL) divergence between two Gaussian distributions.

    Parameters:
    ----------
    Sigma1 : numpy.ndarray
        Covariance matrix of the true distribution (n x n).
    Q2 : numpy.ndarray
        Precision matrix (inverse covariance matrix) of the approximate distribution (n x n).

    Returns:
    -------
    float
        The KL divergence between the two distributions.

    Notes:
    -----
    The KL divergence for Gaussian distributions is given by:
        D_KL(P || Q) = 1/2 * [ trace(Q2 * Sigma1) - log(det(Sigma1)) - log(det(Q2)) - n ]
    where:
        - Sigma1 is the covariance of the true distribution P,
        - Q2 is the precision matrix of the approximate distribution Q.
    """
    # Compute log-determinants of Sigma1 and Q2
    logdetS1 = np.linalg.slogdet(Sigma1)[1]
    logdetQ2 = np.linalg.slogdet(Q2)[1]

    # Compute KL divergence
    kl_div = 0.5 * (np.trace(Q2 @ Sigma1) - logdetS1 -
                    logdetQ2 - Sigma1.shape[0])

    return kl_div


# %% Convert a covariance matrix to a correlation matrix

def cov2corr(A):
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters:
    ----------
    A : numpy.ndarray
        Covariance matrix of shape (n, n).

    Returns:
    -------
    numpy.ndarray
        Correlation matrix of the same shape (n, n).

    Notes:
    -----
    The formula to convert a covariance matrix A to a correlation matrix R is:
        R_ij = A_ij / (sqrt(A_ii) * sqrt(A_jj))
    The diagonal entries of the correlation matrix are set to 1.
    """
    # Compute the standard deviations (sqrt of diagonal elements)
    d = np.sqrt(A.diagonal())

    # Normalize rows and columns to obtain the correlation matrix
    corr_matrix = ((A.T / d).T) / d

    return corr_matrix


# %% Compute the block diagonal 3D

def block_diag_3D(*arrs):
    """
    Create a 3D block diagonal matrix from given 3D matrices where the first
    two dimensions can vary but the last dimension is the same.
    Each input array should be of shape (n_i, m_i, p), where p is constant.

    Parameters:
    *arrs : 3D matrices to be stacked in block diagonal manner.

    Returns:
    np.ndarray : 3D block diagonal matrix.
    """
    # Determine the total shape for the first two dimensions
    total_shape_0 = sum(arr.shape[0] for arr in arrs)
    total_shape_1 = sum(arr.shape[1] for arr in arrs)
    # the last dimension should be the same for all arrays
    total_shape_2 = arrs[0].shape[2]

    # Initialize the block diagonal matrix with zeros
    block_diag_matrix = np.zeros(
        (total_shape_0, total_shape_1, total_shape_2))

    # Current start index for the first two dimensions
    current_index_0 = 0
    current_index_1 = 0

    for arr in arrs:
        shape_0, shape_1, shape_2 = arr.shape
        block_diag_matrix[current_index_0:current_index_0+shape_0,
                          current_index_1:current_index_1+shape_1,
                          :] = arr
        current_index_0 += shape_0
        current_index_1 += shape_1

    return block_diag_matrix

# %% Write function on a csv file


def write(filename, grid_obs, ssm_model, mode='a'):
    # write the class into a file

    # Get current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s = f"Current Time: {current_time} \n"

    s += "OBSERVATION GRID \n"
    s += ''.join([gr.__str__() for gr in grid_obs])
    s += "\n"

    s += "SSM REPRESENTATION \n"
    s += ssm_model.__str__()
    s += "\n"

    s += "MODEL PARAMITER ESTIMATE"
    s += "\n"

    s += "HARDWARE \n"
    s += getHardware()

    with open(filename, mode) as f:
        f.write(s)

    return filename, mode

# %% Get Hardware infromation


def getHardware():

    # Get CPU info
    cpu_name = platform.processor()
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().current

    # Get architecture
    architecture = platform.architecture()[0]

    # Get memory info
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total / (1024 ** 3)  # Convert to GB
    available_memory = virtual_memory.available / \
        (1024 ** 3)  # Convert to GB

    # Get system info
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    node = platform.node()

    # Compile all information into a string
    info = (f"System: {system} {release} {version}\n"
            f"Node Name: {node}\n"
            f"Machine: {machine}\n"
            f"Architecture: {architecture}\n"
            f"CPU: {cpu_name}\n"
            f"Physical CPUs: {cpu_count}\n"
            f"Logical CPUs: {cpu_count_logical}\n"
            f"Current CPU Frequency: {cpu_freq:.2f} MHz\n"
            f"Total Memory: {total_memory:.2f} GB\n"
            f"Available Memory: {available_memory:.2f} GB\n")
    return info


# %% Find the nearest positive-definite matrix to input
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

# %% Check the positive-definite matrix via Cholesky


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

# %% SPDE approximation of the Matern cov function


class spdeAppoxCov(Matern):
    r"""The SPDE approximation of the Matérn covariance model.

    Notes
    -----
    This model is given by the following correlation function

    Using Neumann boundary conditions

    References
   ----------
   .. [Rasmussen2003] Rasmussen, C. E.,
          "Gaussian processes in machine learning." Summer school on
          machine learning. Springer, Berlin, Heidelberg, (2003)


    """

    def __init__(self, points, triangles=None, latlon=True, geo_scale=gs.DEGREE_SCALE,
                 mesh=None, uniformRef=True, add_boundary=False,
                 boundary_poly=None, boundary_step=None, outer_index=None, level=1,
                 mesh_geotype='triangularized', mesh_geoshape='TRIANGLE',
                 mesh_boundary_type=None, nu=1, s2=1, rescale=0.1):

        # update the geo matern parameter
        # self._nu = nu         # smoothness
        # self._s2 = s2         # Marginal variance of the observed process
        self.rescale = rescale  # Rescale paramter
        self.level = level
        self.boundary_step = boundary_step
        self.add_boundary = add_boundary
        self.boundary_step = boundary_step
        self.outer_index = outer_index
        self.level = level
        self.boundary_poly = boundary_poly
        self.points = points

        if boundary_poly is not None:
            add_boundary = True

        # build the super __init__
        super().__init__(dim=2, var=s2, len_scale=1.0, nugget=0.0, anis=1.0,
                         angles=0.0, integral_scale=None, rescale=rescale, latlon=latlon,
                         geo_scale=geo_scale, temporal=False, spatial_dim=2,
                         var_raw=None, hankel_kw=None)

        # update the point
        if points.shape[0] != 2 and points.shape[1] != 2:
            raise ValueError('Points size must be (n,2) or (2,n)')

        if points.shape[0] == 2:
            points = points.T

        # Update the mesh geometry
        self._mesh_geotype = mesh_geotype
        self._mesh_geoshape = mesh_geoshape
        self._mesh_boundary_type = 'BOX' if (
            boundary_poly is None and outer_index is None) else 'POLY'
        self._mesh_uniformRef = uniformRef

        # Include point Array of bool or scalar (TRUE/FALSE)

        # Set the points (observation)

        # Create the mesh (physical domain) with the boundary edges
        self._mesh = None    # Create the private attibute
        self._fespace = None  # (See the mesh setter)
        self._mass = None
        self._stiff = None

        # Update mesh (if mesh = none) or build (if mesh = None)
        # and create the associate fespace. Finally compute stiff and mass

        self._boundary_points = None
        if mesh is not None:
            self.mesh = mesh  # set the private attribute (see the subfunction)
        else:
            # Create the mesh (physical domain) with the boundary edges and nodes
            self.mesh, self._inner_points, self._boundary_points = self._build_mesh(
                points, triangles=triangles,
                add_boundary=add_boundary, boundary_poly=boundary_poly,
                boundary_step=boundary_step, outer_index=outer_index,
                level=level)

    # redefine some super class methods

    def cor(self, h):
        """Matérn normalized correlation function."""
        # h = distance
        # h = np.asarray(np.abs(h), dtype=np.double)
        # for nu > 20 we just use the gaussian model

        return None

    def _build_mesh(self, points, triangles=None, add_boundary=False, boundary_poly=None,
                    outer_index=None, boundary_step=None, level=1):

        mesh = None
        if self._mesh_geotype == 'triangularized':
            # Perform Delaunay triangulation on the points
            # Delaunay triangulation maximizes the minimum angle, leading to a more regular
            # distribution of triangle shapes. However, in practice, to meet specific size
            # and shape constraints, adding extra vertices (sometimes referred to as mesh
            # refinement or adaptive mesh refinement) can reduce the number of triangles
            # while still maintaining the desired properties.

            # Combine interior points and boundary points
            # # uncomment this line to include such border

            # Compute the boundary points
            # vertex of the mesh = inner + outer(boundary)
            outer_points = None    # # Outer vertex
            boundary_points = None  # Boundary vertex

            if add_boundary or boundary_poly is not None or outer_index is not None:
                if outer_index is not None:

                    inner_points = points[~outer_index, :]
                    outer_points = points[outer_index, :]

                elif boundary_poly is not None:
                    inner_points = points
                    outer_points, level_points = self._add_boundary_poly(
                        boundary_poly, boundary_step, level)

                else:
                    inner_points = points
                    outer_points, level_points = self._add_boundary_box(
                        inner_points, boundary_step, level)

                # all points = (inner + outer)
                all_points = np.vstack(
                    (inner_points, outer_points), dtype=float)

                # outer vertex indices
                outer_indices = set(
                    range(len(inner_points), len(all_points)))
            else:
                # Update the varialbe name
                inner_points = points
                all_points = points

            # Compute the Delaunay triangolarisation (end refine it)
            if triangles is None:
                tri = Delaunay(all_points)

                # if self._mesh_boundary_type == 'POLY':
                #     # remove triangle along the boundary
                #     points, triangles = self._refine_mesh(
                #         tri, boundary_points, level_points)

                triangles = tri.simplices
                all_points = tri.points

            # self.vertex = points (handeld by the mesh)
            self.triangle = triangles

            # Find the boundary points pf all mesh
            boundary_indices = self._find_boundary_vertices(triangles)
            index = np.ones(all_points.shape[0], dtype=bool)
            index[boundary_indices] = 0
            boundary_points = all_points[~index]

            # update the object actribbute
            # Fix the over computational issus
            # self.n_totpoints = len(points)
            # self.n_boundary_points = len(boundary_points)
            # self.boundary_vertex = boundary_vertex

            # Define the MFEM mesh with vertices and triangular elements
            # 2D mesh, n_points vertices, len(triangles) elements, boundery element, space
            mesh = mfem.Mesh(2, len(all_points), len(triangles), 0, 2)

            self._add_element_vertex(
                mesh, all_points, triangles, boundary_indices)

            # Finalize the boundary elements
            # Finalize mesh generation (Finalize the construction of a triangular Mesh)
            # # mesh.FinalizeTriMesh(1)  # 1 means it is linear order mesh

            mesh.FinalizeTopology()
            mesh.Finalize()

        if self._mesh_uniformRef == True and mesh is not None:
            # Uniform refinement of the mesh Delanuay trinagolarisation
            mesh.UniformRefinement()

        return mesh, inner_points, outer_points

    def _add_boundary_box(self, points, boundary_step, level, box=None):
        """
            Generate boundary points around a predefined domain with the specified number of levels.

            Parameters:
            ----------
            level : int
                Number of "layers" around the boundary to generate.
            boundary_step : float
                Step size between boundary points.

            Returns:
            -------
            numpy.ndarray
                Array of boundary points, shape (num_points, 2).
        """
        # self._inner_points = points

        if box is None:
            # points = self.inner_points

            # Define the box
            box = np.array([points[:, 0].min(), points[:, 1].min(),
                           points[:, 0].max(), points[:, 1].max()])

        # define the boundary step (2% of the box)
        if boundary_step is None:
            boundary_step = np.abs(box).max()*0.20

        bLeft = box[0]
        bBottom = box[1]

        # bTop = box[1] + np.ceil((box[2] - box[1]) /
        #                         boundary_step)*boundary_step

        alpha = np.ceil(abs(box[2] - box[0]) / boundary_step)
        bRight = bLeft + alpha*boundary_step

        alpha = np.ceil(abs(box[1] - box[3]) / boundary_step)
        bTop = bBottom + alpha*boundary_step

        level_points = []
        for l in range(level):
            # Expand the box dimensions
            bLeft -= boundary_step
            bBottom -= boundary_step
            bRight += boundary_step
            bTop += boundary_step

            # Generate bottom and top edges
            x_bottom_top = np.arange(
                bLeft, bRight, boundary_step)
            bot_top_left = np.column_stack(
                (x_bottom_top, np.full_like(x_bottom_top, bBottom)))
            bot_top_right = np.column_stack(
                (x_bottom_top, np.full_like(x_bottom_top, bTop)))

            # Generate left and right edges
            y_left_right = np.arange(
                bBottom, bTop + 0.1*boundary_step, boundary_step)
            left_right_bot = np.column_stack(
                (np.full_like(y_left_right, bLeft), y_left_right))
            left_right_top = np.column_stack(
                (np.full_like(y_left_right, bRight), y_left_right))

            # Combine the edges to form a closed boundary for this level
            le_points = np.vstack((bot_top_left, left_right_top, np.flip(
                bot_top_right, axis=0), np.flip(left_right_bot, axis=0)))
            level_points.append(le_points)

        # Concatenate all levels
        return np.vstack(level_points), level_points

    def _add_element_vertex(self, mesh, all_points, all_triangles, boundary_indices):

        all_points = np.asarray(all_points, dtype=float)
        # Add vertices to the mesh
        for i in range(len(all_points)):
            # x, y = points[i]
            # Add each vertex and get the ids
            # print(i, all_points[i])
            ids = mesh.AddVertex(all_points[i])
            # vertex_ids.append(ids)

        # Add triangular elements to the mesh
        for i in range(len(all_triangles)):
            mesh.AddTriangle(
                all_triangles[i, 0], all_triangles[i, 1], all_triangles[i, 2])

        # Find edges connecting boundary points
        bdr_indices = boundary_indices  # Indices of bdr pt
        bdr_edges = set()  # Store unique boundary edges

        for simplex in all_triangles:
            # For each triangle, add its edges
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                edge = tuple(sorted((simplex[i], simplex[j])))
                if edge[0] in bdr_indices and edge[1] in bdr_indices:
                    # Only add edges between boundary points
                    bdr_edges.add(edge)

        # Add boundary edges to the mesh
        for edge in bdr_edges:
            mesh.AddBdrSegment(edge[0], edge[1])  # Add edge by vertex IDs

    # def _find_boundary_vertices(triangles):
    #     edge_count = defaultdict(int)

    #     for t in triangles:
    #         edges = [(t[i], t[(i+1) % 3]) for i in range(3)]
    #         for edge in edges:
    #             # Sort to avoid direction dependency
    #             edge_count[tuple(sorted(edge))] += 1

    #     boundary_vertices = set()
    #     for (v1, v2), count in edge_count.items():
    #         if count == 1:
    #             boundary_vertices.add(v1)
    #             boundary_vertices.add(v2)

    #     return np.array(list(boundary_vertices))

    def _find_boundary_vertices(self, triangles):
        """Find boundary vertices"""
        edge_set = set()
        duplicate_edges = set()

        # Track unique and duplicate edges
        for t in triangles:
            edges = [tuple(sorted((t[i], t[(i + 1) % 3]))) for i in range(3)]
            for edge in edges:
                if edge in edge_set:
                    duplicate_edges.add(edge)  # Edge appears twice
                else:
                    edge_set.add(edge)

        # Boundary edges = edges that appear only once
        boundary_edges = edge_set - duplicate_edges

        # Extract unique boundary vertices
        boundary_vertices = set(v for e in boundary_edges for v in e)

        return np.array(list(boundary_vertices))

    def _build_feSpace(self):

        # Create a finite element space
        # Define a finite element space on the mesh. Here we use vector finite
        # elements, i.e. dim copies of a scalar finite element space. The vector
        # dimension is specified by the last argument of the FiniteElementSpace
        # constructor.
        # Order 1 finite elements
        fec = mfem.H1_FECollection(1, self.mesh.Dimension())
        fespace = mfem.FiniteElementSpace(self.mesh, fec)

        return fespace

    def _compute_basis(self, points=None, thr=1e-5):
        # @Points = physical point

        # Create the list of pysical points
        if points is None:
            points = self.vertex  # total grid point
        else:
            points = np.asarray(points, dtype=np.float64)

        physical_points = np.asarray(points)
        npoint = physical_points.shape[0]

        # The shape functions or finite element functions define the behavior of the
        # finite element solution. These functions are piecewise polynomials that are
        # defined on the reference element but describe the field you are trying to solve

        # 1) Find the element of the mesh that contains the physical point p
        # This becouse the map function are locally defined
        # 2) Map the physical point to the reference space (TrasformBack).
        # 3) Evaluate the shape functions at the reference point.

        # Find the id of the elements on the mesh [pysical domain] that contain the
        # given points, and their corresponding reference coordinates.
        # This method is not 100 percent reliable, i.e. it is not guaranteed to
        # find a point, even if it lies inside a mesh element.
        count, elem_ids, int_points = self.mesh.FindPoints(physical_points)
        nbasis = self.GetVSize

        # Termporary vector
        phys_point = mfem.Vector(self.mesh.Dimension())
        shape_vals = mfem.Vector(self.fespace.GetNDofs())

        # integrator point in the reference domain
        ref_point = mfem.IntegrationPoint()

        H = np.zeros((npoint, nbasis))
        notfindInx = []
        for i in range(len(elem_ids)):
            ids = elem_ids[i]

            # -1 = points not found in the reference domain
            if ids != -1:

                # Get the element map function (pysical domain to reference domain)
                phys_point.Assign(physical_points[i, :])
                tran = self.mesh.GetElementTransformation(ids)

                # Compute the basis func in the pyhisical space
                # CalcPhysShape(tran, phys_point)

                # Map the point in the reference domain
                tran._TransformBack(phys_point, ref_point)

                # Get the functional element assosicate with the element ids
                # Shape basis function
                fe = self.fespace.GetFE(ids)

                # Dof = number of shape functions (save value in shape_vals)
                fe.CalcShape(ref_point, shape_vals)

                # Get the local-to-global DOF mapping for this element
                # Returns indices of degrees of freedom for the i'th element.
                col_indices = self.fespace.GetElementVDofs(ids)

                H[i, col_indices] = shape_vals.GetDataArray()

            else:
                notfindInx.append(i)

        # Create the sparse matrix [p x n] of the basis function (evaluate in the
        # reference domain)

        # check the thr and put 0 (numerical stability)
        H[H <= 1e-8] = 0
        H = sp.csr_matrix(H)

        # Check the "parition of unity" rule
        # H.sum(axis=1)
        if len(notfindInx) != 0:
            s = "The following point index need to be removed because can't be find in the latent domain \n"
            s += "See the MFEM FindPoints function documentatios \n"
            s += f"Index {notfindInx} \n"
            warnings.warn(s)

        notfindInx = np.asarray(notfindInx)
        return count, notfindInx, H

    def getBasis(self, points=None):
        count, notfindInx, H = self._compute_basis(points)
        return count, notfindInx, H

    def _compute_mass_stiff(self):
        # Compute the mass and stiff matrix (static matrix -> computed just one time)

        # Get the Mass (C matrix in RUE-LINGDEN) and Stiffness matrix (G in RUE)
        # Initialize BilinearForms to represent mass and stiffness matrices

        # Define the constant coefficient '1' for integration (scalar)
        one = mfem.ConstantCoefficient(1.0)

        # DomainLFIntegrator  -- SPACE (H1, L2) -- (phi,phi)
        # This become the inner product with <phi, 1> where phi are the basis function
        # defined in the finte element (FE) space
        # 1) Create the linear form for the mass matrix C
        # 2) Add a domain integrator for the inner product <phi, 1>
        # 3) This integrator computes the inner product between the basis function
        # and the constant scalar 1 over each element.

        c = mfem.LinearForm(self.fespace)
        c.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
        c.Assemble()

        # Therefore, be careful not to access after the matrix is freed.
        # [n x 1] since LinearForm results in a vector rather than a full matrix
        temp = c.GetDataArray()
        mass = sp.diags(temp, offsets=0, shape=(
            self.nvertex, self.nvertex), format="csr")

        # 1) Create the bilinear form for the stiff matrix (with the one coefficient)
        # 2) Add a diffusion integrator to compute <Grad(phi_i), Grad(phi_j)>
        # The coefficient for the diffusion term. The default is 1.0, meaning we
        # compute the standard internal product
        g = mfem.BilinearForm(self.fespace)
        g.AddDomainIntegrator(mfem.DiffusionIntegrator())
        g.Assemble()

        # Finalize to convert the assembled form to a sparse matrix
        g.Finalize()

        # Get the stiffness sparse matrix
        spmat = g.SpMat()

        # GetIArray, GetJArray, and GetDataArray. These methods give NumPy array of CSR
        # matrix data.

        i = spmat.GetIArray()       # Row pointers (cumulative sums of non-zero elements per row)
        j = spmat.GetJArray()       # get index of j (column)
        dt = spmat.GetDataArray()   # Non zero values of the sparse matrix

        # Get the number of degrees of freedom (number of rows/columns in the matrix)
        n_dofs = self.fespace.GetNDofs()

        # Build the stifness sparse matrix G using the CSR format
        stiff = sp.csr_matrix(
            (dt, j, i), shape=(n_dofs, n_dofs), copy=True)

        return mass, stiff

    def _update_geo_parameter(self, rescale):

        if rescale is None:
            raise ValueError('rescale must be provided')

        # if (rescale > self.box[1]).any():
        #     th_scale = np.round(np.sqrt(8*self.nu) / self.box[1].min(), 2)
        #     raise ValueError(
        #         f"Min value for rescale is {th_scale}")

        return rescale              # rescale

    def _compute_precision_spde(self, rescale=None):
        """
        @rescale = rescale factor
        Compute the precision matrix (Q, sparse) of the process y with marginal variance 
        sigma2(k) = sigma2_spde

        """

        # update the geo parameter
        # self._update_geo_parameter(rescale)
        if rescale is not None:
            self.rescale = self._update_geo_parameter(
                rescale)  # Rescale factor (k)

        # Compute the inverse of the mass matrix
        Cinv = sp.diags(1/self.mass.diagonal(), offsets=0, shape=(
            self.nvertex, self.nvertex), format="csr")

        # Compute the K matrix
        k = self.rescale
        K = (k**2) * self.mass + self.stiff

        # Compute the precision matrix of the process y
        Q = self.sigma2k * (K @ Cinv @ K)

        return Q

    def precision(self, rescale=None):
        """ 
        Compute the precision matrix (Q, sparse) for spatial process z with marginal variance
        sigma2_process

        """

        return self._compute_precision_spde(rescale)

    def loglikelihood(self, yTrue, rescale=None, s2=None):
        """Compute the log-likelihood function starting from the precision matrix"""

        # compute Sigma true
        Q = self.precision(rescale, s2)

        # compute the normla pdf (loglikelihhod)
        return utils.logpdfQ(yTrue, Q)

    def plot_mesh(self, ax, alpha_vertex=1, alpha_triangle=0.5, alpha_border=0.5):

        # Convert the vertex array to a numpy array
        vertex = self.vertex

        # Collect all triangles (elements)
        inner_triangles = []
        for i in range(self.nelement):
            tri = self.mesh.GetElement(i).GetVerticesArray()
            inner_triangles.append(tri)
        inner_triangles = np.array(inner_triangles)

        # Collect all boundary edges
        bdr_edges = []
        for i in range(self.nbElement):
            bElem = self.mesh.GetBdrElement(i).GetVerticesArray()
            bdr_edges.append(bElem)

        boundary_edge = np.array(bdr_edges)

        # Plot the mesh
        # fig, ax = plt.subplots()

        # Plot the triangles
        ax.triplot(vertex[:, 0], vertex[:, 1], inner_triangles,
                   color='grey', alpha=alpha_triangle)

        # Plot the interior vertices
        ax.plot(self.inner_points[:, 0], self.inner_points[:, 1],
                'xm', label='Interior Vertices', alpha=alpha_vertex)

        # Plot the boundary vertices
        ax.plot(self.boundary_points[:, 0], self.boundary_points[:, 1],
                'x', color='black', label='Boundary Vertices', alpha=alpha_border)

        # Plot the boundary edges as dashed lines
        for edge in boundary_edge:
            x_coords = vertex[edge, 0]
            y_coords = vertex[edge, 1]
            ax.plot(x_coords, y_coords, '--r', alpha=alpha_border)

        # Add a legend
        ax.legend()

        # Display the plot
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Mesh Visualization")
        # plt.show()
        return ax

    # %% property:: spatial process

    @property
    def emp_range(self):
        """Return the empirical range paramiter  """
        return np.sqrt(8*self.nu) / self.rescale

    @property
    def sigma2k(self):
        """
        Return the marginal variance of the standardise approximate spatial SPDE process
        Variance of the aproximate field x(u). Eq. 2 and Eq. 9
        """
        return sc.special.gamma(1) / (sc.special.gamma(2) * 4*np.pi * (self.rescale**2))

    @property
    def inner_totpoints(self):
        """Return the point of the grid"""
        return self._inner_totpoint

    @property
    def include_inner_point(self):
        """Return the point of the grid"""
        return self._include_inner_point

    @property
    def inner_points(self):
        """Return the point of the grid"""
        return self._inner_points

    @property
    def n_inner_points(self):
        """Return the point of the grid"""
        return self.inner_points.shape[0]

    @property
    def boundary_points(self):
        """Return the boundary point of the grid"""
        return self._boundary_points

    @property
    def n_boundary_points(self):
        """Return the number of the boundary point of the grid"""
        return self.boundary_points.shape[0]

    @property
    def totpoints(self):
        """Return the total point (inner + boundary) of the grid (same as vertex)"""
        return self.vertex

    @property
    def n_totpoints(self):
        """Return the total point number of the grid"""
        return self.nvertex

    # %% property:: FE SPACE (fespace)

    @property
    def fespace(self):
        """Return the finite elment space"""
        return self._fespace

    # Property for the space dimension (fespace.GetVDim())
    @property
    def fespace_dim(self):
        return self._fespace.GetVDim()

    # Property for number of local degrees of freedom (fespace.GetNDofs())
    @property
    def ndofs(self):
        return self._fespace.GetNDofs()

    # Property for number of vector DOFs (fespace.GetVSize())
    @property
    def GetVSize(self):
        return self._fespace.GetVSize()

    # %% property:: MESH

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh=None):

        # compute the new stiff and mass matrix points boundary etc etc

        # Set the proviate attibute
        self._mesh = new_mesh
        self._points = self.nvertex

        # Create the finite element space (wich cannot be buld outside the class)
        self._fespace = self._build_feSpace()

        # Compute the stiff and mass matrix (just private matrix)
        # once the finite element is build -> compute the mass and stiff matrix
        self._mass, self._stiff = self._compute_mass_stiff()

        return self.shape

    # Property for number of vertices (mesh.GetNV())
    @property
    def shape(self):
        return (self.nvertex, self.nelement)

    @property
    def nvertex(self):
        """Get the number of vertex of the discretisation"""
        return self.mesh.GetNV()

    @property
    def nelement(self):
        """Get the number of discretisation elements (like the number of triangle)"""
        return self.mesh.GetNE()

    # Property for vertex array (mesh.GetVertexArray())
    @property
    def vertex(self):
        """Get the vertex array"""
        return np.array(self.mesh.GetVertexArray())

    def get_distance(self, points=None):
        """Get the distance between the (vertex, points) or (vertex, vertex)"""

        return cdist(self.vertex, self.vertex) if points is None else cdist(points, self.vertex)

    @property
    def distance(self):
        """Get the distance between the (vertex, points) or (vertex, vertex)"""

        return self.get_distance()

    # Property for bounding box (mesh.GetBoundingBox())

    @property
    def box(self):
        return self.mesh.GetBoundingBox()

    @property
    def domain(self):
        return self.box.max()

    # Property for number of boundary elements (mesh.GetNBE())
    @property
    def nbElement(self):
        return self.mesh.GetNBE()

    @property
    def nbEdges(self):
        return self.mesh.GetNEdges()

    @property
    def geoshape(self):
        return self.mesh.GetElementGeometry(0)

    @property
    def getBoundaryEdge(self):
        boundaryEdge = []
        for i in range(self.nbElement):
            boundaryEdge.append(self.mesh.GetBdrElementVertices(i))

        return np.array(boundaryEdge)

    @property
    def boundary_vertex(self, boolean=True):
        bdr_vertex = []
        for i in range(self.mesh.GetNBE()):
            bdr_vertex.append(self.mesh.GetBdrElement(i).GetVerticesArray())

        return np.array(bdr_vertex)

    # %% property: stiff and mass matrix

    @property
    def stiff(self):
        return self._stiff

    @property
    def mass(self):
        return self._mass

    def __getstate__(self):
        # Create a dictionary of the object's state excluding non-picklable attributes
        state = self.__dict__.copy()
        # Exclude the attributes that can't be pickled
        excluded_attrs = ['_mesh', '_fespace', '_mass', '_stiff']
        for attr in excluded_attrs:
            if attr in state:
                del state[attr]
        return state

    def __setstate__(self, state):
        # Restore the object's state
        self.__dict__.update(state)
        # Reinitialize the attributes that were excluded from pickling
        self._mesh = None
        self._fespace = None
        self._mass = None
        self._stiff = None
        # Rebuild the mesh and finite element space if needed
        if 'mesh' in state and state['mesh'] is not None:
            self.mesh = state['mesh']
        else:
            self.mesh, self._inner_points, self._boundary_points = self._build_mesh(
                self.points, triangles=self.triangle,
                add_boundary=self.add_boundary, boundary_poly=self.boundary_poly,
                boundary_step=self.boundary_step, outer_index=self.outer_index,
                level=self.level)

            # Recompute mass and stiffness matrices if needed
        if self._mass is None or self._stiff is None:
            self._mass, self._stiff = self._compute_mass_stiff()


# %% SAR approximation

# # In the geo lattrice model, the Q function is build by considering the SAR model.
# # See the Nyckca 2015 (L = rho, theta = ks)

# # get the hidden lattice
# hlattice = self.grid_latent
# hsize = self.qsize
# offset = self.qpoints[0] if self.lattice == 'regular' else np.nan
# rho = L
# ks = theta

# # Create the sparse B matrix
# main_diag = (4 + ks**2) * np.ones(hsize)   # main diagional
# side_diag = -1 * np.ones(hsize-1)       # up/down main diagonal
# up_down_diag = -1 * np.ones(hsize-offset)    # Periodic up/down element

# B = sp.diags_array([main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
#                    offsets=[0, -1, 1, -offset, offset], shape=(hsize, hsize), format='csc')

# if justB:
#     return B
# else:
#     # create the sparse precision matrix
#     invQ = (1/rho) * (B.T @ B)

#     # compute the covariance matrix and logdet

#     factor = cholesky(invQ)     # compute the sparse cholenky
#     logdetInvQ = factor.logdet()    # compute the logdet
#     # cov = factor.solve_A(np.eye(N)) # compute the inverse matrix

#     # fix later (really time consuming -> use sparse cholensky solve_A)
#     if cov == True:
#         # TODO: FIX INVERSION
#         Q = np.linalg.solve(invQ.toarray(), np.eye(hsize))
#         return Q, invQ.toarray(), logdetInvQ

#     else:
#         return invQ.toarray(), logdetInvQ


# %% Circular embedding approximation CE

# %% La vecchia approx.

class netApproxCov():

    def precision(self, d, dist='Euclidean'):

        return Matern(d)