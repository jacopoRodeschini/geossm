#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

import matplotlib.tri as mtri  # For Triangulation object
import mfem.ser as mfem
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy as sc
import gstools as gs
import meshio
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from gstools.covmodel import Matern
from gstools.tools.geometric import latlon2pos, chordal_to_great_circle
from statsmodels.iolib.summary import Summary
from datetime import datetime, timezone
import time


# % Utility functions

def meshio_to_mfem_mesh(meshio_mesh):
    """
    Converts a meshio.Mesh object to an mfem.Mesh object in memory
    by iteratively adding vertices and elements.
    """
    # 1. Determine dimension and prepare vertices
    if meshio_mesh.points.shape[1] == 2:
        # dim = 2
        # MFEM internally works with 3D coordinates, so pad 2D points with zeros
        vertices_mfem = np.hstack(
            [meshio_mesh.points, np.zeros((meshio_mesh.points.shape[0], 1))]
        ).astype(np.float64)
    elif meshio_mesh.points.shape[1] == 3:
        # dim = 3
        vertices_mfem = meshio_mesh.points.astype(np.float64)
    else:
        raise ValueError("meshio_mesh.points must have 2 or 3 columns.")

    # 2. Get cell data and MFEM cell type
    cells_data = None
    cell_type_mfem = -1
    for cell_block in meshio_mesh.cells:
        if cell_block.type == "triangle":
            cells_data = cell_block.data
            cell_type_mfem = mfem.Element.TRIANGLE
            break
        elif cell_block.type == "quad":
            cells_data = cell_block.data
            cell_type_mfem = mfem.Element.QUADRILATERAL
            break
        elif cell_block.type == "tetra":
            cells_data = cell_block.data
            cell_type_mfem = mfem.Element.TETRAHEDRON
            break
        elif cell_block.type == "hexahedron":
            cells_data = cell_block.data
            cell_type_mfem = mfem.Element.HEXAHEDRON
            break

    if cells_data is None or cell_type_mfem == -1:
        raise ValueError(
            "No supported 2D or 3D cells found in meshio_mesh (e.g., triangle, quad, tetra, hexa)."
        )

    # 3. Create an empty MFEM mesh
    # The default constructor mfem.Mesh(dim, num_vertices, num_elements, type, gen_edges=True)
    # is generally for very simple, uniform meshes or when you're building elements later.
    # We will use the mfem.Mesh() default constructor and add elements.
    # mfem_mesh = mfem.Mesh()
    mfem_mesh = mfem.Mesh(2, vertices_mfem.shape[0], cells_data.shape[0], 0, 2)

    # 4. Add vertices to the MFEM mesh
    for i, vert_coord in enumerate(vertices_mfem):
        mfem_mesh.AddVertex(vert_coord)

    # 5. Add elements to the MFEM mesh
    # Assign element attributes (e.g., 1 for the domain)
    # This assumes all domain elements get the same attribute for simplicity.
    # element_attribute = 1  # Common attribute for domain cells
    all_triangles = cells_data
    for i in range(len(all_triangles)):
        mfem_mesh.AddTriangle(
            all_triangles[i, 0], all_triangles[i, 1], all_triangles[i, 2]
        )

    # 6. Finalize the mesh
    # This step is crucial. It builds connectivity tables, boundary elements, etc.
    mfem_mesh.FinalizeMesh()

    # If you want to refine (optional, as in your original code)
    # mfem_mesh.UniformRefinement()

    # Note: If you defined physical groups for boundaries in Gmsh and want to
    # assign specific boundary attributes in MFEM, that's a more advanced step
    # involving iterating through meshio_mesh.cells (for boundary cells) and
    # `meshio_mesh.cell_data` to get the physical group IDs, then adding them
    # to MFEM as `mfem.BdrElement` with corresponding attributes.
    # For now, MFEM's `FinalizeMesh()` will auto-detect boundaries and assign default attributes.

    return mfem_mesh


def _validate_domain(domain):
    """
    Validate and normalize the ``domain`` argument shared by
    :class:`FEMSolver` and :class:`spdeAppoxCov`.

    ``domain`` describes the region(s) of scientific interest, one polygon
    per region (e.g. one polygon per landmass for a multi-polygon country
    such as Italy: Sicily, Sardinia, mainland). It accepts a single shapely
    ``Polygon``, a single ``MultiPolygon``, or a list/tuple mixing either --
    any ``MultiPolygon`` is expanded into its constituent polygons, so the
    return value is always a flat tuple of ``Polygon`` objects.

    ``domain`` is used only to classify *mesh* vertices as inner/outer (see
    ``FEMSolver.isinner``) -- a vertex is inner if it is covered by *any*
    single polygon in ``domain``, not by their convex hull.

    ``domain`` is distinct from the FEM computational domain :math:`\\Omega`:
    :math:`\\Omega` is implicitly defined by the mesh handed to
    ``FEMSolver``/``spdeAppoxCov.setup()``. A reasonable default choice is
    the convex hull of the union of the ``domain`` polygons (extended by an
    offset to limit boundary effects), computed by ``_domain_hull(domain)``
    and exposed as the ``domain_hull`` property on both classes. Neither
    class enforces that the mesh matches this choice -- use
    ``FEMSolver.covers_domain()`` to check.
    """
    if isinstance(domain, (Polygon, MultiPolygon)):
        domain = [domain]
    elif not isinstance(domain, (list, tuple)):
        raise TypeError(
            "domain must be a shapely Polygon/MultiPolygon, or a list/tuple of them"
        )

    polygons = []
    for poly in domain:
        if isinstance(poly, MultiPolygon):
            polygons.extend(poly.geoms)
        elif isinstance(poly, Polygon):
            polygons.append(poly)
        else:
            raise ValueError(
                "Each domain element must be a shapely Polygon or MultiPolygon, "
                f"got {type(poly).__name__}"
            )

    if len(polygons) == 0:
        raise ValueError("domain must contain at least one Polygon")

    return tuple(polygons)


def _domain_hull(domain):
    """
    Convex hull of the union of the ``domain`` polygons -- a reasonable
    default choice of FEM computational domain :math:`\\Omega`, e.g. to pass
    as ``boundary`` to ``buildMesh2d``. See ``_validate_domain``.
    """
    return unary_union(domain).convex_hull


# % FEM solver class
class FEMSolver:
    def __init__(self, meshio_obj: meshio.Mesh, domain=None, verbose=True, stats=True):
        """
        Parameters
        ----------
        meshio_obj : meshio.Mesh
            The FEM mesh, covering the full computational domain
            :math:`\\Omega` (e.g. `domain_hull`, the convex hull of the
            union of `domain`).
        domain : shapely.geometry.Polygon or MultiPolygon, or a list/tuple
            of them, optional
            Region(s) of scientific interest, used only to classify mesh
            vertices as inner/outer (`isinner`); see `_validate_domain` for
            the full contract. Defaults to the convex hull of the mesh
            vertices, i.e. every vertex is treated as inner.
        verbose : bool, optional
            Whether to log progress messages.
        stats : bool, optional
            Whether to compute mesh quality statistics (angles, areas).
        """

        # Validate inputs
        mesh = None
        self.verbose = verbose
        
        try:
            if meshio_obj is not None:
                if not isinstance(meshio_obj, meshio.Mesh):
                    raise TypeError(
                        f"meshio_obj must be meshio.Mesh, got {type(meshio_obj).__name__}"
                    )
                # Convert to MFEM if needed (or keep as meshio)
                self._log("Converting meshio.Mesh to mfem.Mesh...")
                mesh = meshio_to_mfem_mesh(meshio_obj)
                self._log("Mesh conversion successful.")

        except Exception as e:
            raise RuntimeError(f"Error loading mesh: {str(e)}") from e

        if not hasattr(mesh, "Dimension"):
            raise TypeError(
                f"mesh must be an mfem.Mesh object, got {type(mesh).__name__}"
            )

        if domain is not None:
            domain = _validate_domain(domain)

        # Store mesh
        self._mesh = mesh

        # Fall back to the convex hull of the mesh vertices so every vertex
        # is treated as inner (no marginalisation) when no domain is given.
        self._domain = domain if domain is not None else (MultiPoint(self.vertex[:, :2]).convex_hull,)

        # Compute the stats associate with the mesh (angles and areas of the triangles)
        if stats == True:
            self._log("Computing mesh quality statistics:angles and areas (takes a while)...")
            self._angles, self._areas = self.compute_stats()
            self._log("Mesh quality statistics computed successfully.")
        else:
            self._angles, self._areas = None, None
        
        # Print the mesh quality statistics
        if self._angles is not None and self._areas is not None:
            msg = f"Angles [min, mean, max] = [{self._angles.min():.2f}, {self._angles.mean():.2f}, {self._angles.max():.2f}] degrees \n"
            msg += f"Areas  [min, mean, max] = [{self._areas.min():.2f}, {self._areas.mean():.2f}, {self._areas.max():.2f}]"
            self._log(msg)

        # Get warning if the mesh has bad quality (e.g. small angles)
        if self._angles is not None and self._angles.min() < 10:
            self._log(
                f"Mesh has small angles (min angle = {self._angles.min():.2f} degrees). "
                "This may lead to numerical instability. Consider refining the mesh or improving its quality."
            )
        

        
        # Initialize matrices as None (will be computed)
        self._fespace = None
        self._mass = None
        self._stiff = None
        self._inner = None

        # Build finite element space
        self._log("Building FE space and computing mass/stiffness matrices...")
        self._fespace = self._build_fespace()

        # Classify vertices as inner or outer (boolean array)
        self._inner = self.isinner()

        # Warn if the mesh's convex hull does not fully cover the domain's
        # (see `covers_domain`); this signals outer vertices may be missing
        # near the region of interest, biasing the Neumann boundary approx.
        if not self.covers_domain():
            self._log(
                "Mesh does not fully cover convex_hull(unary_union(domain)) "
                "(see `covers_domain()`/`domain_hull`). Outer vertices needed "
                "to push boundary effects away from the domain of interest "
                "may be missing; consider rebuilding the mesh over `domain_hull`."
            )

        # compute matrices now
        self._log("Computing mass and stiffness matrices...")
        self._mass, self._stiff = self._compute_mass_stiff()
        self._log("Computing mass and stiffness matrices... Done.")
        self._log("FEMSolver initialization complete.")

    @property
    def mesh(self):
        return self._mesh

    @property
    def inner(self):
        return self._inner

    def _build_fespace(self):

        # Create a finite element space
        # Define a finite element space on the mesh. Here we use vector finite
        # elements, i.e. dim copies of a scalar finite element space. The vector
        # dimension is specified by the last argument of the FiniteElementSpace
        # constructor.
        # Order 1 finite elements
        try:
            fec = mfem.H1_FECollection(1, self.mesh.Dimension())
            fespace = mfem.FiniteElementSpace(self.mesh, fec)
            return fespace

        except Exception as e:
            raise RuntimeError(f"Failed to build finite element space: {str(e)}")

    def isinner(self, points=None):
        """
        Classify vertices as inner or outer with respect to `domain`.

        A point is inner if it is covered by *any single* polygon in
        `domain` -- each polygon is tested on its own (not on the convex
        hull of their union), so for a multi-polygon domain (e.g. Sicily,
        Sardinia, mainland Italy) a mesh vertex sitting between two
        landmasses, inside their shared convex hull but outside every
        individual polygon, is classified as outer.
        """
        # Use provided domain polygons
        if points is not None:
            phy_points = np.asarray(points, dtype=np.float64)
        else:
            phy_points = np.asarray(self.vertex, dtype=np.float64)

        inner = np.logical_or.reduce(
            [
                np.array([poly.covers(Point(p)) for p in phy_points])
                for poly in self._domain
            ]
        )
        return inner

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
        mass = sp.diags(temp, offsets=0, shape=(self.ndofs, self.ndofs), format="csr")

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

        i = (
            spmat.GetIArray()
        )  # Row pointers (cumulative sums of non-zero elements per row)
        j = spmat.GetJArray()  # get index of j (column)
        dt = spmat.GetDataArray()  # Non zero values of the sparse matrix

        # Build the stifness sparse matrix G using the CSR format
        stiff = sp.csr_matrix((dt, j, i), shape=(self.ndofs, self.ndofs), copy=True)

        # cut on the effective dofs
        return (
            mass[: self.effective_dofs, :][:, : self.effective_dofs],
            stiff[: self.effective_dofs, :][:, : self.effective_dofs],
        )

    def getBasis(self, phy_points=None):
        count, notfindInx, H = self._compute_basis(phy_points)
        return count, notfindInx, H

    def _compute_basis(self, phy_points=None, thr=1e-5):
        # @Points = physical point

        # Create the list of pysical points
        if phy_points is None:
            phy_points = np.asarray(self.vertex, dtype=np.float64)  # total grid points
        else:
            phy_points = np.asarray(phy_points, dtype=np.float64)

        npoint = phy_points.shape[0]

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
        count, elem_ids, int_points = self.mesh.FindPoints(phy_points)
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
                phys_point.Assign(phy_points[i, :])
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
        # if len(notfindInx) != 0:
        #     s = "The following point index need to be removed because can't be find in the latent domain \n"
        #     s += "See the MFEM FindPoints function documentatios \n"
        #     s += f"Index {notfindInx} \n"
        #     warnings.warn(s)

        notfindInx = np.asarray(notfindInx)

        return count, notfindInx, H[:, : self.effective_dofs]

    def plot_mesh(
        self,
        ax=None,
        figsize=(10, 8),
        title="Title",
        alpha_vertex=1,
        alpha_triangle=0.5,
        alpha_border=0.5,
    ):

        # Convert the vertex array to a numpy array
        vertex = self.vertex
        triangles = self.elements

        triang = mtri.Triangulation(vertex[:, 0], vertex[:, 1], triangles)

        # Collect all triangles (elements)
        # inner_triangles = []
        # for i in range(self.nelement):
        #     tri = self.mesh.GetElement(i).GetVerticesArray()
        #     inner_triangles.append(tri)
        # inner_triangles = np.array(inner_triangles)

        # Collect all boundary edges
        outer_edges = []
        for i in range(self.nbElements):
            bElem = self.mesh.GetBdrElement(i).GetVerticesArray()
            outer_edges.append(bElem)

        boundary_edge = np.array(outer_edges)

        # Plot the mesh
        # fig, ax = plt.subplots()

        # Handle the 'ax' argument
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        else:
            _ = ax.figure  # Get the figure from the provided axes
            
        ax.triplot(triang, "k-", lw=0.3, alpha=0.5, label="All Mesh Edges")

        # Plot the interior vertices
        # ax.plot(vertex[self.inner, 0], vertex[self.inner, 1],
        #         'xm', label=f'Interior vertices ({sum(self.inner)}) ', alpha=alpha_vertex)
        # ax.plot(vertex[~self.inner, 0], vertex[~self.inner, 1],
        #         'xg', label=f'Outer vertices ({sum(~self.inner)}) ', alpha=alpha_vertex)

        ax.plot(
            vertex[self.inner, 0],
            vertex[self.inner, 1],
            "xm",
            label="Interior vertices",
            alpha=alpha_vertex,
        )
        ax.plot(
            vertex[~self.inner, 0],
            vertex[~self.inner, 1],
            "xg",
            label="Outer vertices",
            alpha=alpha_vertex,
        )

        # Plot the boundary vertices
        # ax.plot(self.outer_points[:, 0], self.outer_points[:, 1],
        #         'x', color='black', label='Outer Vertices', alpha=alpha_border)

        # Plot the boundary edges as dashed lines
        for edge in boundary_edge:
            x_coords = vertex[edge, 0]
            y_coords = vertex[edge, 1]
            ax.plot(x_coords, y_coords, "--r", alpha=alpha_border)

        for poly in self.domain:
            x, y = poly.exterior.xy
            ax.plot(x, y, "-", color="orange", alpha=1)

        # Add a legend
        ax.legend()

        # Display the plot
        ax.set_title(title)
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.6)
        # plt.show()
        return ax

    def get_distance(self, points=None):
        """Get the distance between the (vertex, points) or (vertex, vertex)"""
        return (
            cdist(self.vertex, self.vertex)
            if points is None
            else cdist(points, self.vertex)
        )

    def distance(self, points=None):
        """Get the distance between the (vertex, points) or (vertex, vertex)"""

        return self.get_distance(points=points)

    @property
    def vertex(self):
        """
        Returns the coordinates of mesh vertices that have a DOF associated
        with them in the given H1_FECollection(1, dim) finite element space.
        """
        # Get total number of unique vertices in the mesh
        all_mesh_vertices = np.array(self._mesh.GetVertexArray())

        # Get the mapping from element-local DOFs to global DOFs
        # For H1 degree 1, local DOFs are vertex indices within the element
        # Global DOFs are what we are interested in.

        # We can get the *global* vertex indices that correspond to DOFs
        # The fespace.GetVSize() is the total number of DOFs.
        # For H1 degree 1, each DOF corresponds to a unique vertex.

        dof_map_vertices = []

        # Iterate through all elements in the mesh
        for i in range(self._mesh.GetNE()):
            # Get the global vertex indices for the current element
            element_vertex_indices = self._mesh.GetElementVertices(
                i
            )  # This should return a Python list

            # Add these to our set of unique DOF-associated vertex indices
            for v_idx in element_vertex_indices:
                dof_map_vertices.append(v_idx)

        # Get unique vertex indices
        unique_dof_vertex_indices = sorted(list(set(dof_map_vertices)))

        # Now, retrieve the coordinates for these unique vertices
        dof_associated_vertex_coords = all_mesh_vertices[unique_dof_vertex_indices]

        return dof_associated_vertex_coords

    @property
    def nvertex(self):
        return len(self.vertex)

    @property
    def nelements(self):
        """Get the number of discretisation elements (like the number of triangle)"""
        return self.mesh.GetNE()

    @property
    def elements(self):
        triangles = []

        for i in range(self.mesh.GetNE()):
            # Get the list of vertex IDs for element 'i'
            # For a triangle mesh, this will always return 3 integers
            triangles.append(self.mesh.GetElementVertices(i))

        # 3. Convert to a structured NumPy array
        triangles_array = np.array(triangles)

        return triangles_array

    @property
    def box(self):

        box  = self.mesh.GetBoundingBox()
        box = np.round(box, 2).tolist()

        box_flat = [x for sublist in box for x in sublist]
 
        return box_flat # [minx, miny, maxx, maxy]

    @property
    def domain(self):
        return self._domain

    @property
    def domain_hull(self):
        """Convex hull of the union of `domain` -- see `_domain_hull`."""
        return _domain_hull(self._domain)

    def covers_domain(self):
        """
        Check that the mesh covers the FEM computational domain Ω.

        Simple necessary (not sufficient) check: whether the convex hull of
        the mesh vertices covers `domain_hull`, the convex hull of the union
        of `domain`. Does not check the mesh's actual (possibly non-convex)
        boundary shape, only its convex extent.
        """
        mesh_hull = MultiPoint(self.vertex[:, :2]).convex_hull
        return mesh_hull.covers(self.domain_hull)

    # Property for number of boundary elements (mesh.GetNBE())
    @property
    def nbElements(self):
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

    @property
    def fespace(self):
        """Return the finite elment space"""
        return self._fespace

    # Property for the space dimension (fespace.GetVDim())
    @property
    def fespace_order(self):
        return self._fespace.GetVDim()

    # Property for number of local degrees of freedom (fespace.GetNDofs())
    @property
    def ndofs(self):
        return self._fespace.GetNDofs()

    @property
    def effective_dofs(self):
        """Return the effective number of DOFs (associated with vertices)"""
        return len(self.vertex)

    # Property for number of vector DOFs (fespace.GetVSize())
    @property
    def GetVSize(self):
        return self._fespace.GetVSize()

    @property
    def stiff(self):
        return self._stiff

    @property
    def mass(self):
        return self._mass

    # Property for number of vertices (mesh.GetNV())
    @property
    def inner_points(self):
        """Return the point of the grid"""
        return self.vertex[self.inner]

    @property
    def n_inner_points(self):
        """Return the point of the grid"""
        return self.inner.sum()

    @property
    def outer_points(self):
        """Return the boundary point of the grid"""
        return self.vertex[~self.inner]

    @property
    def n_outer_points(self):
        """Return the number of the boundary point of the grid"""
        return (~self.inner).sum()

    @property
    def totpoints(self):
        """Return the total point (inner + boundary) of the grid (same as vertex)"""
        return self.vertex

    @property
    def n_totpoints(self):
        """Return the total point number of the grid"""
        return self.nvertex

    @property
    def shape(self):
        return (self.nvertex, self.nelements, self.nbElements)


    def compute_angles(self, vertex, triangle):
        A, B, C = vertex[triangle]

        def angle(a, b, c):
            ab = np.linalg.norm(b - a)
            ac = np.linalg.norm(c - a)
            bc = np.linalg.norm(c - b)
            cos_theta = (ab**2 + ac**2 - bc**2) / (2 * ab * ac)
            # Convert to degrees
            return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

        return np.array([angle(A, B, C), angle(B, C, A), angle(C, A, B)])

    def compute_area(self, vertex, triangle):
        A, B, C = vertex[triangle]
        return 0.5 * abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))


    def compute_stats(self):

        # compute angles for all triangles
        angles = np.array([self.compute_angles(self.vertex, tri) for tri in self.elements])
        
        # compute areas for all triangles
        areas = np.array([self.compute_area(self.vertex, tri) for tri in self.elements])

        return angles, areas

    
    @property
    def angles(self):
        return self._angles
    
    @property
    def areas(self):
        return self._areas
    

    def generate_summary(self):
        # compute the angles and areas of the triangles

        top_left = dict(
                    [
                        ("Solver. type:", lambda: [self.__class__.__name__]),
                        #("Scale (kappa):", lambda: f"{self.rescale:.2f}"),
                        #("Range (theta):", lambda: f"{np.sqrt(8)/self.range:.2f}"),
                        #("Variance (s2):", lambda: f"{self.var:.2f}"),
                        #("Nu:", lambda: f"{self.nu:.2f}"),
                        ("Mesh vertex",lambda: [f"{self.nvertex}"]),
                        ("Mesh triangles:", lambda: [f"{self.nelements}"]),
                        ("Mesh lines:", lambda: [f"{self.nbElements}"]), 
                        ("Mesh inner vertex (rank):", lambda: [f"{self.n_inner_points}"]),
                        ("Mesh outer vertex:", lambda: [f"{self.n_outer_points}"]),   
                    ]
                )
        if self.angles is None or self.areas is None:
            top_left["Mesh angle [min, mean, max]:"] = lambda: ["N/A"]
            top_left["Mesh area [min, mean, max]:"] = lambda: ["N/A"]
        else:
            top_left["Mesh angle [min, mean, max]:"] = lambda: [f"[{self.angles.min():.2f}, {self.angles.mean():.2f}, {self.angles.max():.2f}]"]
            top_left["Mesh area [min, mean, max]:"] = lambda: [f"[{self.areas.min():.2f}, {self.areas.mean():.2f}, {self.areas.max():.2f}]"]
            


        top_right = dict(
                    [
                        ("Box:", lambda: [f"{self.box}"]),
                        ("FE space order:", lambda: [f"{self.fespace_order}"]),
                        ("Mesh DOFs:", lambda: [f"{self.ndofs}"]),
                        ("Mesh shape:", lambda: [f"{self.shape}"]),
                        ("Mass matrix shape:", lambda: [f"{self.mass.shape}"]),
                        ("Stiff matrix shape:", lambda: [f"{self.stiff.shape}"]),
                        ("Inner indx. (shape)", lambda: [f"{self.inner.shape}"]),
                        ("Inner indx. (sum)", lambda: [f"{(self.inner).sum()}"]),
                    ]
                )

        # Generate the dictionaly
        gen_top_left = []
        for item in top_left.keys():
            gen_top_left.append((item, list(top_left[item]())))

        gen_top_right = []
        for item in top_right.keys():
            gen_top_right.append((item, top_right[item]()))

        return gen_top_left, gen_top_right

    def summary(self):

        
        # Add the header to the summary
        gen_top_left, gen_top_right = self.generate_summary()
        
        smry = Summary()
        smry.add_table_2cols(
            self,
            title="Covariance FEM solver",
            gleft=gen_top_left,
            gright=gen_top_right,
            yname= "N/A",
            xname= "N/A",
        )
  
        return smry

    def __str__(self):
        return self.summary().as_text()
    
    def _is_verbose(self, verbose=None) -> bool:
        return self.verbose if verbose is None else verbose

    def _log(self, msg: str, verbose=None) -> None:
        if self._is_verbose(verbose):
            self.print_info(msg)

    def print_info(self, msg):
        dt = datetime.fromtimestamp(time.time(), tz=timezone.utc)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


# % Heat kernel

# Euclidean distance is wrong in a non-convex domain like PacMan.
# hdist = cdist(points, points)

# Graph shortest path that stays entirely inside the polygon
# 1 - Compute distances on a FEM mesh graph (Dijkstra on barycentric graph)
# 2 : Laplacian distance (encodes how information diffuses through geometry)


# solve the hteat equation
# def heat_geodesic_kernel(fem_solver, i):
#     fes = femsolver._fespace
#     mesh = femsolver._mesh

#     # setup the mass and stiff matrix
#     M = mfem.BilinearForm(fes)
#     M.AddDomainIntegrator(mfem.MassIntegrator())
#     M.Assemble()
#     M.Finalize()
#     Mmat = M.SpMat()

#     K = mfem.BilinearForm(fes)
#     K.AddDomainIntegrator(mfem.DiffusionIntegrator())
#     K.Assemble()
#     K.Finalize()
#     Kmat = K.SpMat()

#     # solve the heat equation
#     t = mesh.GetElementSize(0)**2  # rule of thumb

#     u = mfem.GridFunction(fes)
#     rhs = mfem.GridFunction(fes)

#     rhs.Assign(0.0)
#     rhs[i] = 1.0  # source of the heat

#     A = mfem.Add(1.0, Mmat, t, Kmat)

#     solver = mfem.CGSolver()
#     solver.SetOperator(A)
#     solver.SetRelTol(1e-8)
#     solver.Mult(rhs, u)

#     # % Compute normalized gradient field
#     dim = mesh.Dimension()
#     vfes = mfem.FiniteElementSpace(mesh, fes.FEColl(), dim)

#     #  Extract the gradient of the heat solution 'u'
#     X = mfem.GridFunction(vfes)
#     grad_u_coeff = mfem.GradientGridFunctionCoefficient(u)

#     # Perform an L2 projection (averaging the gradients from neighboring elements)
#     # This is actually okay for smoothing, but the normalization still has to happen point-by-point.
#     X.ProjectCoefficient(grad_u_coeff)

#     # Pointwise normalization
#     data = X.GetDataArray()  # (this is a view, not a copy)
#     data_reshaped = data.reshape(-1, dim)  # (num_nodes, dimension)

#     # Normalize: -grad(u) / |grad(u)|
#     norms = np.linalg.norm(data_reshaped, axis=1, keepdims=True) + 1e-12
#     data_reshaped[:] = -data_reshaped / norms

#     # # X is already normalized
#     # print(np.max(np.abs(np.linalg.norm(X.GetDataArray().reshape(-1, vdim),
#     #                                    axis=1) - 1.0)))

#     # %Set up the final Poisson problem: \Delta d = \nabla \cdot (\phi)

#     divX = mfem.GridFunction(fes)
#     divX.ProjectCoefficient(mfem.DivergenceGridFunctionCoefficient(X))

#     phi = mfem.GridFunction(fes)

#     solver.SetOperator(Kmat)
#     solver.Mult(divX, phi)

#     return phi.GetDataArray()

# Compute the laplacian godesis distance using the heat kernel

# femsolver = cov_matern.setup(meshio).fem_solver

# hdist = np.zeros((femsolver.nvertex, femsolver.nvertex))
# for i, s in enumerate(points):
#     hdist[i, :] = heat_geodesic_kernel(femsolver, i)

# # cut on the inner points
# hdist = hdist[:, femsolver.inner][femsolver.inner, :]
# hdist = (hdist + hdist.T)/2


# % SPDE Approximation of the Matern covariance model
class spdeAppoxCov(Matern):
    r"""The SPDE approximation of the Matérn covariance model.

    Solves the SPDE :math:`(\kappa^2 - \Delta)^{\alpha/2} x = W` by a finite
    element (FEM) discretisation with Neumann boundary conditions on a mesh
    covering the domain (see `setup`), giving the sparse precision matrix
    `precision()` -- optionally marginalised onto the "inner" vertices of
    `domain` via `precision(marginal=True)`, which is what other models
    (e.g. `LRStateSpaceModel`) should generally use.

    Notes
    -----
    `spdeAppoxCov` extends `gstools.covmodel.Matern`, but only `precision()`
    (and `emp_range`, `sigma2k`, `distance()`) are re-defined in terms of the
    FEM discretisation. The methods inherited unchanged from `Matern` --
    `variogram`, `covariance`, `correlation`, `spectrum`, `spectral_density`,
    and the `*_yadrenko` variants -- describe the *nominal*, stationary,
    infinite-domain Matérn process that this SPDE approximates, using
    `rescale` as :math:`\kappa`. They are exact only in the far interior of
    the mesh, away from the boundary; they do **not** reflect the finite
    mesh's boundary effects, nor the Schur-complement marginalisation used
    by `precision(marginal=True)`. Use them for diagnostics/plotting
    against the nominal target kernel, not as the exact model -- for that,
    use `precision()`/`precision(marginal=True)`.

    References
    ----------
    .. [Rasmussen2003] Rasmussen, C. E.,
           "Gaussian processes in machine learning." Summer school on
           machine learning. Springer, Berlin, Heidelberg, (2003)
    """

    def __init__(
        self, domain, latlon=True, geo_scale=gs.DEGREE_SCALE, nu=1, var=1.0, rescale=1.0, verbose = True
    ):
        """
        Parameters
        ----------
        domain : shapely.geometry.Polygon or MultiPolygon, or a list/tuple
            of them
            Region(s) of scientific interest, e.g. one polygon (or a single
            MultiPolygon) per landmass for a multi-polygon country (Sicily,
            Sardinia, mainland Italy). Any MultiPolygon is expanded into its
            constituent polygons. Forwarded to `FEMSolver` on `setup()`,
            where it is used only to classify mesh vertices as inner/outer
            -- see `_validate_domain` for the full contract, including how
            it differs from the FEM computational domain :math:`\\Omega`
            (implicitly given by the mesh passed to `setup()`; see
            `domain_hull` for a reasonable default choice).
        latlon, geo_scale, nu, var, rescale : see `gstools.covmodel.Matern`.
        verbose : bool, optional
            Whether to log progress messages.
        """

        # Validate domain (same contract as FEMSolver, see _validate_domain)
        self.verbose = verbose
        self._domain = _validate_domain(domain)
        self._ndomain = len(self._domain)
        self._log(f"Validated domain: {self._ndomain} polygon(s).")

        # Mesh storage
        self._meshIO = None

        # FEM components
        self._fem_solver = None

        # Initialize parent Matern class
        self._log(f"Initializing Matern covariance model (nu={nu}, rescale={rescale})...")
        super().__init__(
            dim=2,
            var=var,
            len_scale=1.0,
            nugget=0.0,
            anis=1.0,
            angles=0.0,
            rescale=rescale,
            latlon=latlon,
            geo_scale=geo_scale,
            temporal=False,
            spatial_dim=2,
            nu=nu,
        )
        self._log("spdeAppoxCov initialization complete.")

    @property
    def meshIO(self):
        if self._meshIO is None:
            raise RuntimeError("Mesh not loaded. Call setup() with a mesh first.")
        return self._meshIO

    @property
    def fem_solver(self):
        if self._fem_solver is None:
            raise RuntimeError("FEM solver not initialized. Call setup() first.")
        return self._fem_solver

    def __str__(self):
        # Get the string representation from the parent class
        base = super(Matern, self).__str__()

        if self._fem_solver is not None:
            base += f"\n {str(self.fem_solver)}"

        else:
            base += "\n - FEM solver not initialized. Call setup() with a mesh to initialize."

        return base

    def setup(self, mesh_obj: meshio._mesh.Mesh, stats=True):
        """
        Initialize the covariance model with a mesh.

        This method must be called after instantiation to set up the FEM
        discretization. Provide either meshpath or mesh_obj, not both.

        Parameters
        ----------
        mesh_obj : meshio.Mesh, optional
            A meshio.Mesh object (already loaded in memory)
        stats : bool, optional
            Whether to compute mesh quality statistics (default is True)

        Returns
        -------
        self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If neither or both arguments are provided
        IOError
            If meshpath file cannot be read
        RuntimeError
            If mesh is invalid or FEM setup fails

        Examples
        --------
        >>> mesh = meshio.read("my_mesh.msh")
        >>> cov.setup(mesh_obj=mesh)
        """
        # Load mesh
        self._log("Loading mesh for SPDE FEM discretization...")
        try:
            if mesh_obj is not None:
                if not isinstance(mesh_obj, meshio.Mesh):
                    raise TypeError(
                        f"mesh_obj must be meshio.Mesh, got {type(mesh_obj)}"
                    )
        except Exception as e:
            raise RuntimeError(f"Error loading mesh: {str(e)}") from e

        # Initialize FEM solver
        try:
            self._meshIO = mesh_obj  # Store the mesh for potential reinitialization
            self._log("Initializing FEM solver...")
            self._fem_solver = FEMSolver(mesh_obj, domain=self._domain, verbose=self.verbose, stats=stats)
            self._log("FEM solver initialized.")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize FEM solver: {str(e)}") from e

        return self

    def _compute_precision_spde(self, rescale=None):
        """
        @rescale = rescale factor
        Compute the precision matrix (Q, sparse) of the process y with marginal variance
        sigma2(k) = sigma2_spde

        """

        # update the geo parameter
        # self._update_geo_parameter(rescale)
        if rescale is not None:
            self.rescale = rescale  # Rescale factor (k)

        effective_dofs = self.fem_solver.effective_dofs

        # Compute the inverse of the mass matrix
        Cinv = sp.diags(
            1 / self.fem_solver.mass.diagonal(),
            offsets=0,
            shape=(effective_dofs, effective_dofs),
            format="csr",
        )

        # Compute the K matrix
        k = self.rescale
        K = (k**2) * self.fem_solver.mass + self.fem_solver.stiff

        # Compute the precision matrix of the process with unit marginal variance
        Q = self.sigma2k * (K @ Cinv @ K)

        return Q

    def precision(self, rescale=None, marginal=False):
        """
        Compute the precision matrix (Q) for the spatial process z with
        marginal variance sigma2_process.

        Parameters
        ----------
        rescale : float, optional
            If given, updates `self.rescale` (the SPDE kappa) before
            computing the precision matrix.
        marginal : bool, optional
            If True, return the precision matrix marginalised (via the
            Schur complement, see `_schur_marginal_precision`) onto the
            `domain` inner vertices only, integrating out the outer
            (boundary) vertices -- i.e. the actual precision of the latent
            field once the mesh's boundary effects have been removed, as
            used e.g. by `LRStateSpaceModel`. Default: False, i.e. the full
            FEM precision matrix over every mesh vertex (inner and outer).

        Returns
        -------
        scipy.sparse matrix if `marginal=False`, else a dense numpy array
            (the Schur complement is generally dense even though `Q` is
            sparse).
        """
        Q = self._compute_precision_spde(rescale)
        if not marginal:
            return Q
        return self._schur_marginal_precision(Q, self.fem_solver.inner)

    @staticmethod
    def _schur_marginal_precision(Q, inner_mask):
        """
        Marginal precision matrix of the "inner" (domain-of-interest)
        vertices, `Q_11 - Q_12 @ inv(Q_22) @ Q_12.T`, via the Schur
        complement of the full SPDE precision matrix `Q` with respect to
        the "outer" (boundary) vertices.

        Computes the middle term as `solve(Q_22, Q_12.T)` instead of
        `inv(Q_22) @ Q_12.T`: solving directly for the (n_outer, n_inner)
        right-hand side `Q_12.T` avoids the wasted work of inverting all of
        Q_22 (an (n_outer, n_outer) matrix) when only its action on Q_12.T
        is ever used, and is the more numerically stable formulation.
        """
        inx = np.asarray(inner_mask, dtype=bool)
        Q_11 = Q[inx, :][:, inx]
        Q_12 = Q[inx, :][:, ~inx]
        Q_22 = Q[~inx, :][:, ~inx]

        Q_22_dense = Q_22.toarray() if sp.issparse(Q_22) else Q_22
        Q_12_dense = Q_12.toarray() if sp.issparse(Q_12) else Q_12

        return Q_11 - Q_12 @ np.linalg.solve(Q_22_dense, Q_12_dense.T)

    def distance(self, points=None):
        """
        Pairwise distance between mesh vertices (or `points` and mesh
        vertices).

        Honors `latlon`/`geo_scale` the same way the parent `Matern` model
        does: great-circle distance (in `geo_scale` units) when
        `latlon=True`, flat Euclidean distance on the raw mesh/point
        coordinates otherwise.

        Note this only affects distance-based diagnostics (e.g. comparing
        against `variogram`/`covariance`, see the class Notes) -- the FEM
        stiffness/mass matrices behind `precision()` are always assembled
        directly on the raw mesh coordinates with a flat metric, regardless
        of `latlon`. A raw lon/lat mesh under `latlon=True` therefore still
        carries anisotropic distortion (up to ~cos(latitude)) in the FEM
        operator that this distance correction does not fix; for a
        geodesically accurate SPDE solve, build the mesh in a projected
        (e.g. UTM/km) CRS instead.
        """
        if not self.latlon:
            return self._fem_solver.distance(points=points)

        vertex = self._fem_solver.vertex[:, :2]
        query = vertex if points is None else np.asarray(points, dtype=np.float64)[:, :2]

        # lon/lat (degrees) -> chordal 3D positions on a `geo_scale`-radius
        # sphere -> honest great-circle distance, matching how the parent
        # Matern's vario_yadrenko/cov_yadrenko/cor_yadrenko interpret `r`.
        pos_vertex = latlon2pos([vertex[:, 1], vertex[:, 0]], radius=self.geo_scale)
        pos_query = latlon2pos([query[:, 1], query[:, 0]], radius=self.geo_scale)
        chordal = cdist(pos_query.T, pos_vertex.T)
        return chordal_to_great_circle(chordal, radius=self.geo_scale)

    # property:: spatial process
    @property
    def emp_range(self):
        """Return the empirical range paramiter"""
        return np.sqrt(8 * self.nu) / self.rescale

    @property
    def sigma2k(self):
        """
        Return the marginal variance of the standardise approximate spatial SPDE process
        Variance of the aproximate field x(u). Eq. 2 and Eq. 9
        """
        return sc.special.gamma(1) / (
            sc.special.gamma(2) * 4 * np.pi * (self.rescale**2)
        )

    @property
    def domain(self):
        """
        Region(s) of scientific interest (see `_validate_domain`). Once
        `setup()` has been called, delegates to `fem_solver.domain` so the
        FEM solver is the single source of truth.
        """
        return self._fem_solver.domain if self._fem_solver is not None else self._domain

    @property
    def domain_hull(self):
        """
        Convex hull of the union of `domain` -- a reasonable default choice
        of FEM computational domain Ω, e.g. to pass as `boundary` to
        `buildMesh2d` before calling `setup()`. See `_domain_hull`.
        """
        return _domain_hull(self.domain)

    def covers_domain(self):
        """
        Check that the mesh (set via `setup()`) covers `domain_hull`. See
        `FEMSolver.covers_domain`.
        """
        return self.fem_solver.covers_domain()


    def generate_summary(self):
        
        top_left = dict(
                    [
                        ("Cov. type:", lambda: [self.__class__.__name__]),
                        ("Scale (kappa):", lambda: [f"{self.rescale:.2f}"]),
                        ("Range (theta):", lambda: [f"{np.sqrt(8*self.nu)/self.rescale:.2f}"]),
                    ]
                )
        top_right = dict(
                    [
                        ("Variance (s2):", lambda: [f"{self.var:.2f}"]),
                        ("Sigma2(k):", lambda: [f"{self.sigma2k:.2f}"]),
                        ("Nu:", lambda: [f"{self.nu:.2f}"]),
                        
                    ]
                )

        # Generate the dictionaly
        gen_top_left = []
        for item in top_left.keys():
            gen_top_left.append((item, list(top_left[item]())))

        gen_top_right = []
        for item in top_right.keys():
            gen_top_right.append((item, top_right[item]()))

        
        if hasattr(self,"fem_solver"):
            gen_top_left_solver, gen_top_right_solver = self.fem_solver.generate_summary()
       
            gen_top_left = gen_top_left + gen_top_left_solver
            gen_top_right = gen_top_right + gen_top_right_solver
            

        return gen_top_left, gen_top_right

    def summary(self):

        gen_top_left, gen_top_right = self.generate_summary()

        
        # Add the header to the summary
        
        smry = Summary()
        smry.add_table_2cols(
            self,
            title="Matern SPDE Approximation",
            gleft=gen_top_left,
            gright=gen_top_right,
            yname= "N/A",
            xname= "N/A",
        )
  
        return smry


    def _is_verbose(self, verbose=None) -> bool:
        return self.verbose if verbose is None else verbose

    def _log(self, msg: str, verbose=None) -> None:
        if self._is_verbose(verbose):
            self.print_info(msg)

    def print_info(self, msg):
        dt = datetime.fromtimestamp(time.time(), tz=timezone.utc)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

    def __getstate__(self):
        # Create a dictionary of the object's state excluding non-picklable attributes
        state = self.__dict__.copy()
        # Exclude the attributes that can't be pickled
        excluded_attrs = ["_fem_solver", "_mesh"]
        for attr in excluded_attrs:
            if attr in state:
                del state[attr]
        return state

    def __setstate__(self, state):
        # Restore the object's state
        self.__dict__.update(state)
        # Reinitialize the attributes that were excluded from pickling
        self._mesh = None

        # Rebuild the mesh and finite element solver if meshIO is available
        if self._meshIO is not None:
            self.setup(self._meshIO)

    

