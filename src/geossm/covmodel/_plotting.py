"""Plotting of finite-element meshes (internal; see `FEMSolver.plot_mesh`).

Kept apart from the numerical classes (Section B, P8). matplotlib is imported
inside the function, so `import geossm` does not load it (F29).
"""

import numpy as np


def plot_mesh(
    fem,
    ax=None,
    figsize=(10, 8),
    title="Title",
    alpha_vertex=1,
    alpha_triangle=0.5,
    alpha_border=0.5,
):
    """
    Plot the mesh: all triangle edges, inner/outer vertices, boundary
    edges (dashed red), and the `domain` polygon(s) (orange outline).

    Parameters
    ----------
    fem : FEMSolver
        The finite-element mesh to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not given.
    figsize : tuple of float, default (10, 8)
        Figure size, only used when `ax` is not given.
    title : str, default "Title"
        Plot title.
    alpha_vertex : float, default 1
        Transparency of the inner/outer vertex markers.
    alpha_triangle : float, optional
        Unused (kept for interface compatibility); triangle edges are
        always drawn with a fixed alpha of 0.5.
    alpha_border : float, default 0.5
        Transparency of the dashed boundary-edge lines.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # imported here, not at module level: matplotlib is only needed for
    # plotting and costs ~0.35 s on `import geossm` (F29)
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    # Convert the vertex array to a numpy array
    vertex = fem.vertex
    triangles = fem.elements

    triang = mtri.Triangulation(vertex[:, 0], vertex[:, 1], triangles)

    # Collect all triangles (elements)
    # inner_triangles = []
    # for i in range(fem.nelement):
    #     tri = fem.mesh.GetElement(i).GetVerticesArray()
    #     inner_triangles.append(tri)
    # inner_triangles = np.array(inner_triangles)

    # Collect all boundary edges
    outer_edges = []
    for i in range(fem.n_boundary_elements):
        bElem = fem.mesh.GetBdrElement(i).GetVerticesArray()
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
    # ax.plot(vertex[fem.inner, 0], vertex[fem.inner, 1],
    #         'xm', label=f'Interior vertices ({sum(fem.inner)}) ', alpha=alpha_vertex)
    # ax.plot(vertex[~fem.inner, 0], vertex[~fem.inner, 1],
    #         'xg', label=f'Outer vertices ({sum(~fem.inner)}) ', alpha=alpha_vertex)

    ax.plot(
        vertex[fem.inner, 0],
        vertex[fem.inner, 1],
        "xm",
        label="Interior vertices",
        alpha=alpha_vertex,
    )
    ax.plot(
        vertex[~fem.inner, 0],
        vertex[~fem.inner, 1],
        "xg",
        label="Outer vertices",
        alpha=alpha_vertex,
    )

    # Plot the boundary vertices
    # ax.plot(fem.outer_points[:, 0], fem.outer_points[:, 1],
    #         'x', color='black', label='Outer Vertices', alpha=alpha_border)

    # Plot the boundary edges as dashed lines
    for edge in boundary_edge:
        x_coords = vertex[edge, 0]
        y_coords = vertex[edge, 1]
        ax.plot(x_coords, y_coords, "--r", alpha=alpha_border)

    for poly in fem.domain:
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
