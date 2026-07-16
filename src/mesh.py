import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path

from scipy.spatial import Voronoi
from utils import ADIMB1, ADIMB2
b1 = ADIMB1
b2 = ADIMB2


def mesh_k_unit_cell(b1, b2, Nx=10, Ny=10):
    """
    Sample one parallelogram unit cell of the reciprocal lattice.
    i/Nx and j/Ny run over [0, 1), so no double-counting at boundaries.
    """
    i = np.arange(0, Nx)
    j = np.arange(0, Ny)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    k_points = (ii[..., None] * b1 / Nx) + (jj[..., None] * b2 / Ny)
    return k_points.reshape(-1, k_points.shape[-1])


def mesh_k_unit_cell_full(b1, b2, Nx=10, Ny=10):
    """
    Sample one parallelogram unit cell of the reciprocal lattice.
    i/Nx and j/Ny run over [0, 1), so no double-counting at boundaries.
    """
    i = np.arange(-Nx, Nx)
    j = np.arange(-Ny, Ny)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    k_points = (ii[..., None] * b1 / Nx) + (jj[..., None] * b2 / Ny)
    return k_points.reshape(-1, k_points.shape[-1])


def map_to_first_BZ(k_points, b1, b2, shells=1):
    """
    Map k-points to the first Brillouin Zone (Voronoi cell of the
    reciprocal lattice) by finding the nearest reciprocal lattice point.

    Parameters
    ----------
    k_points : (N, 2) array
    b1, b2   : (2,) reciprocal lattice vectors
    shells   : int, number of shells of G-vectors to search (1 is
               sufficient for hexagonal; increase if you see artefacts)
    """
    B = np.column_stack([b1, b2])           # (2,2) matrix, columns are b1, b2

    # Step 1: initial reduction into the parallelogram unit cell [-0.5, 0.5)
    B_inv = np.linalg.inv(B)
    frac = k_points @ B_inv.T              # fractional coordinates, (N, 2)
    frac_reduced = frac - np.floor(frac + 0.5)
    k_reduced = frac_reduced @ B.T         # back to Cartesian, (N, 2)

    # Step 2: build all G-vectors in the requested shells — vectorized
    n_range = np.arange(-shells, shells + 1)
    ns = np.array(np.meshgrid(n_range, n_range)).T.reshape(-1, 2)  # ((2s+1)^2, 2)
    G_vectors = ns @ B.T                   # ((2s+1)^2, 2) in Cartesian

    # Step 3: for every k find the G that minimises |k - G|  (fully vectorized)
    # k_reduced: (N, 2), G_vectors: (M, 2)
    diff = k_reduced[:, None, :] - G_vectors[None, :, :]  # (N, M, 2)
    norms = np.linalg.norm(diff, axis=-1)                 # (N, M)
    best = np.argmin(norms, axis=-1)                      # (N,)
    k_first_BZ = k_reduced - G_vectors[best]              # (N, 2)

    return k_first_BZ


def get_BZ_vertices(b1, b2):
    B = np.column_stack([b1, b2])
    ns = np.array([[n1, n2] for n1 in range(-2, 3) for n2 in range(-2, 3)])
    G_ext = ns @ B.T
    vor = Voronoi(G_ext)
    gamma_idx = np.argmin(np.linalg.norm(G_ext, axis=1))
    region_idx = vor.point_region[gamma_idx]
    vertices = vor.vertices[vor.regions[region_idx]]
    angles = np.arctan2(vertices[:, 1], vertices[:, 0])
    return vertices[np.argsort(angles)]


def plot(k_original, k_mapped, b1, b2):
    bz_vertices = get_BZ_vertices(b1, b2)
    bz_closed = np.vstack([bz_vertices, bz_vertices[0]])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, k_pts, title in zip(
        axes,
        [k_original, k_mapped],
        ["Full mesh (Derivatives)", r"Valid points in BZ ($15\times 15$)"]
    ):
        ax.plot(bz_closed[:, 0], bz_closed[:, 1], 'k-', linewidth=1.5)
        ax.add_patch(Polygon(bz_vertices, closed=True, facecolor='lightblue', alpha=0.2))
        ax.scatter(k_pts[:, 0], k_pts[:, 1], s=8, color='royalblue', zorder=3, marker="h")
        ax.scatter(0, 0, s=8, color='red', zorder=4, label='Γ', marker="h")
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        # b1 and b2 arrows
        if title == "Full mesh (Derivatives)":
            for vec, label, color in zip([b1, b2], [r'$\mathbf{b}_1$', r'$\mathbf{b}_2$'], ['darkred', 'darkgreen']):
                ax.quiver(0, 0, vec[0], vec[1],
                        angles='xy', scale_units='xy', scale=1,
                        color=color, width=0.005, zorder=5)
                ax.text(vec[0] * .95, vec[1] * .95, label, fontsize=13, color=color)

    plt.tight_layout()
    plt.savefig('bz_mesh.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    k_original = mesh_k_unit_cell(b1, b2, Nx=15, Ny=15)
    k_full = mesh_k_unit_cell_full(b1, b2, Nx=15, Ny=15)
    k_mapped = map_to_first_BZ(k_original, b1, b2)
    plot(k_original, k_mapped, b1, b2)


if __name__ == "__main__":
    main()