import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path

from scipy.spatial import Voronoi
from utils import ADIMB1, ADIMB2
b1 = ADIMB1
b2 = ADIMB2


def mesh_k_unit_cell(b1, b2, Nx=10, Ny=10):
    i = np.arange(0, Nx)
    j = np.arange(0, Ny)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    k_points = (ii[..., None] * b1 / Nx) + (jj[..., None] * b2 / Ny)
    return k_points.reshape(-1, k_points.shape[-1])


def mesh_k_ghost_shell(b1, b2, Nx=10, Ny=10):
    """Return the ghost shell at i=-1 and i=Nx+1 (and j=-1, j=Ny+1)."""
    i_inner = np.arange(0, Nx)
    j_inner = np.arange(0, Ny)
    i_ghost = np.array([-1, Nx])
    j_ghost = np.array([-1, Ny])

    ghost_points = []

    # Ghost columns: i = -1 and i = Nx+1, all j (inner + ghost)
    j_all = np.arange(-1, Ny + 1)
    for ig in i_ghost:
        for jg in j_all:
            ghost_points.append(ig * b1 / Nx + jg * b2 / Ny)

    # Ghost rows: j = -1 and j = Ny+1, inner i only (corners already added)
    for jg in j_ghost:
        for ig in i_inner:
            ghost_points.append(ig * b1 / Nx + jg * b2 / Ny)

    return np.array(ghost_points)


def map_to_first_BZ(k_points, b1, b2):
    B = np.column_stack([b1, b2])
    B_inv = np.linalg.inv(B)
    frac = k_points @ B_inv.T
    frac_reduced = frac - np.floor(frac + 0.5)
    k_reduced = frac_reduced @ B.T

    ns = np.array([[n1, n2] for n1 in [-1, 0, 1] for n2 in [-1, 0, 1]])
    G_candidates = ns @ B.T

    for i, k in enumerate(k_reduced):
        diffs = k - G_candidates
        norms = np.linalg.norm(diffs, axis=1)
        k_reduced[i] = diffs[np.argmin(norms)]

    return k_reduced


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


def plot(k_original, k_ghost, k_mapped, b1, b2):
    bz_vertices = get_BZ_vertices(b1, b2)
    bz_closed = np.vstack([bz_vertices, bz_vertices[0]])

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(bz_closed[:, 0], bz_closed[:, 1], 'k-', linewidth=1.5)
    ax.add_patch(Polygon(bz_vertices, closed=True, facecolor='lightblue', alpha=0.2))

    ax.scatter(k_original[:, 0], k_original[:, 1],
               s=8, color='royalblue', zorder=3, marker='h', label='k-mesh')
    ax.scatter(k_ghost[:, 0], k_ghost[:, 1],
               s=8, facecolors='none', edgecolors='k',
               zorder=3, marker='h', linewidths=0.6, label='ghost shell')
    ax.scatter(0, 0, s=8, color='red', zorder=4, label='Γ', marker='h')

    # b1 and b2 arrows
    for vec, label, color in zip([b1, b2], [r'$\mathbf{b}_1$', r'$\mathbf{b}_2$'], ['darkred', 'darkgreen']):
        ax.quiver(0, 0, vec[0], vec[1],
                angles='xy', scale_units='xy', scale=1,
                color=color, width=0.005, zorder=5)
        ax.text(vec[0] * 1.07, vec[1] * 1.07, label, fontsize=13, color=color)

    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title(r'Unit cell mesh $(15+1)\times(15+1)$')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.savefig('bz_mesh.png', dpi=150, bbox_inches='tight')

    plt.tight_layout()
    plt.show()



def main():
    Nx, Ny = 15, 15
    k_original = mesh_k_unit_cell(b1, b2, Nx=Nx, Ny=Ny)
    k_ghost = mesh_k_ghost_shell(b1, b2, Nx=Nx, Ny=Ny)
    k_mapped = map_to_first_BZ(k_original, b1, b2)
    plot(k_original, k_ghost, k_mapped, b1, b2)


if __name__ == "__main__":
    main()