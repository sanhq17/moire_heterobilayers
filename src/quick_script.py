import numpy as np
import matplotlib.pyplot as plt
from Moire import Moire
import utils

def main():
    sys = Moire()
    sys.theta_deg = 3.2
    sys.psi_deg = 15.0
    sys.V = 20

    kpath, kdist, ticks = sys.make_k_path(*sys.default_path, n_pts=400)
    bands, _ = sys.calculate_bands(kpath=kpath)
    print(f"Gap at {sys.kappa}: {sys.calculate_gap()}")

    fig, ax = plt.subplots()
    ax = sys.plot_bands(ax, bands, kdist, ticks, max_bands=4)
    plt.tight_layout()
    plt.text(0.85, 0.85, "$\\psi = " + str(np.round(np.degrees(sys.psi), 2)) + "^\\circ$ \n$\\theta = " + str(np.degrees(sys.theta)) + "^\\circ$\n" + str(sys.n_shells) + " shells", fontsize=12, ha='center', color='blue')

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    scale = 2.0
    extent = [-scale * sys.aM, scale * sys.aM, -scale * sys.aM, scale * sys.aM]
    im2 = sys.plot_moire_potential(ax2, extent, resolution=400)
    fig2.colorbar(im2, ax=ax2)
    sys.plot_triangular_lattice(ax2, extent)

    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_xlabel(r'$x (\AA)$')
    ax2.set_ylabel(r'$y (\AA)$')
    ax2.set_title(r'$\Delta(\mathbf{r})$', fontsize=16)
    plt.tight_layout()

    fig4, ax4 = plt.subplots(figsize=(6, 5))
    k_points = sys.mesh_k_unit_cell()
    x = np.linspace(-2 * sys.aM, 2 * sys.aM, 300)
    y = np.linspace(-2 * sys.aM, 2 * sys.aM, 300)
    X, Y = np.meshgrid(x, y)
    sys.plot_wannier_function(ax4, X, Y, k_points)
    sys.plot_triangular_lattice(ax4, extent)

    plt.xlabel(r'$x (\AA)$')
    plt.ylabel(r'$y (\AA)$')
    plt.legend()
    plt.tight_layout()

    fig5, ax5 = plt.subplots(figsize=(6, 5))
    scale = 1.5
    kx = np.linspace(-scale * sys.kappa_pp[0], scale * sys.kappa_pp[0], 100)
    ky = np.linspace(-scale * sys.kappa_p[1], scale * sys.kappa_p[1], 100)
    KX, KY = np.meshgrid(kx, ky)
    sys.plot_berry_curvature(ax5, KX, KY)

    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$k_y$')
    plt.tight_layout()

    fig6, ax6 = plt.subplots(2, 2, figsize=(6, 5))
    scale = 1.5
    kx = np.linspace(-scale * sys.kappa_pp[0], scale * sys.kappa_pp[0], 100)
    ky = np.linspace(-scale * sys.kappa_p[1], scale * sys.kappa_p[1], 100)
    KX, KY = np.meshgrid(kx, ky)
    sys.plot_metric(ax6, KX, KY)

    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$k_y$')
    plt.tight_layout()

    fig7, ax7 = plt.subplots(1, 2, figsize=(12, 5))
    scale = 1.5
    kx = np.linspace(-scale * sys.kappa_pp[0], scale * sys.kappa_pp[0], 100)
    ky = np.linspace(-scale * sys.kappa_p[1], scale * sys.kappa_p[1], 100)
    KX, KY = np.meshgrid(kx, ky)
    sys.plot_trace_metric(ax7, KX, KY)

    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$k_y$')
    plt.tight_layout()

    print(np.sqrt(sys.integrate_trace(KX, KY) / sys.BZ_area))

    plt.show()

if __name__ == "__main__":
    main()