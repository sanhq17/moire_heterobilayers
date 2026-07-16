import numpy as np
import matplotlib.pyplot as plt
from Moire import Moire
import utils


def integrate_wavefunction(psi, x, y):
    density = np.abs(psi) ** 2
    return np.trapz(np.trapz(density, x, axis=1), y, axis=0)

def main():
    sys = Moire()
    sys.theta_deg = 3.2
    sys.psi_deg = 26
    sys.V = 3.0
    sys.n_shells = 7

    scale = 1
    x = np.linspace(-scale * sys.aM, scale * sys.aM, 300)
    y = np.linspace(-scale * sys.aM, scale * sys.aM, 300)
    X, Y = np.meshgrid(x, y)
    psi_ho = utils.ho_2d_groundstate(X, Y, sys.aM, sys.V, sys.psi)
    psi_ho_perturbed = utils.ho_2d_groundstate_perturbed(X, Y, sys.aM, sys.V, sys.psi)

    norm_ho = integrate_wavefunction(psi_ho, x, y)
    norm_ho_perturbed = integrate_wavefunction(psi_ho_perturbed, x, y)
    print(f"Unperturbed wavefunction integral: {norm_ho}")
    print(f"Perturbed wavefunction integral: {norm_ho_perturbed}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        np.abs(psi_ho) ** 2,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        cmap='RdBu_r',
        aspect='auto',
        interpolation='none'
    )
    fig.colorbar(im, ax=ax, label=r'$|\psi_{0,0}(x, y)|^2$')
    ax.set_xlabel(r'$x\, (\AA)$')
    ax.set_ylabel(r'$y\, (\AA)$')
    ax.set_title('2D harmonic oscillator ground state')
    sys.plot_triangular_lattice(ax, extent=[x.min(), x.max(), y.min(), y.max()])
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(
        np.abs(psi_ho_perturbed) ** 2,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        cmap='RdBu_r',
        aspect='auto',
        interpolation='none'
    )
    fig.colorbar(im2, ax=ax2, label=r'$|\psi(x, y)|^2$')
    ax2.set_xlabel(r'$x\, (\AA)$')
    ax2.set_ylabel(r'$y\, (\AA)$')
    ax2.set_title('2D harmonic oscillator perturbed ground state')
    sys.plot_triangular_lattice(ax2, extent=[x.min(), x.max(), y.min(), y.max()])
    plt.savefig("outputs/ho_perturbed.png", dpi=300)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()