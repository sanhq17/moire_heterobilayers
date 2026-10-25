import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.special import eval_hermite, factorial

import io
from PIL import Image

from mpl_toolkits.axisartist.floating_axes import FloatingSubplot, GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import FixedLocator
from matplotlib.transforms import Affine2D


@njit(cache=True)
def _qho_basis_1d(coordinates, n_max, sigma):
    """Build normalized QHO basis functions using a stable recurrence."""
    basis = np.empty((n_max + 1, coordinates.size), dtype=np.float64)
    xi = coordinates / sigma
    gaussian = np.exp(-0.5 * xi**2)
    basis[0] = gaussian / (np.pi**0.25 * np.sqrt(sigma))

    if n_max >= 1:
        basis[1] = np.sqrt(2.0) * xi * basis[0]

    for n in range(2, n_max + 1):
        basis[n] = (
            np.sqrt(2.0 / n) * xi * basis[n - 1]
            - np.sqrt((n - 1.0) / n) * basis[n - 2]
        )

    return basis


@njit(cache=True, parallel=True)
def _project_qho_2d(phi_x, phi_y, W, dx, dy):
    """Evaluate all 2D QHO overlaps without allocating tensor-product states."""
    n_states = phi_x.shape[0]
    n_y, n_x = W.shape
    coeffs = np.zeros((n_states, n_states), dtype=np.complex128)

    for nx in prange(n_states):
        for ny in range(n_states):
            overlap = 0.0j
            for iy in range(n_y):
                phi_y_conjugate = np.conjugate(phi_y[ny, iy])
                for ix in range(n_x):
                    overlap += (
                        phi_y_conjugate
                        * np.conjugate(phi_x[nx, ix])
                        * W[iy, ix]
                    )
            coeffs[nx, ny] = overlap * dx * dy

    return coeffs


def qho_1d(x, n, sigma):
    """
    1D quantum harmonic oscillator eigenstate.

    Parameters
    ----------
    x : array_like
        Position coordinates.
    n : int
        Harmonic oscillator quantum number.
    sigma : float
        Oscillator length.

    Returns
    -------
    phi : ndarray
        Normalized QHO eigenstate.
    """
    xi = x / sigma

    normalization = (
        1.0
        / np.sqrt(
            2**n * factorial(n) * np.sqrt(np.pi) * sigma
        )
    )

    return normalization * np.exp(-xi**2 / 2) * eval_hermite(n, xi)


def project_qho_2d(X, Y, W, n_max, sigma_x, sigma_y=None):
    """
    Project a 2D wavefunction W(X,Y) onto 2D QHO eigenstates.

    Parameters
    ----------
    X, Y : ndarray
        2D meshgrid arrays.
    W : ndarray
        Complex-valued 2D wavefunction on the mesh.
    n_max : int
        Maximum QHO quantum number in each direction.
    sigma_x : float
        QHO oscillator length in x.
    sigma_y : float, optional
        QHO oscillator length in y. If None, sigma_y = sigma_x.

    Returns
    -------
    coeffs : ndarray
        Complex coefficient matrix with
        coeffs[nx, ny] = <nx,ny|W>.
    """
    if sigma_y is None:
        sigma_y = sigma_x

    # Grid spacings
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    # 1D basis functions; the recurrence avoids calling SciPy once per state.
    phi_x = _qho_basis_1d(np.ascontiguousarray(X[0, :]), n_max, sigma_x)
    phi_y = _qho_basis_1d(np.ascontiguousarray(Y[:, 0]), n_max, sigma_y)

    return _project_qho_2d(
        phi_x,
        phi_y,
        np.ascontiguousarray(W),
        dx,
        dy,
    )

def plot_qho_coefficients(coeffs, angle=45, figsize=(6, 6)):
    """
    Plot 2D QHO coefficients as points at (nx, ny) inside a diamond
    (frame rotated 45deg CCW), with intensity proportional to |c_nx,ny|.
    Tick labels, axis labels, and colorbar stay upright.
    """
    nx, ny = np.indices(coeffs.shape)
    magnitude = np.abs(coeffs)  / np.abs(coeffs[0, 0])
    n0, n1 = coeffs.shape

    fig = plt.figure(figsize=figsize)

    tr = Affine2D().rotate_deg(angle)  # the rotation applied to the frame

    extremes = (-0.5, n0 - 0.5, -0.5, n1 - 0.5)
    grid_helper = GridHelperCurveLinear(
        tr,
        extremes=extremes,
        grid_locator1=FixedLocator(np.arange(n0)),
        grid_locator2=FixedLocator(np.arange(n1)),
    )

    ax = FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax)
    aux_ax = ax.get_aux_axes(tr)  # plot into this using UNROTATED data coords

    scatter = aux_ax.scatter(
        nx.ravel(), ny.ravel(),
        c=magnitude.ravel(), s=120, cmap="grey_r",
        edgecolors=None, linewidths=0.5,
        vmin=0, vmax=1
    )

    # Force every tick label to stay horizontal (by default axisartist
    # rotates labels to follow the axis direction, which is what tilts them)
    for key in ["left", "right", "top", "bottom"]:
        axis = ax.axis[key]
        axis.major_ticks.set_tick_out(True)   # <-- ticks now point outward
        tl = axis.major_ticklabels
        tl._text_follow_ref_angle = False
        tl.set_rotation(0)
        tl.set_ha("center")
        tl.set_va("center")
        tl.set_pad(8)

    # Axis labels, same trick to keep them unrotated
    ax.axis["bottom"].label.set_text(r"$n_x$")
    ax.axis["left"].label.set_text(r"$n_y$")
    for key in ["bottom", "left"]:
        lbl = ax.axis[key].label
        lbl._text_follow_ref_angle = False
        lbl.set_rotation(0)
    ax.axis["top"].label.set_text("")
    ax.axis["right"].label.set_text("")

    # Colorbar is a normal separate Axes -> unaffected by the rotation
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.05)
    cbar.set_label(r"$|\lambda_{n_x,n_y}|$")

    plt.tight_layout()
    plt.show()


# Mesh
x = np.linspace(-5, 5, 201)
y = np.linspace(-5, 5, 201)

X, Y = np.meshgrid(x, y)

# Example wavefunction
sigma = 1.0
W = (
    np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    / (np.sqrt(np.pi) * sigma)
)

# Project onto states nx,ny <= 5
coeffs = project_qho_2d(
    X, Y, W,
    n_max=8,
    sigma_x=sigma/1.5
)


np.set_printoptions(precision=4, suppress=True)
print(coeffs / coeffs[0, 0])  # normalized to the ground state

plot_qho_coefficients(coeffs)