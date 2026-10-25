"""
Z0(k) and its derivatives, built from Jacobi theta-3 functions.

    Z0(kx,ky) = (sqrt(3)/16pi) (aM/sigma)^2 *
                [ theta3(alpha*kx, qx)     * theta3(beta*ky, qy)
                + theta3(alpha*kx - pi/2, qx) * theta3(beta*ky + pi/2, qy) ]

    alpha = sqrt(3)*aM/4,  beta = aM/4
    qx = exp(-3 aM^2 / (16 sigma^2)),  qy = exp(-aM^2 / (16 sigma^2))

Because Z0 is a sum of two terms, each a product of a pure function of kx
and a pure function of ky, every mixed derivative d^nx/dkx^nx d^ny/dky^ny Z0
factorizes the same way. The only building block needed is the n-th
z-derivative of theta3(z,q) = sum_m q^(m^2) exp(2i*m*z):

    d^n/dz^n theta3(z,q) = sum_m q^(m^2) (2i*m)^n exp(2i*m*z)

which is evaluated directly as a (numerically real) truncated sum below.
"""

import numpy as np


def _theta3_zderiv(z, q, n, tol=1e-15):
    """
    n-th derivative w.r.t. z of theta3(z,q) = sum_{m=-inf}^{inf} q^(m^2) exp(2i*m*z).

    z : float or ndarray, real argument
    q : float, nome (0 < q < 1)
    n : int >= 0, derivative order (n=0 gives theta3 itself)
    tol : truncation tolerance on q^(m^2)

    Returns a real ndarray/float (imaginary part is numerically ~0 by
    construction and is discarded).
    """
    if not (0.0 < q < 1.0):
        raise ValueError("nome q must satisfy 0 < q < 1")

    # smallest M such that q^(M^2) < tol
    M = int(np.ceil(np.sqrt(np.log(tol) / np.log(q)))) + 1
    m = np.arange(-M, M + 1)

    z = np.asarray(z, dtype=float)
    scalar_input = (z.ndim == 0)
    z = np.atleast_1d(z)

    # shape (2M+1, len(z))
    phase = np.exp(2j * np.outer(m, z))
    coeff = (q ** (m.astype(float) ** 2)) * (2j * m) ** n
    result = (coeff[:, None] * phase).sum(axis=0)

    result = result.real  # imaginary part cancels by m -> -m symmetry
    return result.item() if scalar_input else result


def Z0(kx, ky, sigma, aM):
    """Z0(kx, ky) for given sigma (Gaussian width) and aM (lattice constant)."""
    alpha = np.sqrt(3) * aM / 4
    beta = aM / 4
    qx = np.exp(-3 * aM**2 / (16 * sigma**2))
    qy = np.exp(-aM**2 / (16 * sigma**2))

    prefac = np.sqrt(3) / (16 * np.pi) * (aM / sigma) ** 2

    t1 = _theta3_zderiv(alpha * kx, qx, 0) * _theta3_zderiv(beta * ky, qy, 0)
    t2 = _theta3_zderiv(alpha * kx - np.pi / 2, qx, 0) * _theta3_zderiv(beta * ky + np.pi / 2, qy, 0)

    return prefac * (t1 + t2)


def dZ0(kx, ky, nx, ny, sigma, aM):
    """
    (nx, ny)-th mixed partial derivative d^nx/dkx^nx d^ny/dky^ny Z0(kx,ky).

    nx, ny : nonnegative integers (order of derivative in kx, ky respectively)
    """
    alpha = np.sqrt(3) * aM / 4
    beta = aM / 4
    qx = np.exp(-3 * aM**2 / (16 * sigma**2))
    qy = np.exp(-aM**2 / (16 * sigma**2))

    prefac = np.sqrt(3) / (16 * np.pi) * (aM / sigma) ** 2

    # chain rule: d/dkx = alpha * d/dz on the first factor, etc.
    t1 = (alpha ** nx) * (beta ** ny) * \
         _theta3_zderiv(alpha * kx, qx, nx) * _theta3_zderiv(beta * ky, qy, ny)
    t2 = (alpha ** nx) * (beta ** ny) * \
         _theta3_zderiv(alpha * kx - np.pi / 2, qx, nx) * _theta3_zderiv(beta * ky + np.pi / 2, qy, ny)

    return prefac * (t1 + t2)


import numpy as np
import matplotlib.pyplot as plt

# ---------- parameters ----------
aM = 1.0       # moire lattice constant
sigma = 0.35   # Wannier Gaussian width

def quantum_metric_zeroth(kx, ky, sigma, aM):
    """g_ij^(0)(k) = sigma^2/2 * delta_ij + 1/4 d_i d_j log Z0,
    evaluated on flattened kx, ky arrays of equal shape."""
    shape = kx.shape
    kxf, kyf = kx.ravel(), ky.ravel()

    Z   = Z0(kxf, kyf, sigma, aM)
    Zx  = dZ0(kxf, kyf, 1, 0, sigma, aM)
    Zy  = dZ0(kxf, kyf, 0, 1, sigma, aM)
    Zxx = dZ0(kxf, kyf, 2, 0, sigma, aM)
    Zyy = dZ0(kxf, kyf, 0, 2, sigma, aM)
    Zxy = dZ0(kxf, kyf, 1, 1, sigma, aM)

    dlogZ_xx = Zxx / Z - (Zx / Z) ** 2
    dlogZ_yy = Zyy / Z - (Zy / Z) ** 2
    dlogZ_xy = Zxy / Z - (Zx * Zy) / Z ** 2

    gxx = sigma**2 / 2 + dlogZ_xx / 4
    gyy = sigma**2 / 2 + dlogZ_yy / 4
    gxy = dlogZ_xy / 4

    return gxx.reshape(shape), gyy.reshape(shape), gxy.reshape(shape)

# ---------- grid over a few moire BZs ----------
Kwin = 2 * (2 * np.pi / aM)
N = 220
kx = np.linspace(-Kwin, Kwin, N)
ky = np.linspace(-Kwin, Kwin, N)
KX, KY = np.meshgrid(kx, ky, indexing='xy')

gxx, gyy, gxy = quantum_metric_zeroth(KX, KY, sigma, aM)

# sanity checks: Q^(0) should be PSD -> gxx,gyy >=0 and det(g) = gxx*gyy - gxy^2 >= 0
det_g = gxx * gyy - gxy**2
print("min gxx:", gxx.min(), " min gyy:", gyy.min(), " min det(g):", det_g.min())

# ---------- plot ----------
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
data = [gxx, gyy, gxy, np.zeros_like(gxx)]
titles = [r"$g_{xx}^{(0)}(\mathbf{k})$", r"$g_{yy}^{(0)}(\mathbf{k})$",
          r"$g_{xy}^{(0)}(\mathbf{k})$", r"$\Omega_{xy}^{(0)}(\mathbf{k}) \equiv 0$"]

for ax, d, t in zip(axes, data, titles):
    cmap = 'viridis' if 'Omega' not in t else 'RdBu_r'
    vmax = np.max(np.abs(d)) if np.max(np.abs(d)) > 0 else 1
    vmin = 0 if 'Omega' not in t and 'xy' not in t else -vmax
    im = ax.pcolormesh(KX * aM / (2*np.pi), KY * aM / (2*np.pi), d, cmap=cmap, shading='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(t)
    ax.set_xlabel(r"$k_x a_M/2\pi$")
    ax.set_ylabel(r"$k_y a_M/2\pi$")
    ax.set_aspect('equal')

fig.suptitle(rf"Zeroth-order (Gaussian) QGT:  $\sigma/a_M={sigma}$", y=1.03)
plt.tight_layout()
plt.show()