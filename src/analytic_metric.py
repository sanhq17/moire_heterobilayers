import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import utils
# ---------- theta + metric (from before) ----------
def _theta3(z, q, eps=1e-18):
    z = np.asarray(z, float)
    nmax = 2 if q <= 0 else int(np.ceil(np.sqrt(abs(np.log(eps)/np.log(q))))) + 2
    n = np.arange(1, nmax + 1, dtype=float)
    qn = q ** (n*n)
    ang = 2.0 * n * z[..., None]
    c, s = np.cos(ang), np.sin(ang)
    t0 = 1.0 + 2.0*(qn*c).sum(-1)
    t1 = -4.0*(qn*n*s).sum(-1)
    t2 = -8.0*(qn*n*n*c).sum(-1)
    return t0, t1, t2

def moire_metric(kx, ky, sigma, aM):
    kx = np.asarray(kx, float); ky = np.asarray(ky, float)
    alpha = np.sqrt(3.0)*aM/4.0; beta = aM/4.0
    qx = np.exp(-3.0*aM**2/(16.0*sigma**2)); qy = np.exp(-aM**2/(16.0*sigma**2))
    X0, x1, x2    = _theta3(alpha*kx,            qx)
    Xt0, xt1, xt2 = _theta3(alpha*kx - np.pi/2,  qx)
    X1, X2, Xt1, Xt2 = alpha*x1, alpha**2*x2, alpha*xt1, alpha**2*xt2
    Y0, y1, y2    = _theta3(beta*ky,             qy)
    Yt0, yt1, yt2 = _theta3(beta*ky + np.pi/2,   qy)
    Y1, Y2, Yt1, Yt2 = beta*y1, beta**2*y2, beta*yt1, beta**2*yt2
    S   = X0*Y0 + Xt0*Yt0
    Sx  = X1*Y0 + Xt1*Yt0;  Sy  = X0*Y1 + Xt0*Yt1
    Sxx = X2*Y0 + Xt2*Yt0;  Syy = X0*Y2 + Xt0*Yt2
    Sxy = X1*Y1 + Xt1*Yt1
    Lx, Ly = Sx/S, Sy/S
    gxx = 0.5*sigma**2 + 0.25*(Sxx/S - Lx*Lx)
    gyy = 0.5*sigma**2 + 0.25*(Syy/S - Ly*Ly)
    gxy = 0.25*(Sxy/S - Lx*Ly)
    return gxx, gyy, gxy

# ---------- lattice / BZ geometry ----------
def reciprocal_vectors(aM):
    a1 = np.array([np.sqrt(3)*aM/2, aM/2])
    a2 = np.array([0.0, aM])
    B = 2*np.pi*np.linalg.inv(np.column_stack([a1, a2])).T
    return B[:, 0], B[:, 1]          # b1, b2  (a_i . b_j = 2 pi delta_ij)

def bz_hexagon(b1, b2):
    """Wigner-Seitz cell corners (K points) from the 6 shortest G vectors."""
    Gs = np.array([b1, b2, b1+b2, -b1, -b2, -(b1+b2)])
    Gs = Gs[np.argsort(np.arctan2(Gs[:, 1], Gs[:, 0]))]
    corners = []
    for i in range(6):                # corner = intersection of two perp. bisectors
        G1, G2 = Gs[i], Gs[(i+1) % 6]
        corners.append(np.linalg.solve([G1, G2], 0.5*np.array([G1@G1, G2@G2])))
    return np.array(corners)

# ---------- plot ----------
def plot_metric_bz(KX, KY, sigma, aM, V, psi, cmap="RdBu_r", normalize=False):
    if normalize:
        ktheta = 4*np.pi/(np.sqrt(3)*aM)
    else:
        ktheta = 1.0

    b1, b2 = reciprocal_vectors(aM) / ktheta
    hexc = bz_hexagon(b1, b2)
    path = Path(np.vstack([hexc, hexc[0]]))

    # R = np.abs(hexc).max()*1.02
    # ax_k = np.linspace(-R, R, n)
    # KX, KY = np.meshgrid(ax_k, ax_k)
    gxx, gyy, gxy = moire_metric(KX, KY, sigma, aM)
    gxx = gxx / utils.spread_harmonic_approx(aM, V, psi)**2
    gyy = gyy / utils.spread_harmonic_approx(aM, V, psi)**2
    gxy = gxy / utils.spread_harmonic_approx(aM, V, psi)**2

    

    inside = path.contains_points(
        np.column_stack([KX.ravel(), KY.ravel()])).reshape(KX.shape)
    fields = {r"$g_{xx} / a_W^2$": gxx, r"$g_{xy} / a_W^2$": gxy,
              r"$g_{yx} / a_W^2$": gxy, r"$g_{yy} / a_W^2$": gyy}

    # high-symmetry points
    K = hexc
    M = 0.5*np.array([b1, b2, b1+b2, -b1, -b2, -(b1+b2)])

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    for ax, (name, F) in zip(axes.flat, fields.items()):
        Fm = np.where(inside, F, np.nan)
        Fm = Fm / ktheta**2
        pc = ax.pcolormesh(KX / ktheta, KY / ktheta, F, shading="auto", cmap=cmap, rasterized=True)
        fig.colorbar(pc, ax=ax, shrink=0.85, fraction=0.046, pad=0.04)
        ax.plot(*np.vstack([hexc, hexc[0]]).T, "k-", lw=2.5)
        # ax.scatter(0, 0, c="w", s=25, zorder=5)
        # ax.scatter(*K.T, c="w", s=18, marker="o", zorder=5)
        # ax.scatter(*M.T, c="w", s=18, marker="s", zorder=5)
        ax.set_title(name, fontsize=30)
        ax.set_xlabel(r"$k_x / k_\theta$",fontsize=28)
        ax.set_ylabel(r"$k_y / k_\theta$",fontsize=28)
    # return fig

def plot_trace_bz(KX, KY, sigma, aM, V, psi, cmap="Reds"):
    b1, b2 = reciprocal_vectors(aM)
    hexc = bz_hexagon(b1, b2)
    path = Path(np.vstack([hexc, hexc[0]]))

    gxx, gyy, gxy = moire_metric(KX, KY, sigma, aM)
    gxx = gxx / utils.spread_harmonic_approx(aM, V, psi)**2
    gyy = gyy / utils.spread_harmonic_approx(aM, V, psi)**2
    gxy = gxy / utils.spread_harmonic_approx(aM, V, psi)**2
    trace_g = gxx + gyy

    inside = path.contains_points(
        np.column_stack([KX.ravel(), KY.ravel()])).reshape(KX.shape)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    trace_gm = np.where(inside, trace_g, np.nan)
    pc = ax.pcolormesh(KX, KY, trace_gm, shading="auto", cmap=cmap)
    fig.colorbar(pc, ax=ax)
    ax.plot(*np.vstack([hexc, hexc[0]]).T, "k-", lw=1.2)
    ax.set(title=r"$\mathrm{Tr}(g) / a_W^2$", aspect="equal",
           xlabel=r"$k_x$", ylabel=r"$k_y$")

# if __name__ == "__main__":
#     fig = plot_metric_bz(a=120.0, aM=300)
#     plt.show()