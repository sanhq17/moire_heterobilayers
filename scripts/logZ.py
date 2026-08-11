import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({
    'figure.figsize': (6, 5),
    'figure.dpi': 120,          # crisper on screen
    'savefig.dpi': 300,         # high-res when saving
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.spines.top': True,   # remove top/right borders
    'axes.spines.right': True,
    'axes.grid': False,
    'lines.linewidth': 2,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

# ---------- theta3 and Z (same conventions as before) ----------
def theta3(z, q, nmax=40):
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    for n in range(1, nmax + 1):
        out += 2.0 * q**(n * n) * np.cos(2.0 * n * z)
    return out

aM_over_sigma = 4
alpha_fac = 0.5                 # multiplies X = kx*aM
beta_fac  = np.sqrt(3) / 2      # multiplies Y = ky*aM
qx = np.exp(-0.5 * aM_over_sigma**2)
qy = np.exp(-1.5 * aM_over_sigma**2)
prefactor = np.sqrt(3) / (16 * np.pi) * aM_over_sigma**2

def Z(X, Y):
    zx, zy = alpha_fac * X, beta_fac * Y
    t1 = theta3(zx,           qx) * theta3(zy,           qy)
    t2 = theta3(zx - np.pi/2, qx) * theta3(zy + np.pi/2, qy)
    return prefactor * (t1 + t2)

# ---------- grid ----------
N = 500
xlim = ylim = np.pi
Xax = np.linspace(-xlim, xlim, N)
Yax = np.linspace(-ylim, ylim, N)
XX, YY = np.meshgrid(Xax, Yax)
ZZ = Z(XX, YY)

# ---------- moiré reciprocal lattice + BZ (in kx*aM, ky*aM units) ----------
g1 = np.array([np.pi,  np.pi/np.sqrt(3)])   # primitive reciprocal vectors
g2 = np.array([np.pi, -np.pi/np.sqrt(3)])   # nn-distance = 2*pi/sqrt(3)
Rv = 2*np.pi/3                              # center -> vertex (K) distance
ang = np.deg2rad(np.arange(0, 360, 60))     # vertices at 0,60,...,300 deg
hex0 = np.column_stack([Rv*np.cos(ang), Rv*np.sin(ang)])   # BZ around Gamma

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(7, 6))
pc = ax.pcolormesh(XX, YY, np.log(ZZ), shading="auto", cmap="viridis")
cb = fig.colorbar(pc, ax=ax); cb.set_label(r"$Z(\mathbf{k})$")

# tile the BZ over nearby reciprocal-lattice (Gamma) points
for m in range(-3, 4):
    for n in range(-3, 4):
        c = m*g1 + n*g2
        if abs(c[0]) <= xlim + Rv and abs(c[1]) <= ylim + Rv:
            ax.add_patch(Polygon(hex0 + c, closed=True, fill=False,
                                 edgecolor="white", lw=0.8, alpha=0.55))
            ax.plot(*c, ".", color="white", ms=3, alpha=0.7)

# highlight the central BZ and high-symmetry points
ax.add_patch(Polygon(hex0, closed=True, fill=False, edgecolor="red", lw=2.0))
K = hex0[0]                    # vertex  -> K
M = 0.5*(hex0[0] + hex0[1])   # edge midpoint (= g1/2) -> M
for pt, c, lab in [((0, 0), "red", r"$\Gamma$"), (K, "orange", "K"), (M, "cyan", "M")]:
    ax.plot(*pt, "o", color=c, ms=5)
    ax.annotate(lab, pt, color=c, fontsize=12,
                textcoords="offset points", xytext=(6, 4))

ax.set_xlim(-xlim, xlim); ax.set_ylim(-ylim, ylim)
ax.set_xlabel(r"$k_x a_M$", fontsize=20); ax.set_ylabel(r"$k_y a_M$", fontsize=20)
ax.set_title(r"$Z(\mathbf{k})$", fontsize=20)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()