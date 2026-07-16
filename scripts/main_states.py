import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0
psi  = np.radians(20)
V    = 10 # meV
a0  = 3.32 # lattice constant in Angstroms
theta = np.radians(0.5)
aM = a0 / (2 * np.sin(theta / 2))  # moiré lattice constant
mstar = 0.35
h2m = 7618  # in units of meV*Angstrom^2
n_shells = 10

phi = np.radians(120) + psi
beta = 16 * np.pi**2 * np.cos(phi)
aW = np.sqrt(aM) * (h2m / (mstar * V * beta))**0.25

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"  # Optional for advanced math
})

### Triangular lattice and basis
def triangular_lattice(a):
    b1 = np.array([4 * np.pi / (np.sqrt(3) * a), 0.0])
    phi = np.radians(120)
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    b2 = R @ b1
    return b1, b2

b1, b2 = triangular_lattice(aM)

def generate_basis(b1, b2, n_shells):
    N = n_shells + 2
    candidates = []
    for n1 in range(-N, N + 1):
        for n2 in range(-N, N + 1):
            vec = n1 * b1 + n2 * b2
            mag2 = round(float(vec @ vec), 6)  # consistent precision
            candidates.append((mag2, vec, (n1, n2)))

    # find the n_shells smallest NONZERO distinct magnitudes
    nonzero_mag2 = sorted(set(m for m, _, _ in candidates if m > 1e-10))
    thresholds = nonzero_mag2[:n_shells]

    basis = [(np.zeros(2), (0, 0))]  # origin / shell 0
    for thresh in thresholds:
        for m, v, n in candidates:
            if abs(m - thresh) < 1e-5:
                basis.append((v, n))

    return basis

basis = generate_basis(b1, b2, n_shells)

### 

def V_fourier(dm, dn):
    if (dm, dn) in ((1, 0), (0, 1), (-1, -1)):
        return V * np.exp( 1j * psi)
    if (dm, dn) in ((-1, 0), (0, -1), (1, 1)):
        return V * np.exp(-1j * psi)
    return 0.0

def H_k(k):
    b1, b2 = triangular_lattice(aM)
    basis = generate_basis(b1, b2, n_shells)  # flat list of (g, (n1,n2))
    N = len(basis)
    H = np.zeros((N, N), dtype=complex)

    for i, (g, n) in enumerate(basis):
        k_plus_g = k + g
        H[i, i] = -h2m * np.linalg.norm(k_plus_g)**2 / (2 * mstar)

        for j in range(i + 1, N):
            g2, n2 = basis[j]
            dm, dn = n2[0] - n[0], n2[1] - n[1]
            potential = V_fourier(dm, dn)
            if abs(potential) > 1e-10:
                H[i, j] = potential
                H[j, i] = np.conj(potential)

    return H

def make_path(n_pts):
    gamma  = np.zeros(2)
    kappa_pp = ( 2 * b1 + b2 ) / 3            # κ''
    
    phi = np.radians(60)
    c, s = np.cos(phi), np.sin(phi)
    R1 = np.array([[c, -s], [s, c]])
    kappa_p = R1 @ kappa_pp

    phi = np.radians(180)
    c, s = np.cos(phi), np.sin(phi)
    R2 = np.array([[c, -s], [s, c]])
    kappa = R2 @ kappa_p

    segments = [
        (gamma,    kappa_p,  "γ",   "κ'"),
        (kappa_p,  kappa_pp, "κ'",  "κ''"),
        (kappa_pp, kappa,    "κ''", "κ"),
    ]

    # Distribute n_pts across segments proportional to their length
    lengths = np.array([np.linalg.norm(end - start) for start, end, *_ in segments])
    total_length = lengths.sum()
    # Number of points per segment (ensure they sum to n_pts)
    pts_per_seg = np.round(lengths / total_length * (n_pts - 1)).astype(int)
    pts_per_seg[-1] = (n_pts - 1) - pts_per_seg[:-1].sum()  # fix rounding

    kpath = []
    kdist = []
    ticks = []
    cumulative = 0.0

    for i, (start, end, label_start, label_end) in enumerate(segments):
        n = pts_per_seg[i] + 1  # +1 to include endpoint, trimmed below
        pts = np.linspace(0, 1, n)
        seg_pts = start[None, :] + pts[:, None] * (end - start)[None, :]
        seg_len = np.linalg.norm(end - start)
        seg_dist = cumulative + pts * seg_len

        if i == 0:
            ticks.append((cumulative, label_start))
        # Drop last point (it's the start of the next segment)
        kpath.append(seg_pts[:-1])
        kdist.append(seg_dist[:-1])

        cumulative += seg_len
        ticks.append((cumulative, label_end))

    # Append the final endpoint
    kpath.append(end[None, :])
    kdist.append(np.array([cumulative]))

    kpath = np.concatenate(kpath, axis=0)
    kdist = np.concatenate(kdist, axis=0)

    return kpath, kdist, ticks

def compute_bands(n_pts):
    kpath, kdist, ticks = make_path(n_pts)

    # Diagonalize at first k-point to get system size
    e0 = np.linalg.eigvalsh(H_k(kpath[0]))
    N = len(e0)

    bands = np.empty((n_pts, N))
    states = np.empty((n_pts, N, N), dtype=complex)
    for i, k in enumerate(kpath):
        eigenvalues, eigenvectors = np.linalg.eigh(H_k(k))
        idx = np.argsort(eigenvalues)[::-1]
        bands[i, :] = eigenvalues[idx]
        states[i, :, :] = eigenvectors[:, idx]  # take top N and reverse to descending order

    return kdist, kpath, bands, states, ticks

def bloch_wavefunction(k, r, basis, z):
    wavefunction = 0.0
    for i, (g, _) in enumerate(basis):
        k_plus_g = k + g
        phase = np.exp(1.0j * k_plus_g @ r)
        wavefunction += z[i] * phase
        
    return wavefunction


def sample_k_points(k_mesh):
    """Return the k-points used in the Brillouin-zone mesh sum."""
    n1 = np.arange(k_mesh)
    n2 = np.arange(k_mesh)
    k_points = []
    for i in n1:
        for j in n2:
            k_points.append(i * b1 / k_mesh + j * b2 / k_mesh)
    return np.array(k_points)

def wannier_wavefunction(x, y, k_mesh = 10, gauge_fixed=True):
    N1 = k_mesh # number of lattice points in each direction to sum over
    N2 = N1
    k_points = sample_k_points(k_mesh)

    phi = np.radians(0)
    c, s = np.cos(phi), np.sin(phi)
    rot = np.array([[c, -s], [s, c]])
    R0 = rot @ np.array([0, 0])  # shift to center Wannier function

    wannier = 0.0
    print(len(k_points))
    i = 0
    for k in k_points:
        print(i)
        i += 1
        eigenvalues, eigenvectors = np.linalg.eigh(H_k(k))
        idx = np.argsort(eigenvalues)[::-1]
        states = eigenvectors[:, idx]

        zk = states[:, 0]  # top band state
        bloch0 = 0.0

        for m, (g, _) in enumerate(basis):
            k_plus_g = k + g
            bloch0 += zk[m] * np.exp(1.0j * (k_plus_g[0] * x + k_plus_g[1] * y))
        shift = np.exp(-1.0j * k @ R0)  # optional phase factor to center Wannier function
        
        if gauge_fixed:
            overlap = np.sum(np.conj(zk))
            wannier += bloch0 * shift * np.exp(1.0j * np.angle(overlap))
        else:
            wannier += bloch0 * shift

    Omega = (np.sqrt(3) / 2) * aM**2
    wannier = wannier / (len(k_points) * np.sqrt(Omega))

    return wannier

def integrate_wannier_wavefunction(wannier, x, y):
    density = np.abs(wannier) ** 2
    return np.trapz(np.trapz(density, x, axis=1), y, axis=0)

def spread_wannier_wavefunction(wannier, X, Y, x, y):
    density = np.abs(wannier) ** 2
    R2 = X**2 + Y**2
    return np.trapz(np.trapz(density * R2, x, axis=1), y, axis=0)

def bloch_u(r, k, gauge_fixed=True):
    eigenvalues, eigenvectors = np.linalg.eigh(H_k(k))
    idx = np.argsort(eigenvalues)[::-1]
    states = eigenvectors[:, idx]

    zk = states[:, 0]  # top band state
    bloch0 = 0.0

    for m, (g, _) in enumerate(basis):
        bloch0 += zk[m] * np.exp(1.0j * g @ r)

    if gauge_fixed:
        overlap = np.sum(np.conj(zk))
        bloch_u = bloch0 * np.exp(1.0j * np.angle(overlap))
    else:
        bloch_u = bloch0

    return bloch_u

def fourier_gaussian(k):
    return np.sqrt(aW**2 / np.pi) * np.exp(- aW**2 * np.linalg.norm(k)**2 / 2)

def plot_wavefunction(ax, wavefunctions, extent):
    im = ax.imshow(np.abs(wavefunctions)**2, extent=extent, origin='lower', 
                   interpolation='none', cmap='RdBu_r')
    return im

def plot_triangular_lattice(ax, aM, extent):
    # Overlay triangular lattice as dashed lines (real-space primitive vectors)
    r1 = np.array([0, 1.0]) * aM
    r2 = np.array([ np.sqrt(3) / 2.0, 1.0 / 2.0]) * aM
    r3 = r2 - r1

    x_min, x_max, y_min, y_max = extent

    # Build a grid of lattice sites large enough to cover the plotting extent.
    n_max = int(np.ceil(max(x_max - x_min, y_max - y_min) / aM)) + 4
    lattice_pts = []
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            lattice_pts.append(n1 * r1 + n2 * r2)
    lattice_pts = np.array(lattice_pts)

    # Draw nearest-neighbor bonds as dashed lines.
    offsets = [r1, r2, r3]
    for p in lattice_pts:
        for off in offsets:
            q = p + off
            x0, y0 = p
            x1, y1 = q
            if (min(x0, x1) <= x_max and max(x0, x1) >= x_min and
                min(y0, y1) <= y_max and max(y0, y1) >= y_min):
                ax.scatter([x0, x1], [y0, y1], color='r', s=20, alpha=0.9)
                ax.plot([x0, x1], [y0, y1], color='k', ls='--', lw=0.8, alpha=0.9)

def moire_potential(r):
    phi = np.radians(120)
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    b3 = R @ b2

    return 2 * V * (np.cos(b1 @ r + psi) + np.cos(b2 @ r + psi) + np.cos(b3 @ r + psi))


def plot_moire_potential(ax, extent, resolution=200):
    """Compute and plot the moiré potential over the given extent.

    Parameters
    - ax: matplotlib Axes
    - extent: (x_min, x_max, y_min, y_max)
    - resolution: number of points per axis
    """
    x = np.linspace(extent[0], extent[1], resolution)
    y = np.linspace(extent[2], extent[3], resolution)
    X, Y = np.meshgrid(x, y)

    pot = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r = np.array([X[i, j], Y[i, j]])
            pot[i, j] = moire_potential(r)

    im = ax.imshow(pot, extent=extent, origin='lower', cmap='RdBu_r')
    return im


def plot_k_mesh(ax, k_mesh=7):
    """Plot the reciprocal-space mesh used in the k-point sum."""
    k_points = sample_k_points(k_mesh)
    vertices = np.array([
        np.zeros(2),
        b1,
        b1 + b2,
        b2,
        np.zeros(2),
    ])

    ax.scatter(k_points[:, 0], k_points[:, 1], s=18, color='tab:blue', alpha=0.85)
    ax.plot(vertices[:, 0], vertices[:, 1], color='k', lw=1.5, ls='--')
    ax.quiver(
        0.0, 0.0, b1[0], b1[1],
        angles='xy', scale_units='xy', scale=1,
        color='k', width=0.006
    )
    ax.quiver(
        0.0, 0.0, b2[0], b2[1],
        angles='xy', scale_units='xy', scale=1,
        color='k', width=0.006
    )
    ax.text(b1[0] * 0.5, -0.01, r'$\mathbf{b}_1$', fontsize=12, ha='center', va='center')
    ax.text(b2[0], b2[1] * 0.5, r'$\mathbf{b}_2$', fontsize=12, ha='center', va='center')

    padding = 0.05
    x_min = min(0.0, b1[0], b2[0], (b1 + b2)[0]) - padding
    x_max = max(0.0, b1[0], b2[0], (b1 + b2)[0]) + padding
    y_min = min(0.0, b1[1], b2[1], (b1 + b2)[1]) - padding
    y_max = max(0.0, b1[1], b2[1], (b1 + b2)[1]) + padding

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$k_x$', fontsize=16)
    ax.set_ylabel(r'$k_y$', fontsize=16)

def main():
    # --- compute bands ---
    scale = 0.4
    k_mesh = 8
    x = np.linspace(-scale * aM, scale * aM, 400)
    y = np.linspace(-scale * aM, scale * aM, 400)
    X, Y = np.meshgrid(x, y)

    wannier = np.zeros(X.shape, dtype=complex)
    blochU = np.zeros(X.shape, dtype=complex)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         r = np.array([X[i, j], Y[i, j]])
    wannier = wannier_wavefunction(X, Y, k_mesh=k_mesh, gauge_fixed=True)
            # blochU[i, j] = bloch_u(r, np.array([0, 0]), gauge_fixed=True)

    wannier_integral = integrate_wannier_wavefunction(wannier, x, y)
    print(f"Wannier wavefunction integral: {wannier_integral}")

    spread = spread_wannier_wavefunction(wannier, X, Y, x, y)
    print(f"Spread of Wannier wavefunction: {np.sqrt(spread)}")

    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [-scale * aM, scale * aM, -scale * aM, scale * aM]
    im = plot_wavefunction(ax, wannier, extent)
    fig.colorbar(im, ax=ax)
    plot_triangular_lattice(ax, aM, extent)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r'$x (Å)$')
    ax.set_ylabel(r'$y (Å)$')
    ax.set_title(r"$w_{0}(\boldmath{r})$", fontsize=16)
    plt.tight_layout()
    plt.savefig("outputs/wannier_function.png", dpi=300)

    # --- moiré potential on same spatial region ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = plot_moire_potential(ax2, extent, resolution=300)
    fig2.colorbar(im2, ax=ax2)
    plot_triangular_lattice(ax2, aM, extent)

    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_xlabel(r'$x (Å)$')
    ax2.set_ylabel(r'$y (Å)$')
    ax2.set_title(r'$\Delta(\mathbf{r})$', fontsize=16)
    plt.tight_layout()
    plt.savefig("outputs/moire_potential.png", dpi=300)
    plt.show()

    # --- moiré potential on same spatial region ---
    # fig3, ax3 = plt.subplots(figsize=(6, 5))
    # im3 = plot_wavefunction(ax3, blochU, extent)
    # fig3.colorbar(im3, ax=ax3)
    # plot_triangular_lattice(ax3, aM, extent)

    # ax3.set_xlim(extent[0], extent[1])
    # ax3.set_ylim(extent[2], extent[3])
    # ax3.set_xlabel(r'$x (Å)$')
    # ax3.set_ylabel(r'$y (Å)$')
    # ax3.set_title(r'$u_\gamma(\mathbf{r})$', fontsize=16)
    # plt.tight_layout()
    # plt.savefig("outputs/bloch_function.png", dpi=300)
    plt.show()

    # --- reciprocal-space k-mesh ---
    # fig4, ax4 = plt.subplots(figsize=(6, 5))
    # plot_k_mesh(ax4, k_mesh=k_mesh)
    # plt.tight_layout()
    # plt.savefig("outputs/k_mesh.png", dpi=300)
    # plt.show()

if __name__ == "__main__":
    main()

