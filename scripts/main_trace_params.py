import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import utils
from functools import lru_cache
from Moire import Moire

hbar = 1.0
psi  = np.radians(40.0)
V    = 6.6 # meV
a0  = 3.32 # lattice constant in Angstroms
theta = np.radians(2.0)
aM = a0 / (2 * np.sin(theta / 2))  # moiré lattice constant
mstar = 0.35
h2m = 7618  # in units of meV*Angstrom^2
n_shells = 10

phi = psi#np.radians(120) + psi
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

# High-symmetry points
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

@lru_cache(maxsize=None)
def eigensystem(kx, ky):
    evals, evecs = np.linalg.eigh(H_k(np.array([kx, ky])))
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]

def make_path(n_pts):
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

def sample_bz(k_mesh):
    """Return a dense grid of k-points in the Brillouin zone for Wannier construction."""
    n1 = np.arange(-k_mesh, k_mesh+1)
    n2 = np.arange(-k_mesh, k_mesh+1)

    k_points = []
    for i in n1:
        for j in n2:
            temp_k = np.array([i * (2 * kappa_pp[0]) / k_mesh, j * (2 * kappa_p[1]) / k_mesh])
            k_points.append(temp_k)

    qx = 2 * kappa_pp[0] / k_mesh
    qy = 2 * kappa_p[1] / k_mesh
    KX, KY = np.meshgrid(qx * n1, qy * n2)
    print("k': ", kappa_p)
    print("k'': ", kappa_pp)
    return np.array(k_points), qx, qy, KX, KY

def wannier_wavefunction(x, y, k_mesh = 7, gauge_fixed=True):
    N1 = k_mesh # number of lattice points in each direction to sum over
    N2 = N1
    k_points = sample_k_points(k_mesh)

    phi = np.radians(0)
    c, s = np.cos(phi), np.sin(phi)
    rot = np.array([[c, -s], [s, c]])
    R0 = rot @ np.array([0, 0])  # shift to center Wannier function

    wannier = 0.0
    for k in k_points:
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

def dual_state(k, q):
    eigenvalues, eigenvectors = np.linalg.eigh(H_k(k))
    idx = np.argsort(eigenvalues)[::-1]
    states = eigenvectors[:, idx]

    zk = states[:, 0]  # top band state

    eigenvalues, eigenvectors = np.linalg.eigh(H_k(k+q))
    idx = np.argsort(eigenvalues)[::-1]
    states = eigenvectors[:, idx]

    zkq = states[:, 0]  # top band state
    S = np.vdot(zk, zkq)
    return zkq / S

def covariant_derivative(k, q):
    return (dual_state(k, q) - dual_state(k, -q)) / (2 * np.linalg.norm(q))

def berry_curvature(k, qx, qy):
    dX = covariant_derivative(k, np.array([qx, 0]))
    dY = covariant_derivative(k, np.array([0, qy]))
    return - 2 * np.imag(np.vdot(dX, dY))

def qgt_ij(k, qi, qj):
    dUi = covariant_derivative(k, qi)
    dUj = covariant_derivative(k, qj)
    return np.vdot(dUi, dUj)

def qmetric_ij(k, qi, qj):
    # is this just simply the real part?
    return (qgt_ij(k, qi, qj) + qgt_ij(k, qj, qi)) / 2

def qmetric(k, qx, qy):
    q0 = np.array([qx, 0])
    q1 = np.array([0, qy])
    qs = [q0, q1]

    g = np.zeros((2, 2), dtype=float)

    for i in range(2):
        for j in range(2):
            g[i, j] = qmetric_ij(k, qs[i], qs[j])

    return g

def fourier_gaussian(k):
    return np.sqrt(aW**2 / np.pi) * np.exp(- aW**2 * np.linalg.norm(k)**2 / 2)

def plot_wavefunction(ax, wavefunctions, extent):
    im = ax.imshow(np.abs(wavefunctions)**2, extent=extent, origin='lower', 
                   interpolation='none', cmap='viridis')
    return im

def plot_hexagonal_bz(ax):
    phi = np.radians(60)
    c, s = np.cos(phi), np.sin(phi)
    rot60 = np.array([[c, -s], [s, c]])

    corners = np.array([
        kappa_pp,
        kappa_p,
        rot60 @ kappa_p,
        -kappa_pp,
        -kappa_p,
        -(rot60 @ kappa_p),
        kappa_pp,
    ])

    ax.plot(corners[:, 0], corners[:, 1], color='white', lw=2.0, zorder=3)
    ax.scatter(
        [kappa_pp[0], kappa_p[0]],
        [kappa_pp[1], kappa_p[1]],
        color='white',
        s=18,
        zorder=4,
    )

def chern_number_from_grid(berry, KX, KY, kappa_p, kappa_pp):
    """Integrate Berry curvature over the hexagonal 1BZ defined by kappa_p/kappa_pp.

    Returns the dimensionless Chern number C = (1/2pi) * integral Omega(k) d^2k.
    """
    phi = np.radians(60)
    c, s = np.cos(phi), np.sin(phi)
    rot60 = np.array([[c, -s], [s, c]])

    corners = np.array([
        kappa_pp,
        kappa_p,
        rot60 @ kappa_p,
        -kappa_pp,
        -kappa_p,
        -(rot60 @ kappa_p),
    ])

    pts = np.vstack([KX.ravel(), KY.ravel()]).T
    mask = Path(corners).contains_points(pts).reshape(KX.shape)

    # grid spacings (assume uniform)
    dkx = float(np.abs(KX[0, 1] - KX[0, 0])) if KX.shape[1] > 1 else 1.0
    dky = float(np.abs(KY[1, 0] - KY[0, 0])) if KY.shape[0] > 1 else 1.0
    dA = dkx * dky

    total = np.sum(np.real(berry[mask])) * dA
    C = total / (2.0 * np.pi)
    return C

def integral_over_hexagonal_bz(values, KX, KY, kappa_p, kappa_pp):
    """Integrate a scalar field over the same hexagonal 1BZ used in the plots."""
    phi = np.radians(60)
    c, s = np.cos(phi), np.sin(phi)
    rot60 = np.array([[c, -s], [s, c]])

    corners = np.array([
        kappa_pp,
        kappa_p,
        rot60 @ kappa_p,
        -kappa_pp,
        -kappa_p,
        -(rot60 @ kappa_p),
    ])

    pts = np.vstack([KX.ravel(), KY.ravel()]).T
    mask = Path(corners).contains_points(pts).reshape(KX.shape)

    dkx = float(np.abs(KX[0, 1] - KX[0, 0])) if KX.shape[1] > 1 else 1.0
    dky = float(np.abs(KY[1, 0] - KY[0, 0])) if KY.shape[0] > 1 else 1.0
    dA = dkx * dky

    return np.sum(np.real(values[mask])) * dA

def bz_area(KX, KY, kappa_p, kappa_pp):
    """Return the area of the hexagonal 1BZ using the same grid spacing/mask."""
    ones = np.ones_like(KX, dtype=float)
    return integral_over_hexagonal_bz(ones, KX, KY, kappa_p, kappa_pp)

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

def integrate_spread(psi, x, y):
    density = np.abs(psi) ** 2
    r2 = x**2 + y**2
    return np.trapz(np.trapz(density * r2, x, axis=1), y, axis=0)

def main():
    # --- compute bands ---
    global aM, basis, b1, b2, theta
    sys = Moire(theta_deg=np.degrees(0.5),
                V=V,
                psi_deg=psi)
    
    k_mesh = 8
    angles = np.radians(np.linspace(1.5, 3.0, 10))
    angles2 = np.linspace(angles[0], angles[-1], 100)
    r0s = []
    rWs = []

    plt.figure(figsize=(6, 4))
    aMs = a0 / (2 * np.sin(angles2 / 2))
    print("psi:", np.degrees(psi))
    rHarm = utils.spread_harmonic_approx(aMs, V, psi)
    rPert = utils.spread_perturbation(aMs, V, psi)
    plt.plot(np.degrees(angles2), rHarm, label='Harmonic')
    plt.plot(np.degrees(angles2), rPert, label='Perturbed', linestyle='--')

    for angle in angles:
        theta = angle
        aM = a0 / (2 * np.sin(angle / 2))  # moiré lattice constant
        sys.theta_deg = np.degrees(angle)
        print(sys.theta_deg)
        print(sys.theta)

        b1, b2 = triangular_lattice(aM)
        basis = generate_basis(b1, b2, n_shells)

        k_points, qx, qy, KX, KY = sample_bz(k_mesh)
        berry = np.zeros(KX.shape, dtype=float)
        metric = np.zeros((2, 2) + KX.shape, dtype=float)
        for i in range(KX.shape[0]):
            for j in range(KX.shape[1]):
                k = np.array([KX[i, j], KY[i, j]])
                berry[i, j] = berry_curvature(k, qx, qy)
                metric[:, :, i, j] = qmetric(k, qx, qy)
                
        # extent = [KX.min(), KX.max(), KY.min(), KY.max()]

        # third figure: trace and determinant of the metric tensor
        # fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
        trace = metric[0, 0] + metric[1, 1]
        det_metric = metric[0, 0] * metric[1, 1] - metric[0, 1] * metric[1, 0]

        trace_integral = integral_over_hexagonal_bz(trace, KX, KY, kappa_p, kappa_pp)
        area = bz_area(KX, KY, kappa_p, kappa_pp)
        trace_avg = trace_integral / area if area != 0 else np.nan
        r0s.append(np.sqrt(trace_avg))
        print(f"Integral of Tr g(k) over the 1BZ = {trace_integral:.6f}")
        print(f"r0^2 = {trace_avg:.6f}")

        scale = 0.2
        lim = 2 * utils.spread_harmonic_approx(aM, V, psi)
        x = np.linspace(-lim, lim, 400)
        y = np.linspace(-lim, lim, 400)
        X, Y = np.meshgrid(x, y)
        k_points = sys.mesh_k_unit_cell(Nx=k_mesh, Ny=k_mesh)
        print("hola")

        wannier = wannier_wavefunction(X, Y, gauge_fixed=True)
        print("hola")
        rW = np.sqrt(integrate_spread(wannier, x, y))
        rWs.append(rW)
        print(f"Spread of Wannier function = {rW:.6f}")
        # im3 = ax3.imshow(trace, extent=extent, origin='lower',
        #                 interpolation='bilinear', cmap='viridis')
        # plot_hexagonal_bz(ax3)
        # fig3.colorbar(im3, ax=ax3, shrink=0.9)
        # ax3.set_aspect('equal', adjustable='box')
        # ax3.set_xlim(extent[0] * .7, extent[1] * .7)
        # ax3.set_ylim(extent[2] * .7, extent[3] * .7)
        # ax3.set_xlabel(r'$k_x$')
        # ax3.set_ylabel(r'$k_y$')
        # ax3.set_title(r"Tr\,$g(\mathbf{k})$", fontsize=16)

        # im4 = ax4.imshow(2 * np.sqrt(det_metric), extent=extent, origin='lower',
        #                 interpolation='bilinear', cmap='viridis')
        # plot_hexagonal_bz(ax4)
        # fig3.colorbar(im4, ax=ax4, shrink=0.9)
        # ax4.set_aspect('equal', adjustable='box')
        # ax4.set_xlim(extent[0] * .7, extent[1] * .7)
        # ax4.set_ylim(extent[2] * .7, extent[3] * .7)
        # ax4.set_xlabel(r'$k_x$')
        # ax4.set_ylabel(r'$k_y$')
        # ax4.set_title(r"2$\sqrt{\det\,g(\mathbf{k})}$", fontsize=16)
        # plt.tight_layout()
        # plt.savefig("outputs/qmetric_trace_det.png", dpi=300)
        #plt.show()
        
        plt.scatter(np.degrees(angles)[:len(r0s)], r0s, s=30, facecolor='r', edgecolor='none')
        plt.scatter(np.degrees(angles)[:len(rWs)], rWs, s=50, facecolor='none', edgecolor='blue')
        plt.xlabel(r'$\theta\quad(\deg)$')
        plt.ylabel(r'length ($\AA$)')
        plt.grid(True)
        plt.text(1.7, 16, "V = " + str(V) + " meV\n $\psi = " + str(np.degrees(psi)) + "^\circ$\n shells=" + str(n_shells),
                fontsize=12, ha='center', color='blue')
        plt.savefig("outputs/r0s.png", dpi=300)
    plt.scatter(np.degrees(angles)[:len(rWs)], rWs, s=50, label=r'$r_W$', facecolor='none', edgecolor='blue')
    plt.scatter(np.degrees(angles)[:len(r0s)], r0s, label=r'$l_0$', s=30, facecolor='r', edgecolor='none')
    plt.legend()
    plt.savefig("outputs/r0s.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

