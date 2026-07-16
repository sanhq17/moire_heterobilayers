import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0
psi  = np.radians(-94.0)
V    = 6.6 # meV
a0  = 3.3283 # lattice constant in Angstroms
theta = np.radians(2.0)
aM = a0 / (2 * np.sin(theta / 2))  # moiré lattice constant
mstar = 0.35
h2m = 7618  # in units of meV*Angstrom^2
bandwidth_to_hopping = 9 # conversion factor from bandwidth to effective hopping t

def triangular_lattice(a):
    b1 = np.array([4 * np.pi / (np.sqrt(3) * a), 0.0])
    phi = np.radians(120)
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    b2 = R @ b1
    return b1, b2

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

def V_fourier(dm, dn):
    if (dm, dn) in ((1, 0), (0, 1), (-1, -1)):
        return V * np.exp( 1j * psi)
    if (dm, dn) in ((-1, 0), (0, -1), (1, 1)):
        return V * np.exp(-1j * psi)
    return 0.0

def H_k(k, n_shells, aM):
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

def make_path(n_pts, aM):
    b1, b2 = triangular_lattice(aM)
    gamma  = np.zeros(2)
    kappa = ( b1 + 2 * b2) / 3          # κ
    kappa_p  = (2 * b1 + b2) / 3        # κ'
    kappa_pp = ( b1 - b2 ) / 3          # κ''

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

def compute_bands(n_pts, n_bands=4, n_shells=3, aM=None):
    kpath, kdist, ticks = make_path(n_pts, aM)

    # Diagonalize at first k-point to get system size
    e0 = np.linalg.eigvalsh(H_k(kpath[0], n_shells, aM))
    N = len(e0)
    if n_bands is None:
        n_bands = N

    bands = np.empty((n_pts, n_bands))
    for i, k in enumerate(kpath):
        evals = np.linalg.eigvalsh(H_k(k, n_shells, aM))   # sorted ascending by eigvalsh
        bands[i] = evals[-n_bands:][::-1]  # take top n_bands and reverse to descending order 

    return kdist, bands, ticks


def f_theta(theta):
    phi = np.radians(120) + psi
    beta = 16 * np.pi**2 * np.cos(phi)
    A = (h2m / (2 * mstar)) * np.sqrt(mstar * V * beta / h2m)
    B = np.sqrt((mstar * V * beta) / h2m) / 4 - theta / a0
    C = np.exp(- a0 * np.sqrt(mstar * V * beta / h2m) / (4 * theta))
    return A * B * C


def main():
    # --- compute bands ---
    n_pts  = 300
    n_bands = 4
    n_shells_values = [2, 3, 4, 5, 6, 7]

    thetas = np.linspace(0.1, 3, 21)

    # --- plot ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    for idx, n_shells in enumerate(n_shells_values):
        color = colors[idx % len(colors)] if colors else None
        W = []
        gap = []
        for th in thetas:
            theta = np.radians(th)
            aM = a0 / (2 * np.sin(theta / 2))  # moiré lattice constant
            _, bands, _ = compute_bands(n_pts, n_bands=n_bands, n_shells=n_shells, aM=aM)

            bandwidths = np.max(bands, axis=0) - np.min(bands, axis=0)
            W.append(bandwidths[0])  # track bandwidth of the top band
            bandgap = bands[-1, 0] - bands[-1, 1]
            gap.append(bandgap)

        ax_left.scatter(
            thetas,
            np.array(W) / bandwidth_to_hopping,
            color=color,
            lw=2,
            label=f"n_shells={n_shells}",
            s=10,
        )
        ax_right.plot(thetas, gap, color=color, lw=2, label=f"n_shells={n_shells}")

    ax_left.plot(thetas, f_theta(np.radians(thetas)), color='k', lw=1.2)
    ax_left.set_xlim(thetas[0], 1.1)
    ax_left.set_ylim(-0.02, 0.2)
    ax_left.set_ylabel("t (meV)")
    ax_left.set_xlabel("$\\theta$ (deg)")
    ax_left.set_title("Hopping amplitude")
    ax_left.legend(frameon=False)

    ax_right.set_xlim(thetas[0], 1.1)
    ax_right.set_ylabel("$\Delta\epsilon|_\kappa$ (meV)")
    ax_right.set_xlabel("$\\theta$ (deg)")
    ax_right.set_title("Gap vs theta")
    ax_right.legend(frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/moire_bandwidth_zoom.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()