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
    # --- gap vs psi for n_shells in [2, 3, 4] ---
    psi_degs = np.linspace(-180, 180, 100)
    theta_for_psi_deg = 1.0
    theta_for_psi = np.radians(theta_for_psi_deg)
    aM_for_psi = a0 / (2 * np.sin(theta_for_psi / 2))

    fig_psi, ax_psi = plt.subplots(figsize=(6, 4))
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    global psi
    psi_original = psi
    n_shells_values = [2, 3, 4]
    for idx, n_shells_psi in enumerate(n_shells_values):
        color = colors[idx % len(colors)] if colors else None
        gap_vs_psi = []
        for psi_deg in psi_degs:
            psi = np.radians(psi_deg)
            _, bands, _ = compute_bands(n_pts, n_bands=n_bands, n_shells=n_shells_psi, aM=aM_for_psi)
            bandgap = bands[-1, 0] - bands[-1, 1]
            gap_vs_psi.append(bandgap)
        ax_psi.plot(psi_degs, gap_vs_psi, color=color, lw=2, label=f"n_shells={n_shells_psi}")
    psi = psi_original

    ax_psi.set_xlim(psi_degs[0], psi_degs[-1])
    ax_psi.set_xlabel("$\\psi$ (deg)")
    ax_psi.set_ylabel("$\Delta\epsilon|_\kappa$ (meV)")
    ax_psi.set_title(f"Gap vs psi (theta = {theta_for_psi_deg:.1f} deg)")
    ax_psi.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("outputs/moire_gap_vs_psi_shells.png", dpi=300)
    plt.show()

    # --- gap vs V at fixed psi ---
    V_values = np.linspace(2.0, 10.0, 17)
    psi_fixed_deg = -94.0
    n_shells_v = 4

    fig_v, ax_v = plt.subplots(figsize=(6, 4))
    global V
    V_original = V
    psi = np.radians(psi_fixed_deg)
    gap_vs_v = []
    for V_val in V_values:
        V = V_val
        _, bands, _ = compute_bands(n_pts, n_bands=n_bands, n_shells=n_shells_v, aM=aM_for_psi)
        bandgap = bands[-1, 0] - bands[-1, 1]
        gap_vs_v.append(bandgap)
    V = V_original
    psi = psi_original

    ax_v.plot(V_values, gap_vs_v, color='tab:purple', lw=2)
    ax_v.set_xlim(V_values[0], V_values[-1])
    ax_v.set_xlabel("V (meV)")
    ax_v.set_ylabel("$\\Delta\\epsilon|_\\kappa$ (meV)")
    ax_v.set_title(
        f"Gap vs V (psi = {psi_fixed_deg:.1f} deg, theta = {theta_for_psi_deg:.1f} deg, n_shells = {n_shells_v})"
    )
    plt.tight_layout()
    plt.savefig("outputs/moire_gap_vs_V.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()