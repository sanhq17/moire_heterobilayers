import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0
psi  = np.radians(-50.0)
V    = 6.6 # meV
a0  = 3.32 # lattice constant in Angstroms
theta = np.radians(3.2)
aM = a0 / (2 * np.sin(theta / 2))  # moiré lattice constant
mstar = 0.35
kinetic_prefactor = 3809  # in units of meV*Angstrom^2

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

def H_k(k, n_shells):
    b1, b2 = triangular_lattice(aM)
    basis = generate_basis(b1, b2, n_shells)  # flat list of (g, (n1,n2))
    N = len(basis)
    H = np.zeros((N, N), dtype=complex)

    for i, (g, n) in enumerate(basis):
        k_plus_g = k + g
        H[i, i] = -kinetic_prefactor * np.linalg.norm(k_plus_g)**2 / mstar

        for j in range(i + 1, N):
            g2, n2 = basis[j]
            dm, dn = n2[0] - n[0], n2[1] - n[1]
            potential = V_fourier(dm, dn)
            if abs(potential) > 1e-10:
                H[i, j] = potential
                H[j, i] = np.conj(potential)

    return H

def make_path(n_pts):
    b1, b2 = triangular_lattice(aM)
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

def compute_bands(n_pts, n_shells=3):
    kpath, kdist, ticks = make_path(n_pts)

    # Diagonalize at first k-point to get system size
    e0 = np.linalg.eigvalsh(H_k(kpath[0], n_shells))
    N = len(e0)

    bands = np.empty((n_pts, N))
    states = np.empty((n_pts, N, N), dtype=complex)
    for i, k in enumerate(kpath):
        eigenvalues, eigenvectors = np.linalg.eigh(H_k(k, n_shells))
        bands[i] = eigenvalues[-N:][::-1]
        states[i] = eigenvectors[-N:][::-1]  # take top N and reverse to descending order 

    return kdist, bands, states, ticks

def main():
    # --- compute bands ---
    n_pts  = 300
    n_bands = 4
    n_shells = 4
    kdist, bands, states, ticks = compute_bands(n_pts, n_shells=n_shells)
    print(kdist)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(kdist, bands[:,0], color='r', lw=1)
    ax.plot(kdist, bands[:,1:n_bands], color='k', lw=1)

    tick_positions = [t[0] for t in ticks]
    tick_labels    = [t[1] for t in ticks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=12)
    for x in tick_positions:
        ax.axvline(x, color='k', lw=0.5, zorder=0)

    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylabel("Energy (meV)")
    ax.set_title("Moiré band structure")
    plt.tight_layout()
    plt.text(0.85, 0.87, "$\\theta = " + str(np.degrees(theta)) + "^\\circ$\n" + str(n_shells) + " shells", 
             transform=ax.transAxes, fontsize=12, ha='center', color='blue')
    plt.ylim([-60, 10])
    #plt.savefig("outputs/moire_bands.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
