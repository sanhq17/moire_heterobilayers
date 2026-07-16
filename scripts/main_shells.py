import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

hbar = 1.0
psi  = np.radians(-94.0)
V    = 6.6 # meV
a0  = 3.32 # lattice constant in Angstroms
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

def V_fourier(dm, dn, V_val, psi_val):
    if (dm, dn) in ((1, 0), (0, 1), (-1, -1)):
        return V_val * np.exp(1j * psi_val)
    if (dm, dn) in ((-1, 0), (0, -1), (1, 1)):
        return V_val * np.exp(-1j * psi_val)
    return 0.0

def H_k(k, basis, V_val, psi_val):
    N = len(basis)
    H = np.zeros((N, N), dtype=complex)

    for i, (g, n) in enumerate(basis):
        k_plus_g = k + g
        H[i, i] = -kinetic_prefactor * np.linalg.norm(k_plus_g)**2 / mstar

        for j in range(i + 1, N):
            g2, n2 = basis[j]
            dm, dn = n2[0] - n[0], n2[1] - n[1]
            potential = V_fourier(dm, dn, V_val, psi_val)
            if abs(potential) > 1e-10:
                H[i, j] = potential
                H[j, i] = np.conj(potential)

    return H

def make_path(n_pts, aM):
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

def compute_bands_from_path(kpath, n_bands, basis, V_val, psi_val):
    # Diagonalize at first k-point to get system size
    e0 = np.linalg.eigvalsh(H_k(kpath[0], basis, V_val, psi_val))
    N = len(e0)
    if n_bands is None:
        n_bands = N

    bands = np.empty((len(kpath), n_bands))
    for i, k in enumerate(kpath):
        evals = np.linalg.eigvalsh(H_k(k, basis, V_val, psi_val))  # sorted ascending by eigvalsh
        bands[i] = evals[-n_bands:][::-1]  # take top n_bands and reverse to descending order

    return bands

def main():
    # --- compute bands ---
    n_pts = 600
    n_bands = 4
    shells_list = [3]
    theta_min = 0.5
    theta_max = 6.0
    theta_init = 2.0
    V_min = 0.0
    V_max = 12.0
    V_init = V
    psi_min = -180.0
    psi_max = 180.0
    psi_init = np.degrees(psi)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.subplots_adjust(bottom=0.3)

    color_map = {
        1: "#1f77b4",
        3: "#ff7f0e",
        5: "#2ca02c",
        7: "#d62728",
    }

    def draw(theta_deg, V_val, psi_deg):
        ax.clear()
        theta = np.radians(theta_deg)
        psi_val = np.radians(psi_deg)
        aM = a0 / (2 * np.sin(theta / 2))  # moire lattice constant
        kpath, kdist, ticks = make_path(n_pts, aM)
        b1, b2 = triangular_lattice(aM)
        for n_shells in shells_list:
            basis = generate_basis(b1, b2, n_shells)
            bands = compute_bands_from_path(kpath, n_bands=n_bands, basis=basis, V_val=V_val, psi_val=psi_val)
            lines = ax.plot(kdist, bands, lw=1, color=color_map.get(n_shells, "k"))
            if lines:
                lines[0].set_label(f"{n_shells} shells")
                for line in lines[1:]:
                    line.set_label("_nolegend_")

        tick_positions = [t[0] for t in ticks]
        tick_labels = [t[1] for t in ticks]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=12)
        for x in tick_positions:
            ax.axvline(x, color="k", lw=0.5, zorder=0)

        ax.set_xlim(kdist[0], kdist[-1])
        ax.set_ylabel("Energy (meV)")
        ax.set_title("Moire band structure")
        ax.text(
            0.85,
            0.87,
            "$\\theta = " + str(np.round(theta_deg, 2)) + "^\\circ$",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            color="blue",
        )
        ax.legend(fontsize=9, frameon=False)
        ax.set_ylim([-60, 10])

    draw(theta_init, V_init, psi_init)

    theta_ax = fig.add_axes([0.15, 0.19, 0.7, 0.03])
    V_ax = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    psi_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])

    theta_slider = Slider(
        theta_ax,
        "theta (deg)",
        theta_min,
        theta_max,
        valinit=theta_init,
    )
    V_slider = Slider(
        V_ax,
        "V (meV)",
        V_min,
        V_max,
        valinit=V_init,
    )
    psi_slider = Slider(
        psi_ax,
        "psi (deg)",
        psi_min,
        psi_max,
        valinit=psi_init,
    )

    def on_slider_change(val):
        draw(theta_slider.val, V_slider.val, psi_slider.val)
        fig.canvas.draw_idle()

    theta_slider.on_changed(on_slider_change)
    V_slider.on_changed(on_slider_change)
    psi_slider.on_changed(on_slider_change)
    plt.show()


if __name__ == "__main__":
    main()