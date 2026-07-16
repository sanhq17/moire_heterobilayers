import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from functools import cached_property
from functools import lru_cache
import utilsHomo


HBAR2_M = utilsHomo.HBAR2_M  # hbar^2 / (m) in units of meV*Angstrom^2
MSTAR = utilsHomo.MSTAR  # effective mass in units of electron mass

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"  # Optional for advanced math
})

class MoireHomo:
    def __init__(self, a0=3.472, theta_deg=1.2, V=8, w=-8.5, psi_deg=-89.6, n_shells=7):
        self.a0 = a0
        self.theta_deg = theta_deg
        self.V = V
        self.psi_deg = psi_deg
        self.w = w
        self.n_shells = n_shells

    @property
    def theta(self):
        return np.radians(self.theta_deg)
    
    @property
    def psi(self):
        return np.radians(self.psi_deg)
    
    @property
    def aM(self):
        return self.a0 / (2 * np.sin(self.theta / 2))  # moiré lattice constant
    
    @property
    def a1(self):
        return np.array([np.sqrt(3) * self.aM / 2, self.aM / 2])
    
    @property
    def a2(self):
        return np.array([0, self.aM])
    
    @property
    def A_unit_cell(self):
        return np.linalg.norm(np.cross(self.a1, self.a2))

    @property
    def b1(self):
        return np.array([4 * np.pi / (np.sqrt(3) * self.aM), 0.0])

    @property
    def b2(self):
        phi = np.radians(120)
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        return R @ self.b1
    
    @property
    def b3(self):
        phi = np.radians(120)
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        return R @ self.b2

    @property
    def K(self):
        return (self.b1 + self.b2) / 3
    
    @property
    def M(self):
        return (self.b1 + self.b2) / 2
    
    @property
    def basis(self):
        """
        Returns adimensional basis vectors from the reciprocal lattice.
        """
        return utilsHomo.compute_basis_adim(self.n_shells)
    
    @property
    def gamma(self):
        return np.zeros(2)

    @property
    def kappa_pp(self):
        kappa_pp = ( 2 * self.b1 + self.b2 ) / 3
        return kappa_pp
    
    @property
    def kappa_p(self):
        phi = np.radians(60)
        c, s = np.cos(phi), np.sin(phi)
        R1 = np.array([[c, -s], [s, c]])
        kappa_p = R1 @ self.kappa_pp
        return kappa_p
    
    @property
    def kappa(self):
        phi = np.radians(180)
        c, s = np.cos(phi), np.sin(phi)
        R2 = np.array([[c, -s], [s, c]])
        kappa = R2 @ self.kappa_p
        return kappa
    
    @property
    def kappa_minus(self):
        return self.kappa
    
    @property
    def kappa_plus(self):
        phi = np.radians(60)
        c, s = np.cos(phi), np.sin(phi)
        R1 = np.array([[c, -s], [s, c]])
        return R1 @ self.kappa_minus
    
    @property
    def kappa_plus_p(self):
        phi = np.radians(120)
        c, s = np.cos(phi), np.sin(phi)
        R1 = np.array([[c, -s], [s, c]])
        return R1 @ self.kappa_plus
    
    @property
    def A_BZ(self):
        return ( 3 * np.sqrt(3) / 2 ) * np.linalg.norm(self.kappa)**2
    
    @property
    def default_path(self):
        path = [self.kappa_plus_p, self.gamma, self.kappa_minus, self.kappa_plus, self.kappa_plus_p]
        labels = [r'$\kappa_+^\prime$', r'$\gamma$', r'$\kappa_-$', r'$\kappa_+$', r'$\kappa_+^\prime$']
        return path, labels
    
    @property
    def BZ_path(self):
        phi = np.radians(60)
        c, s = np.cos(phi), np.sin(phi)
        rot60 = np.array([[c, -s], [s, c]])

        verts = np.array([
            self.kappa_pp,
            self.kappa_p,
            rot60 @ self.kappa_p,
            -self.kappa_pp,
            self.kappa,
            rot60 @ self.kappa
        ])
        
        return Path(verts)
    
    @property
    def BZ_area(self):
        return 1.5 * np.sqrt(3) * np.linalg.norm(self.kappa_p)**2
    
    def _V_fourier(self, dm, dn, l):
        if (dm, dn) in ((1, 0), (0, 1), (-1, -1)):
            return self.V * np.exp( 1j * l * self.psi)
        if (dm, dn) in ((-1, 0), (0, -1), (1, 1)):
            return self.V * np.exp(-1j * l * self.psi)
        return 0.0
    
    def _T_fourier(self, dm, dn):
        if (dm, dn) in ((0, 0), (-1, 0), (-1, -1)):
            return self.w
        return 0.0
    
    def H_k(self, k):
        N = len(self.basis)
        H = np.zeros((2 * N, 2 * N), dtype=complex)

        for i, (g, n) in enumerate(self.basis):
            # --- Diagonal blocks ---
            # Bottom layer (b)
            k_plus_g_b = k + g / self.aM - self.kappa_plus
            H[i, i] = -HBAR2_M * np.linalg.norm(k_plus_g_b)**2 / (2 * MSTAR)

            # Top layer (t)
            k_plus_g_t = k + g / self.aM - self.kappa_minus
            H[N + i, N + i] = -HBAR2_M * np.linalg.norm(k_plus_g_t)**2 / (2 * MSTAR)

            # --- Intra-layer potentials (Delta_b and Delta_t) ---
            for j in range(i + 1, N):
                _, n2 = self.basis[j]
                dm, dn = n2[0] - n[0], n2[1] - n[1]

                # Bottom layer potential Delta_b(g' - g)
                V_b = self._V_fourier(dm, dn, l=1)
                # Top layer potential Delta_t(g' - g)
                V_t = self._V_fourier(dm, dn, l=-1)
                
                if np.abs(V_b) > 1e-10:
                    H[i, j] = V_b
                    H[j, i] = np.conj(V_b)
    
                if np.abs(V_t) > 1e-10:
                    H[N + i, N + j] = V_t
                    H[N + j, N + i] = np.conj(V_t)

            # --- Off-diagonal tunneling block (Delta_T) ---
            # Couples bottom layer at g to top layer at g'
            for j in range(N):
                _, n2 = self.basis[j]
                dm, dn = n2[0] - n[0], n2[1] - n[1]

                # Delta_T is nonzero only for 3 momentum transfers (0, -G2, -G3)
                T = self._T_fourier(dm, dn)
                if np.abs(T) > 1e-10:
                    H[i, N + j] = T          # bottom -> top block
                    H[N + j, i] = np.conj(T) # top -> bottom block (Hermitian conj)

        return H
    
    def bloch_u(self, k, r, gauge_fixed=True):
        _, eigenvectors = np.linalg.eigh(self.H_k(k))
        states = eigenvectors[:, -1]  # take top band state

        zk = states[:, 0]  # top band state
        bloch0 = 0.0

        for m, (g, _) in enumerate(self.basis):
            bloch0 += zk[m] * np.exp(1.0j * (g / self.aM) @ r)

        if gauge_fixed:
            overlap = np.sum(np.conj(zk))
            bloch_u = bloch0 * np.exp(1.0j * np.angle(overlap))
        else:
            bloch_u = bloch0

        return bloch_u / np.sqrt(self.A_unit_cell)
    
    def make_k_path(self, path, labels, n_pts):
        segments = []
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            label_start, label_end = labels[i], labels[i + 1]
            segments.append((start, end, label_start, label_end))

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
    
    def calculate_bands(self, kpath):
        # Diagonalize at first k-point to get system size
        e0 = np.linalg.eigvalsh(self.H_k(kpath[0]))
        N = len(e0)
        n_pts = kpath.shape[0]

        bands = np.empty((n_pts, N))
        states = np.empty((n_pts, N, N), dtype=complex)
        for i, k in enumerate(kpath):
            eigenvalues, eigenvectors = np.linalg.eigh(self.H_k(k))
            bands[i] = eigenvalues[-N:][::-1]
            states[i] = eigenvectors[-N:][::-1]  # take top N and reverse to descending order

        return bands, states
    
    def calculate_gap(self, k_point=None):
        if k_point is None:
            k_point = self.kappa
        
        eigenvalues = np.linalg.eigvalsh(self.H_k(k_point))
        sorted_eigvals = np.sort(eigenvalues)[::-1]  # descending order

        return sorted_eigvals[1] - sorted_eigvals[0]
    
    def calculate_bandwidth(self):
        eigenvalues_gamma = np.linalg.eigvalsh(self.H_k(self.gamma))
        sorted_eigvals_gamma = np.sort(eigenvalues_gamma)[::-1]  # descending order

        eigenvalues_kappa = np.linalg.eigvalsh(self.H_k(self.kappa))
        sorted_eigvals_kappa = np.sort(eigenvalues_kappa)[::-1]  # descending order

        return sorted_eigvals_gamma[0] - sorted_eigvals_kappa[0]
    
    def moire_potential(self, r):
        return 2 * self.V * (np.cos(self.b1 @ r + self.psi) + np.cos(self.b2 @ r + self.psi) + np.cos(self.b3 @ r + self.psi))
    
    def mesh_k_unit_cell(self, Nx=7, Ny=7):
        n1 = np.arange(Nx)
        n2 = np.arange(Ny)

        k_points = []
        for i in n1:
            for j in n2:
                k_points.append(i * self.b1 / Nx + j * self.b2 / Ny)

        return np.array(k_points)
    
    def mesh_r_space(self, Nx=7, Ny=7):
        n1 = np.arange(Nx)
        n2 = np.arange(Ny)

        N1, N2 = np.meshgrid(n1, n2, indexing='ij')

        r_points = (N1[..., None] * self.b1 / Nx +
                    N2[..., None] * self.b2 / Ny)

        return r_points.reshape(-1, self.b1.shape[-1])

    def wannier_wavefunction(self, x, y, k_points, gauge_fixed=True, R0=np.zeros(2)):
        wannier = 0.0
        for k in k_points:
            _, eigenvectors = np.linalg.eigh(self.H_k(k))
            zk = eigenvectors[:, -1]  # top band state
            bloch0 = 0.0

            for m, (g, _) in enumerate(self.basis):
                k_plus_g = k + g / self.aM
                bloch0 += zk[m] * np.exp(1.0j * (k_plus_g[0] * x + k_plus_g[1] * y))
            shift = np.exp(-1.0j * k @ R0)  # optional phase factor to center Wannier function

            if gauge_fixed:
                overlap = np.sum(np.conj(zk))
                wannier += bloch0 * shift * np.exp(1.0j * np.angle(overlap))
            else:
                wannier += bloch0 * shift

        wannier = wannier / (k_points.shape[0] * np.sqrt(self.A_unit_cell))  # normalize by number of k-points and real space unit cell area

        return wannier
    
    def integrate_wannier(self, fun, X, Y, k_points):
        wannier = self.wannier_wavefunction(X, Y, k_points)
        density = np.abs(wannier)**2

        integral = np.trapz(
            np.trapz(fun(X, Y) * density, X[0,:], axis=1),
            Y[:,0],
            axis=0
        )

        return integral
    
    def wannier_norm(self, X, Y, k_points):
        wannier = self.wannier_wavefunction(X, Y, k_points)
        density = np.abs(wannier)**2

        integral = np.trapz(
            np.trapz(density, X[0,:], axis=1),
            Y[:,0],
            axis=0
        )

        return integral
    
    def dual_states(self, KX, KY):
        u = np.empty((KX.shape[0], KX.shape[1], 2 * len(self.basis)), dtype=complex)

        for i in range(KX.shape[0]):
            for j in range(KX.shape[1]):
                kx = KX[i, j]
                ky = KY[i, j]
                k_temp = np.array([kx, ky])
                
                _, eigvecs = np.linalg.eigh( self.H_k( k_temp ) )
                u[i, j] = eigvecs[:, -1]  # take top band state

        u_xp = np.roll(u, -1, axis=0)
        u_xm = np.roll(u,  1, axis=0)

        u_yp = np.roll(u, -1, axis=1)
        u_ym = np.roll(u,  1, axis=1)

        Sxp = np.sum(np.conj(u) * u_xp, axis=-1)
        Sxm = np.sum(np.conj(u) * u_xm, axis=-1)

        Syp = np.sum(np.conj(u) * u_yp, axis=-1)
        Sym = np.sum(np.conj(u) * u_ym, axis=-1)

        dual_xp = u_xp / Sxp[..., None]
        dual_xm = u_xm / Sxm[..., None]

        dual_yp = u_yp / Syp[..., None]
        dual_ym = u_ym / Sym[..., None]

        return dual_xp, dual_xm, dual_yp, dual_ym
    
    def covariant_derivatives(self, KX, KY):
        dual_xp, dual_xm, dual_yp, dual_ym = self.dual_states(KX, KY)

        dkx = KX[0, 1] - KX[0, 0]
        dky = KY[1, 0] - KY[0, 0]

        dX = (dual_xp - dual_xm) / (2 * dkx)
        dY = (dual_yp - dual_ym) / (2 * dky)

        return dX, dY

    def berry_curvature(self, KX, KY):
        dX, dY = self.covariant_derivatives(KX, KY)
        overlap = np.sum(np.conj(dX) * dY, axis=-1)
        berry = -2 * np.imag(overlap)

        # remove outer frame (unphysical boundary points)
        berry = berry[1:-1, 1:-1]
        KX = KX[1:-1, 1:-1]
        KY = KY[1:-1, 1:-1]

        return berry, KX, KY
    
    def chern_number_from_grid(self, berry, KX, KY):
        """Integrate Berry curvature over the hexagonal 1BZ.
        """
        phi = np.radians(60)
        c, s = np.cos(phi), np.sin(phi)
        rot60 = np.array([[c, -s], [s, c]])

        corners = np.array([
            self.kappa_pp,
            self.kappa_p,
            rot60 @ self.kappa_p,
            -self.kappa_pp,
            -self.kappa_p,
            -(rot60 @ self.kappa_p),
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
    
    def quantum_metric(self, KX, KY):
        dX, dY = self.covariant_derivatives(KX, KY)

        g_xx = np.real(np.sum(np.conj(dX) * dX, axis=-1))
        g_yy = np.real(np.sum(np.conj(dY) * dY, axis=-1))

        g_xy = np.real(np.sum(np.conj(dX) * dY, axis=-1))

        # remove outer frame (unphysical boundary points)
        g_xx = g_xx[1:-1, 1:-1]
        g_xy = g_xy[1:-1, 1:-1]
        g_yy = g_yy[1:-1, 1:-1]
        KX = KX[1:-1, 1:-1]
        KY = KY[1:-1, 1:-1]

        return g_xx, g_xy, g_yy, KX, KY
    
    def integrate_trace(self, KX, KY):
        g_xx, _, g_yy, KXX, KYY = self.quantum_metric(KX, KY)
        trace = g_xx + g_yy

        points = np.column_stack((KXX.ravel(), KYY.ravel()))    # Flatten grid points
        mask = self.BZ_path.contains_points(points)             # Test whether points are inside
        mask = mask.reshape(KXX.shape)                          # Reshape back to grid shape

        trace_masked = np.where(mask, trace, 0.0)
        integral = np.trapz(
            np.trapz(trace_masked, KXX[0,:], axis=1),
            KYY[:,0],
            axis=0
        )

        #average = np.mean(trace[mask])

        return trace_masked, integral, KXX, KYY

    ## Plotting functions
    def plot_bands(self, ax, bands, kdist, ticks, max_bands=4):
        ax.plot(kdist, bands[:, 0], color='r', lw=1)
        ax.plot(kdist, bands[:, 1:max_bands], color='k', lw=1)

        tick_positions = [t[0] for t in ticks]
        tick_labels    = [t[1] for t in ticks]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=12)
        for x in tick_positions:
            ax.axvline(x, color='k', lw=0.5, zorder=0)

        ax.set_xlim(kdist[0], kdist[-1])
        ax.set_ylabel("Energy (meV)")
        ax.set_title("Moiré band structure")

        return ax

    def plot_moire_potential(self, ax, extent, resolution=200):
        x = np.linspace(extent[0], extent[1], resolution)
        y = np.linspace(extent[2], extent[3], resolution)
        X, Y = np.meshgrid(x, y)

        pot = np.empty_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                r = np.array([X[i, j], Y[i, j]])
                pot[i, j] = self.moire_potential(r)

        im = ax.imshow(pot, extent=extent, origin='lower', cmap='RdBu_r')
        return im
    
    def plot_triangular_lattice(self, ax, extent):
        # Overlay triangular lattice as dashed lines (real-space primitive vectors)
        r1 = np.array([0, 1.0]) * self.aM
        r2 = np.array([np.sqrt(3) / 2.0, 1.0 / 2.0]) * self.aM
        r3 = r2 - r1

        x_min, x_max, y_min, y_max = extent

        # Build a grid of lattice sites large enough to cover the plotting extent.
        n_max = int(np.ceil(max(x_max - x_min, y_max - y_min) / self.aM)) + 4
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
                    #ax.scatter([x0, x1], [y0, y1], color='r', s=10, alpha=0.9, edgecolors='k', linewidth=1.0)
                    ax.plot([x0, x1], [y0, y1], color='k', ls='--', lw=0.8, alpha=0.9)
        
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    def plot_wavefunction(self, ax, wavefunction, extent, cmap='RdBu_r', norm=None):
        im = ax.imshow(np.abs(wavefunction)**2, extent=extent, origin='lower', 
                    interpolation='bilinear', cmap=cmap, norm=norm)
        return im
    
    def plot_wavefunction_phase(self, ax, wavefunction, extent):
        im = ax.imshow(np.angle(wavefunction), extent=extent, origin='lower', 
                    interpolation='bilinear', cmap='hsv')
        return im
    
    def plot_wannier_function(self, ax, X, Y, k_points, norm=None):
        wannier = self.wannier_wavefunction(X, Y, k_points, gauge_fixed=True)
        extent = [X.min(), X.max(), Y.min(), Y.max()]
        return self.plot_wavefunction(ax, wannier, extent, norm=norm)
    
    def plot_berry_curvature(self, ax, KX, KY, abs_val=False):
        berry, KXX, KYY = self.berry_curvature(KX, KY)
        if abs_val:
            berry = np.abs(berry)
            cmap = 'Reds'
        else:
            cmap = 'RdBu_r'

        # rebuild matching extent
        extent = [KXX.min(), KXX.max(), KYY.min(), KYY.max()]

        self.plot_hexagonal_bz(ax)
        im = ax.imshow(berry, extent=extent, origin='lower', 
                    interpolation='bilinear', cmap=cmap)
        ax.set_title(r"$\Omega(\mathbf{k})$")
        plt.colorbar(im, ax=ax)
        
        return im
    
    def plot_hexagonal_bz(self, ax, lw=1.5):
        phi = np.radians(60)
        c, s = np.cos(phi), np.sin(phi)
        rot60 = np.array([[c, -s], [s, c]])

        corners = np.array([
            self.kappa_pp,
            self.kappa_p,
            rot60 @ self.kappa_p,
            -self.kappa_pp,
            -self.kappa_p,
            -(rot60 @ self.kappa_p),
            self.kappa_pp,
        ])

        ax.plot(corners[:, 0], corners[:, 1], color='k', lw=lw, zorder=3)
        ax.text((self.kappa[0] + self.kappa_pp[0]) / 2, self.kappa[1], "1BZ", fontsize=12, color='k')
        # ax.scatter(
        #     [self.kappa_pp[0], self.kappa_p[0]],
        #     [self.kappa_pp[1], self.kappa_p[1]],
        #     color='white',
        #     s=18,
        #     zorder=4,
        # )

    def plot_metric(self, ax, KX, KY):
        g_xx, g_xy, g_yy, KXX, KYY = self.quantum_metric(KX, KY)

        # rebuild matching extent
        extent = [KXX.min(), KXX.max(), KYY.min(), KYY.max()]

        lw = 0.6

        im00 = ax[0, 0].imshow(g_xx, origin='lower', extent=extent, cmap='Reds')
        ax[0, 0].set_title(r"$g_{xx}$")
        # self.plot_hexagonal_bz(ax[0, 0], lw=lw)
        plt.colorbar(im00, ax=ax[0, 0])

        im01 = ax[0, 1].imshow(g_xy, origin='lower', extent=extent, cmap='RdBu_r')
        ax[0, 1].set_title(r"$g_{xy}$")
        # self.plot_hexagonal_bz(ax[0, 1], lw=lw)
        plt.colorbar(im01, ax=ax[0, 1])

        im10 = ax[1, 0].imshow(g_xy, origin='lower', extent=extent, cmap='RdBu_r')
        ax[1, 0].set_title(r"$g_{yx}$")
        # self.plot_hexagonal_bz(ax[1, 0], lw=lw)
        plt.colorbar(im10, ax=ax[1, 0])

        im11 = ax[1, 1].imshow(g_yy, origin='lower', extent=extent, cmap='Reds')
        ax[1, 1].set_title(r"$g_{yy}$")
        # self.plot_hexagonal_bz(ax[1, 1], lw=lw)
        plt.colorbar(im11, ax=ax[1, 1])

    def plot_trace_metric(self, ax, KX, KY):
        g_xx, g_xy, g_yy, KXX, KYY = self.quantum_metric(KX, KY)
        
        trace = g_xx + g_yy
        det = g_xx * g_yy - g_xy * g_xy
        extent = [KXX.min(), KXX.max(), KYY.min(), KYY.max()]

        lw = 0.6

        im00 = ax[0].imshow(trace, origin='lower', extent=extent, cmap='Reds')
        ax[0].set_title(r"$\text{Tr}(g(\mathbf{k}))$")
        self.plot_hexagonal_bz(ax[0], lw=lw)
        plt.colorbar(im00, ax=ax[0])

        im01 = ax[1].imshow(2 * np.sqrt(det), origin='lower', extent=extent, cmap='Reds')
        ax[1].set_title(r"$2\sqrt{\text{det}(g(\mathbf{k}))}$")
        self.plot_hexagonal_bz(ax[1], lw=lw)
        plt.colorbar(im01, ax=ax[1])