import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
import utils


# Physical constants
HBAR2_M = utils.HBAR2_M                 # ℏ² / m  [meV · Å²]
MSTAR   = utils.MSTAR                   # Effective mass  [units of electron mass mₑ]
E_SQUARED = utils.CHARGE_SQUARED        # Coulomb's constant  [eV · Å]
DK = utils.DK                           # Infinitesimal k-step for derivatives [Å⁻¹]

# Enable LaTeX rendering for axis labels and annotations
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"  # Optional for advanced math
})


class Moire:
    """
    Moiré superlattice class for a twisted heterobilayer.

    Models a moiré pattern formed by two hexagonal monolayers
    rotated by a small twist angle θ. Computes the moiré lattice,
    Hamiltonian, band structure in a plane-wave basis, and related 
    quantum geometric quantities.

    Parameters
    ----------
    a0       : float  Monolayer lattice constant [Å]. Default 3.32 Å (WSe₂).
    theta_deg: float  Twist angle between layers [degrees]. Default 2.0°.
    V        : float  Moiré potential amplitude [meV]. Default 6.6 meV.
    psi_deg  : float  Moiré potential phase [degrees]. Default −94°.
    n_shells : int    Number of reciprocal-lattice shells to include
                      in the plane-wave basis. Default 6.
    """

    def __init__(self, a0=3.32, theta_deg=2.0, V=6.6, psi_deg=-94.0, n_shells=6):
        self.a0 = a0
        self.theta_deg = theta_deg
        self.V = V
        self.psi_deg = psi_deg
        self.n_shells = n_shells

    # ------------------------------------------------------------------
    # Angles
    # ------------------------------------------------------------------

    @property
    def theta(self) -> float:
        """Twist angle in radians."""
        return np.radians(self.theta_deg)
    
    @property
    def psi(self) -> float:
        """Moiré potential phase in radians."""
        return np.radians(self.psi_deg)
    
    # ------------------------------------------------------------------
    # Lattice geometry
    # ------------------------------------------------------------------

    @property
    def aM(self) -> float:
        """
        Moiré superlattice constant [Å].

            aM = a0 / (2 sin(θ/2))
        """
        return self.a0 / (2 * np.sin(self.theta / 2))

    @property
    def a1(self) -> np.ndarray:
        """
        First moiré real-space lattice vector [Å].

        For a hexagonal lattice:
            a1 = aM · (√3/2,  1/2)
        """
        return np.array([np.sqrt(3) * self.aM / 2, self.aM / 2])
    
    @property
    def a2(self) -> np.ndarray:
        """
        Second moiré real-space lattice vector [Å].

        For a hexagonal lattice:
            a2 = aM · (0, 1)
        """
        return np.array([0, self.aM])
    
    @property
    def A_unit_cell(self) -> float:
        """
        Area of the moiré unit cell [Å²].

        For a hexagonal lattice:
            A_unit_cell = a1 × a2 = |a1| * |a2| * sin(120°) = (√3/2) * aM²
        """
        return np.linalg.norm(np.cross(self.a1, self.a2))

    @property
    def b1(self) -> np.ndarray:
        """
        First moiré reciprocal-space lattice vector [Å⁻¹].

        For a hexagonal lattice:
            b1 = (4π / (√3 * aM), 0)
        """
        return utils.ADIMB1 / self.aM

    @property
    def b2(self) -> np.ndarray:
        """
        Second moiré reciprocal-space lattice vector [Å⁻¹].

        For a hexagonal lattice:
            b2 = R120 * b1
        """
        phi = np.radians(120)
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        return R @ self.b1
    
    @property
    def b3(self) -> np.ndarray:
        """
        Third moiré reciprocal-space lattice vector [Å⁻¹].

        For a hexagonal lattice:
            b3 = R120 * b2
        """
        phi = np.radians(120)
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        return R @ self.b2
    
    # ------------------------------------------------------------------
    # High-symmetry points in the Brillouin zone (BZ)
    # ------------------------------------------------------------------
    
    @property
    def gamma(self) -> np.ndarray:
        """
        Gamma point in the Brillouin zone (BZ) [Å⁻¹].
        
            γ = (0, 0)
        """
        return np.zeros(2)

    @property
    def kappa_pp(self) -> np.ndarray:
        """
        Kappa'' point in the Brillouin zone (BZ) [Å⁻¹].

            κ'' = (2b₁ + b₂) / 3
        """
        kappa_pp = ( 2 * self.b1 + self.b2 ) / 3
        return kappa_pp
    
    @property
    def kappa_p(self) -> np.ndarray:
        """
        Kappa' point in the Brillouin zone (BZ) [Å⁻¹].

            κ' = R60 * κ''
        """
        phi = np.radians(60)
        c, s = np.cos(phi), np.sin(phi)
        R1 = np.array([[c, -s], [s, c]])
        kappa_p = R1 @ self.kappa_pp
        return kappa_p
    
    @property
    def kappa(self) -> np.ndarray:
        """
        Kappa point in the Brillouin zone (BZ) [Å⁻¹].

            κ = R180 * κ'
        """
        phi = np.radians(180)
        c, s = np.cos(phi), np.sin(phi)
        R2 = np.array([[c, -s], [s, c]])
        kappa = R2 @ self.kappa_p
        return kappa
    
    @property
    def A_BZ(self) -> float:
        """
        Area of the BZ [Å⁻²].

        For a hexagonal BZ:
            A_BZ = 3 * √3 / 2 * |kappa|²
        """
        return ( 3 * np.sqrt(3) / 2 ) * np.linalg.norm(self.kappa)**2
    
    @property
    def default_path(self) -> tuple[list, list]:
        """
        Returns a default path through the Brillouin zone (BZ) for band structure calculations.
        
        Default path:
            γ → κ' → κ'' → κ.
        """
        path = [self.gamma, self.kappa_p, self.kappa_pp, self.kappa]
        labels = [r'$\gamma$', r'$\kappa$', r"$\kappa''$", r'$\kappa$']
        return path, labels
    
    @property
    def BZ_path(self) -> mplPath:
        """
        Returns a Matplotlib Path object of the BZ defined by its corners.
        """
        phi = np.radians(60)
        c, s = np.cos(phi), np.sin(phi)
        rot60 = np.array([[c, -s], [s, c]])

        v0 = self.kappa_pp
        verts = [v0]
        v = v0.copy()
        for _ in range(5):
            v = rot60 @ v
            verts.append(v)
        
        return mplPath( np.array(verts) )
    
    @property
    def basis(self) -> np.ndarray[tuple[np.ndarray, tuple[int, int]]]:
        """
        Dimensionless plane-wave basis vectors for the Hamiltonian.
        Each row is a tuple (g_vector, coordinates (n1, n2) in reciprocal space).
        """
        return utils.compute_basis_adim(self.n_shells)

    def _V_fourier(self, dm, dn) -> complex:
        """
        Fourier component of the moiré potential for a given
        reciprocal-lattice displacement (dm, dn).

        Non-zero only for the six nearest reciprocal lattice vectors:
            V · exp(+iψ)   for (dm, dn) ∈ {(1, 0),(0, 1),(-1, -1)}
            V · exp(−iψ)   for (dm, dn) ∈ {(-1, 0),(0, -1),(1, 1)}

        Parameters
        ----------
        dm, dn : int  Reciprocal-lattice index displacement.

        Returns
        -------
        complex  Fourier coefficient; 0.0 if not a nearest neighbor.
        """
        if (dm, dn) in ((1, 0), (0, 1), (-1, -1)):
            return self.V * np.exp( 1j * self.psi)
        if (dm, dn) in ((-1, 0), (0, -1), (1, 1)):
            return self.V * np.exp(-1j * self.psi)
        return 0.0
    
    def _build_offdiag(self):
        N = len(self.basis)
        V = np.zeros((N, N), dtype=complex)
        for i, (_, n) in enumerate(self.basis):
            for j in range(i + 1, N):
                _, n2 = self.basis[j]
                dm, dn = n2[0] - n[0], n2[1] - n[1]
                potential = self._V_fourier(dm, dn)
                if np.abs(potential) > 1e-10:
                    V[i, j] = potential
                    V[j, i] = np.conj(potential)
        return V
    
    def _build_diag(self, kpath):
        """
        Vectorized diagonal blocks for all k-points at once.

        Returns
        -------
        diags : np.ndarray, shape (n_k, N)
        """
        # g_vecs: (N, 2)
        g_vecs = np.array([g for g, _ in self.basis]) / self.aM  # (N, 2)

        # kpath: (n_k, 2), g_vecs: (N, 2)
        # k_plus_g[i, j] = kpath[i] + g_vecs[j]  →  shape (n_k, N, 2)
        k_plus_g = kpath[:, None, :] + g_vecs[None, :, :]

        # squared norms → shape (n_k, N)
        k2 = np.einsum('ijk,ijk->ij', k_plus_g, k_plus_g)

        return -HBAR2_M * k2 / (2 * MSTAR)
    
    def H_k_all(self, kpath):
        """
        Build H(k) for all k-points at once.

        Returns
        -------
        H_all : np.ndarray, shape (n_k, N, N)
        """
        n_k = len(kpath)
        N = len(self.basis)

        diags = self._build_diag(kpath)           # (n_k, N)
        V = self._build_offdiag()                 

        # Broadcast V across all k-points, then fill diagonal per k
        H_all = np.broadcast_to(V, (n_k, N, N)).copy()  # must copy — we'll modify diagonals
        i_idx = np.arange(N)
        H_all[:, i_idx, i_idx] = diags            # set diagonal for each k

        return H_all


    def H_k(self, k):
        """
        Construct the moiré Hamiltonian in the plane-wave basis at k.

        Diagonal entries are the kinetic energies of each shifted plane wave:
            H[i, i] = -ℏ²|k + g_i|² / (2 m*)

        Off-diagonal entries are the moiré potential Fourier components:
            H[i, j] = V_fourier(g_j - g_i),  H[j, i] = conj(H[i,j])

        Parameters
        ----------
        k : array_like, shape (2,)  Crystal momentum [Å⁻¹].

        Returns
        -------
        np.ndarray, shape (N, N), dtype complex Hermitian Hamiltonian matrix.
        """
        N = len(self.basis)
        H = np.zeros((N, N), dtype=complex)

        for i, (g, n) in enumerate(self.basis):
            k_plus_g = k + g / self.aM
            H[i, i] = - HBAR2_M * np.linalg.norm(k_plus_g)**2 / (2 * MSTAR)

            for j in range(i + 1, N):
                _, n2 = self.basis[j]
                dm, dn = n2[0] - n[0], n2[1] - n[1]
                potential = self._V_fourier(dm, dn)
                if np.abs(potential) > 1e-10:
                    H[i, j] = potential
                    H[j, i] = np.conj(potential)

        return H
    
    def bloch_u(self, k, r, target_band=1, gauge_fixed=True) -> complex:
        """
        Periodic part of the Bloch function u_k(r) for the top band.

        Expands the top-band eigenvector in the plane-wave basis and
        optionally applies a smooth gauge by requiring the sum of
        coefficients to be real and positive.

        Parameters
        ----------
        k          : array_like, shape (2,)  Crystal momentum [Å⁻¹].
        r          : array_like, shape (2,)  Real-space position [Å].
        target_band: int  Band index to extract (1 = top band, 2 = second band, etc.). Default is 1 (top band).
        gauge_fixed: bool  If True, apply gauge phase correction. Default is True.

        Returns
        -------
        complex  Value of u_k(r), normalized by √(A_unit_cell).
        """
        _, eigenvectors = np.linalg.eigh(self.H_k(k))
        states = eigenvectors[:, -target_band]  # take specified band state

        zk = states[:, 0]  # specified band state
        bloch0 = 0.0

        for m, (g, _) in enumerate(self.basis):
            bloch0 += zk[m] * np.exp(1.0j * (g / self.aM) @ r)

        if gauge_fixed:
            overlap = np.sum(np.conj(zk))
            bloch_periodic = bloch0 * np.exp(1.0j * np.angle(overlap))
        else:
            bloch_periodic = bloch0

        return bloch_periodic / np.sqrt(self.A_unit_cell)
    
    def make_k_path(self, path, labels, n_pts):
        """
        Construct a discretized k-path through a list of high-symmetry points.

        Points are distributed across segments proportional to their length
        so the k-axis spacing is uniform in reciprocal space.

        Parameters
        ----------
        path   : list of np.ndarray  High-symmetry k-points [Å⁻¹].
        labels : list of str         LaTeX labels for each point.
        n_pts  : int                 Total number of k-points along the path.

        Returns
        -------
        kpath  : np.ndarray, shape (n_pts, 2)  Ordered k-points [Å⁻¹].
        kdist  : np.ndarray, shape (n_pts,)    Cumulative arc length [Å⁻¹].
        ticks  : list of (float, str)          (position, label) pairs for axis ticks.
        """
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
        """
        Diagonalize the Hamiltonian along a k-path.

        Parameters
        ----------
        kpath : np.ndarray, shape (n_pts, 2)  k-points [Å⁻¹].

        Returns
        -------
        bands  : np.ndarray, shape (n_pts, N)     Eigenvalues [meV], descending order.
        states : np.ndarray, shape (n_pts, N, N)  Eigenvectors, top N bands.
        """
        H_all = self.H_k_all(kpath)                        # (n_pts, N, N)
        eigenvalues, eigenvectors = np.linalg.eigh(H_all)  # batched over n_pts

        bands  = eigenvalues[..., ::-1]   # (n_pts, N)
        states = eigenvectors[..., ::-1]  # (n_pts, N, N)

        return bands, states
    
    def calculate_gap(self, k_point=None):
        """
        Energy gap between the top two bands at a given k-point [meV].

        Parameters
        ----------
        k_point : array_like, shape (2,), optional
            Crystal momentum [Å⁻¹]. Defaults to κ.

        Returns
        -------
        float  E_1 − E_0 at k_point [meV].
        """
        if k_point is None:
            k_point = self.kappa
        
        eigenvalues = np.linalg.eigvalsh(self.H_k(k_point))
        sorted_eigvals = np.sort(eigenvalues)[::-1]  # descending order

        return sorted_eigvals[1] - sorted_eigvals[0]
    
    def calculate_bandwidth(self, target_band=1):
        """
        Bandwidth of the top band between γ and κ [meV].

            W = E_top(γ) − E_top(κ)

        Returns
        -------
        float  Bandwidth [meV].
        """
        eigenvalues_gamma = np.linalg.eigvalsh(self.H_k(self.gamma))
        sorted_eigvals_gamma = np.sort(eigenvalues_gamma)[::-1]  # descending order

        eigenvalues_kappa = np.linalg.eigvalsh(self.H_k(self.kappa))
        sorted_eigvals_kappa = np.sort(eigenvalues_kappa)[::-1]  # descending order

        return sorted_eigvals_gamma[target_band-1] - sorted_eigvals_kappa[target_band-1]
    
    def moire_potential(self, r):
        """
        Real-space moiré potential at position r [meV].

        Sum of three cosine terms:
            V(r) = 2V Σ_{j=1,2,3} cos(b_j · r + ψ)

        Parameters
        ----------
        r : array_like, shape (2,)  Real-space position [Å].

        Returns
        -------
        float  Potential value [meV].
        """
        return 2 * self.V * (np.cos(self.b1 @ r + self.psi) + np.cos(self.b2 @ r + self.psi) + np.cos(self.b3 @ r + self.psi))
    
    def moire_potential_XY(self, X, Y):
        """
        Real-space moiré potential on a grid.

        Sum of three cosine terms:
            V(r) = 2V Σ_{j=1,2,3} cos(b_j · r + ψ)

        Parameters
        ----------
        X : array_like, shape (Ny, Nx)  X-coordinates [Å].
        Y : array_like, shape (Ny, Nx)  Y-coordinates [Å].

        Returns
        -------
        float  Potential value [meV].
        """
        return 2 * self.V * (np.cos(self.b1[0] * X + self.b1[1] * Y + self.psi) + np.cos(self.b2[0] * X + self.b2[1] * Y + self.psi) + np.cos(self.b3[0] * X + self.b3[1] * Y + self.psi))
    
    def mesh_k_unit_cell(self, Nx=10, Ny=10):
        """
        Sample one parallelogram unit cell of the reciprocal lattice.
        i/Nx and j/Ny run over [0, 1), so no double-counting at boundaries.
        """
        i = np.arange(0, Nx)
        j = np.arange(0, Ny)
        ii, jj = np.meshgrid(i, j, indexing='ij')
        k_points = (ii[..., None] * self.b1 / Nx) + (jj[..., None] * self.b2 / Ny)
        return k_points.reshape(-1, k_points.shape[-1])

    def wannier_wavefunction_old(self, x, y, k_points, gauge_fixed=True, R0=np.zeros(2)):
        """
        This is an old function. New faster one is wannier_wavefunction.
        """
        x, y = np.asarray(x), np.asarray(y)

        # --- Batched diagonalization (still fully vectorized) ---
        H_all = self.H_k_all(k_points)              # (n_k, N, N)
        _, eigenvectors = np.linalg.eigh(H_all)     # (n_k, N, N)
        zk = eigenvectors[:, :, -1]                 # (n_k, N)

        g_vecs = np.array([g for g, _ in self.basis]) / self.aM  # (N, 2)
        wannier = np.zeros_like(x, dtype=complex)

        for i, k in enumerate(k_points):
            k_plus_g = k + g_vecs                   # (N, 2)

            # Bloch sum over basis: (N,) broadcast against spatial grid
            phases = np.exp(1.0j * (k_plus_g[:, 0, None, None] * x
                                + k_plus_g[:, 1, None, None] * y))  # (N, Ny, Nx)
            bloch = np.einsum('m,mij->ij', zk[i], phases)              # (Ny, Nx)

            shift = np.exp(-1.0j * (k @ R0))

            if gauge_fixed:
                gauge = np.exp(1.0j * np.angle(np.sum(np.conj(zk[i]))))
                wannier += bloch * shift * gauge
            else:
                wannier += bloch * shift

        return wannier / (k_points.shape[0] * np.sqrt(self.A_unit_cell))
    
    def wannier_wavefunction(self, x, y, k_points, gauge_fixed=True, R0=np.zeros(2)):
        orig_shape = x.shape
        xf, yf = x.ravel(), y.ravel()          # (Npts,)

        H_all = self.H_k_all(k_points)          # (n_k, N, N)
        _, eigenvectors = np.linalg.eigh(H_all)
        zk = np.conj(eigenvectors[:, :, -1])             # (n_k, N)

        g_vecs = np.array([g for g, _ in self.basis]) / self.aM   # (N, 2)

        # exp(i g . r): independent of k, computed once -> (N, Npts)
        g_phase = np.exp(1j * (g_vecs[:, 0, None] * xf[None, :]
                            + g_vecs[:, 1, None] * yf[None, :]))

        u_k = zk @ g_phase                       # (n_k, Npts)

        # exp(i k . r)  -> (n_k, Npts)
        k_phase = np.exp(1j * (k_points[:, 0, None] * xf[None, :]
                            + k_points[:, 1, None] * yf[None, :]))

        bloch = u_k * k_phase                    # (n_k, Npts)

        shift = np.exp(-1j * (k_points @ R0))    # (n_k,)

        if gauge_fixed:
            gauge = np.exp(1j * np.angle(np.sum(np.conj(zk), axis=1)))  # (n_k,)
            coeff = shift * gauge
        else:
            coeff = shift

        wannier = coeff @ bloch                  # (n_k,) @ (n_k, Npts) -> (Npts,)
        wannier = wannier.reshape(orig_shape)

        return wannier / (k_points.shape[0] * np.sqrt(self.A_unit_cell))


    def wannier_wavefunction_2band(self, x, y, k_points,
                         trial_centers,               # [[Rax,Ray],[Rbx,Rby]] -> R_A, R_B
                         band_indices=(-2, -1)):
        """
        Two-band projective Wannier functions
        Returns array (n_w, *x.shape), one Wannier function per trial center.
        """
        x, y = np.asarray(x), np.asarray(y)
        orig_shape = x.shape
        xf, yf = x.ravel(), y.ravel()                         # (Npts,)

        trial_centers = np.asarray(trial_centers)             # (n_w, 2)
        n_w = trial_centers.shape[0]
        n_k = k_points.shape[0]
        bands = list(band_indices)

        H_all = self.H_k_all(k_points)                        # (n_k, N, N)
        _, eigenvectors = np.linalg.eigh(H_all)
        zk = eigenvectors[:, :, bands]                        # (n_k, N, n_w)  [k, G, alpha]

        g_vecs = np.array([g for g, _ in self.basis]) / self.aM   # (N, 2)
        kG = k_points[:, None, :] + g_vecs[None, :, :]        # (n_k, N, 2)  (k+G)

        tphase = np.exp(-1j * np.einsum('kGd,id->kGi', kG, trial_centers))  # (n_k, N, n_w)
        O = np.einsum('kGa,kGi->kai', zk, tphase)             # (n_k, n_w, n_w)  [k, alpha, i]

        C = np.empty_like(O)                                  # coeff of psi_alpha in psi_tilde_i
        smin = np.empty(n_k)
        for kk in range(n_k):
            S = O[kk].conj().T @ O[kk]                        # S_{ij} = sum_a O*_{a i} O_{a j}
            d, V = np.linalg.eigh(S)                          # S = V diag(d) V^dag
            smin[kk] = np.sqrt(max(d.min(), 0.0))
            d = np.maximum(d, 1e-12)                          # roundoff guard
            S_inv_sqrt = V @ np.diag(1/np.sqrt(d)) @ V.conj().T
            C[kk] = O[kk] @ S_inv_sqrt                        # (alpha, i)
        self._last_smin = smin.min()                          # conditioning monitor

        ztilde = np.einsum('kai,kGa->kGi', C, np.conj(zk))    # (n_k, N, n_w)

        g_phase = np.exp(1j * (g_vecs[:, 0, None] * xf[None, :]
                            + g_vecs[:, 1, None] * yf[None, :]))   # (N, Npts)
        k_phase = np.exp(1j * (k_points[:, 0, None] * xf[None, :]
                            + k_points[:, 1, None] * yf[None, :]))  # (n_k, Npts)

        u = np.einsum('kGi,GP->kiP', ztilde, g_phase)         # (n_k, n_w, Npts)
        bloch = u * k_phase[:, None, :]                       # (n_k, n_w, Npts)
        wannier = bloch.sum(axis=0) / (np.sqrt(n_k) * np.sqrt(self.A_unit_cell))

        return wannier.reshape((n_w,) + orig_shape)           # (n_w, *orig_shape)

    
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
    
    # ------------------------------------------------------------------
    # Single isolated band
    # ------------------------------------------------------------------

    def dual_states(self, k_points):
        dk = utils.DK
        shifts = np.array([[dk, 0], [-dk, 0], [0, dk], [0, -dk]])  # (4, 2)

        k_shifted = k_points[None, :, :] + shifts[:, None, :]      # (4, N_points, 2)
        k_all = np.vstack([k_points, k_shifted.reshape(-1, 2)])    # (5*N_points, 2)

        H_all = self.H_k_all(k_all)
        _, eigvecs = np.linalg.eigh(H_all)
        u_all = eigvecs[:, :, -1]                                   # (5*N_points, N)

        N = len(k_points)
        u, u_xp, u_xm, u_yp, u_ym = (
            u_all[0*N:1*N], u_all[1*N:2*N], u_all[2*N:3*N],
            u_all[3*N:4*N], u_all[4*N:5*N],
        )                                                            # each (N_points, N)

        Sxp = np.sum(np.conj(u) * u_xp, axis=-1)
        Sxm = np.sum(np.conj(u) * u_xm, axis=-1)
        Syp = np.sum(np.conj(u) * u_yp, axis=-1)
        Sym = np.sum(np.conj(u) * u_ym, axis=-1)

        dual_xp = u_xp / Sxp[..., None]
        dual_xm = u_xm / Sxm[..., None]
        dual_yp = u_yp / Syp[..., None]
        dual_ym = u_ym / Sym[..., None]

        return dual_xp, dual_xm, dual_yp, dual_ym

    def covariant_derivatives(self, k_points):
        dual_xp, dual_xm, dual_yp, dual_ym = self.dual_states(k_points)

        dX = (dual_xp - dual_xm) / (2 * utils.DK)
        dY = (dual_yp - dual_ym) / (2 * utils.DK)

        return dX, dY

    def berry_curvature(self, k_points):
        dX, dY = self.covariant_derivatives(k_points)
        overlap = np.sum(np.conj(dX) * dY, axis=-1)
        berry = -2 * np.imag(overlap)

        return berry
    
    def quantum_metric(self, k_points):
        dX, dY = self.covariant_derivatives(k_points)

        g_xx = np.real(np.sum(np.conj(dX) * dX, axis=-1))
        g_yy = np.real(np.sum(np.conj(dY) * dY, axis=-1))

        g_xy = np.real(np.sum(np.conj(dX) * dY, axis=-1))

        return g_xx, g_xy, g_yy
    
    def integrate_trace(self, Nx, Ny):
        k_points, dkx, dky, jac = self.mesh_k_hex(Nx=Nx, Ny=Ny)
        k_points_bz = self.map_to_first_BZ(k_points)

        g_xx, _, g_yy = self.quantum_metric(k_points_bz)
        trace = g_xx + g_yy

        # Find which indices in k_points match points in k_points_bz
        # k_points: (N, 2), k_points_bz: (M, 2), M < N
        # diffs = np.linalg.norm(k_points[:, None, :] - k_points_bz[None, :, :], axis=-1)  # (N, M)
        # mask = np.any(diffs < 1e-10, axis=1)  # (N,) True if k_point matches any BZ point

        # trace_masked = np.where(mask, trace, 0.0)
        
        dA = dkx * dky * jac
        integral = (1 / self.A_BZ) * np.sum(trace) * dA

        return integral
    
    def map_to_first_BZ(self, k_points, shells=1):
        """
        Map k-points to the first Brillouin Zone by finding the nearest
        reciprocal lattice point.

        Parameters
        ----------
        k_points : (N, 2) array
        b1, b2   : (2,) reciprocal lattice vectors
        shells   : int, number of shells of G-vectors to search (1 is
                   sufficient for hexagonal; increase if you see artefacts)
        """
        B = np.column_stack([self.b1, self.b2])           # (2,2) matrix, columns are b1, b2

        B_inv = np.linalg.inv(B)
        frac = k_points @ B_inv.T              # fractional coordinates, (N, 2)
        frac_reduced = frac - np.floor(frac + 0.5)
        k_reduced = frac_reduced @ B.T         # back to Cartesian, (N, 2)

        n_range = np.arange(-shells, shells + 1)
        ns = np.array(np.meshgrid(n_range, n_range)).T.reshape(-1, 2)  # ((2s+1)^2, 2)
        G_vectors = ns @ B.T                   # ((2s+1)^2, 2) in Cartesian

        diff = k_reduced[:, None, :] - G_vectors[None, :, :]  # (N, M, 2)
        norms = np.linalg.norm(diff, axis=-1)                 # (N, M)
        best = np.argmin(norms, axis=-1)                      # (N,)
        k_first_BZ = k_reduced - G_vectors[best]              # (N, 2)

        return k_first_BZ
    
    def mesh_k_hex(self, Nx=10, Ny=10):
        i = np.arange(0, Nx)
        j = np.arange(0, Ny)
        dkx = np.linalg.norm(self.b1) / Nx
        dky = np.linalg.norm(self.b2) / Ny
        jac = np.sqrt(3) / 2                # area factor for triangular mesh

        ii, jj = np.meshgrid(i, j, indexing='ij')
        k_points = (ii[..., None] * self.b1 / (Nx)) + (jj[..., None] * self.b2 / (Ny))
        
        return k_points.reshape(-1, k_points.shape[-1]), dkx, dky, jac
    
    def hopping_integral(self, X, Y, k_points, order=1):
        if order == 1:
            R0 = self.a2
        elif order == 2:
            R0 = self.a2 + self.a1
        elif order == 3:
            R0 = 2 * self.a2

        wannier0 = self.wannier_wavefunction(X, Y, k_points)
        wannier_xp = self.wannier_wavefunction(X + utils.DX, Y, k_points)
        wannier_xm = self.wannier_wavefunction(X - utils.DX, Y, k_points)
        wannier_yp = self.wannier_wavefunction(X, Y + utils.DX, k_points)
        wannier_ym = self.wannier_wavefunction(X, Y - utils.DX, k_points)
        wannier2 = self.wannier_wavefunction(X, Y, k_points, R0=R0)
        
        laplacian = (wannier_xp + wannier_xm + wannier_yp + wannier_ym - 4 * wannier0) / (utils.DX**2)

        integralT = 0.5 * (HBAR2_M / MSTAR) * np.trapz(
            np.trapz(np.conj(wannier2) * laplacian, X[0,:], axis=1),
            Y[:,0],
            axis=0
        )

        # integralV = np.trapz(
        #     np.trapz(np.conj(wannier2) * self.moire_potential_XY(X, Y) * wannier0, X[0,:], axis=1),
        #     Y[:,0],
        #     axis=0
        # )

        integral = integralT

        return integral

    def coulomb_potential(self, r, epsilon, soft, D=1e6):
        """
        Coulomb potential between two points in 2D.

        Parameters
        ----------
        r : array_like, shape (2,)  Real-space positions [Å].
        epsilon : float  relative permitivity.
        soft : float, optional  Softening parameter [Å]. Default is 1e-6.

        Returns
        -------
        float  Coulomb potential [meV].
        """
        V0 = E_SQUARED / epsilon
        
        if D > 1e5:
            radial = 1 / np.sqrt(r**2 + soft**2)
        else:
            radial = 1 / np.sqrt(r**2 + soft**2) - 1 / np.sqrt(r**2 + D**2)

        return V0 * radial
    
    def nearest_neighbour_interaction(self, X, Y, k_points, order=1, epsilon=1.0, D=1e6):
        if order == 0:
            R0 = np.zeros(2)
        elif order == 1:
            R0 = self.a2
        elif order == 2:
            R0 = self.a2 + self.a1

        dx_grid = np.abs(X[0, 1] - X[0, 0])
        dy_grid = np.abs(Y[1, 0] - Y[0, 0])
        softness = np.min([dx_grid, dy_grid])/2

        wannier1 = self.wannier_wavefunction(X, Y, k_points)
        wannier2 = self.wannier_wavefunction(X, Y, k_points, R0=R0)

        density1 = np.abs(wannier1)**2
        density2 = np.abs(wannier2)**2
        rho1 = density1.ravel()
        rho2 = density2.ravel()

        R = np.column_stack((X.ravel(), Y.ravel()))

        dr = R[:, None, :] - R[None, :, :]
        dist = np.linalg.norm(dr, axis=-1)

        K = self.coulomb_potential(dist, epsilon, soft=softness, D=D)      # shape (N, N)

        integral = (dx_grid * dy_grid)**2 * np.dot(rho2, K @ rho1)
        
        return integral
    
    def exchange_interaction(self, X, Y, k_points, order=1, epsilon=1.0, D=1e6):
        if order == 0:
            R0 = np.zeros(2)
        elif order == 1:
            R0 = self.a2
        elif order == 2:
            R0 = self.a2 + self.a1

        dx_grid = np.abs(X[0, 1] - X[0, 0])
        dy_grid = np.abs(Y[1, 0] - Y[0, 0])
        softness = np.min([dx_grid, dy_grid])/2

        wannier1 = self.wannier_wavefunction(X, Y, k_points)
        wannier2 = self.wannier_wavefunction(X, Y, k_points, R0=R0)

        A = np.conj(wannier1) * wannier2
        a = A.ravel()

        R = np.column_stack((X.ravel(), Y.ravel()))

        dr = R[:, None, :] - R[None, :, :]
        dist = np.linalg.norm(dr, axis=-1)

        K = self.coulomb_potential(dist, epsilon, soft=softness, D=D)      # shape (N, N)

        integral = (dx_grid * dy_grid)**2 * np.vdot(a, K @ a)
        
        return integral

    
    # ------------------------------------------------------------------
    # Plotting functions
    # ------------------------------------------------------------------
    
    def plot_bands(self, ax, bands, kdist, ticks, max_bands=4, highlight=1):
        ax.plot(kdist, bands[:, 0:highlight], color='#0024e6', lw=1.5)
        ax.plot(kdist, bands[:, highlight:max_bands], color='k', lw=1.5)

        tick_positions = [t[0] for t in ticks]
        tick_labels    = [t[1] for t in ticks]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=18)
        for x in tick_positions:
            ax.axvline(x, color='k', lw=0.5, zorder=0)

        ax.set_xlim(kdist[0], kdist[-1])
        ax.set_ylabel(r"$E$ [meV]", fontsize=18)

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
        r1 = self.a1 #np.array([0, 1.0]) * self.aM
        r2 = self.a2 #np.array([np.sqrt(3) / 2.0, 1.0 / 2.0]) * self.aM
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
                    ax.plot([x0, x1], [y0, y1], color='#33221E', ls='--', lw=0.8, alpha=0.9)
        
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    def plot_k_mesh(self, ax, Nx, Ny):
        k_points, _, _, _ = self.mesh_k_hex(Nx=Nx, Ny=Ny)
        k_points_bz = self.map_to_first_BZ(k_points)
        ax.scatter(k_points_bz[:, 0], k_points_bz[:, 1], color='#FF216B', s=30, alpha=0.9, edgecolors='k', linewidth=1.0)
        self.plot_hexagonal_bz(ax)

    def plot_wavefunction(self, ax, wavefunction, extent, cmap='RdBu_r', norm=None):
        im = ax.imshow(np.abs(wavefunction)**2, extent=extent, origin='lower', 
                    interpolation='bilinear', cmap=cmap, norm=norm)
        return im
    
    def plot_wavefunction_phase(self, ax, wavefunction, extent, cmap='RdBu_r'):
        im = ax.imshow(np.abs(np.angle(wavefunction)), extent=extent, origin='lower', 
                    interpolation='bilinear', cmap=cmap)
        return im
    
    def plot_wannier_function(self, ax, X, Y, k_points, R0=np.zeros(2), norm=None, cmap='RdBu_r'):
        wannier = self.wannier_wavefunction(X, Y, k_points, R0=R0, gauge_fixed=True)
        extent = [X.min(), X.max(), Y.min(), Y.max()]
        return self.plot_wavefunction(ax, wannier, extent, norm=norm, cmap=cmap)
    
    def plot_wannier_function2(self, ax, X, Y, k_points, R0=np.zeros(2), norm=None, cmap='RdBu_r'):
        return 0
    
    def plot_berry_curvature(self, ax, KX, KY, abs_val=False, mask_bz=True):
        k_points = np.column_stack([KX.ravel(), KY.ravel()])
        berry = self.berry_curvature(k_points).reshape(KX.shape)
        if abs_val:
            berry = np.abs(berry)
            cmap = 'Reds'
        else:
            cmap = 'RdBu_r'

        # rebuild matching extent
        extent = [KX.min(), KX.max(), KY.min(), KY.max()]

        if mask_bz:
            dk = KX[0, 1] - KX[0, 0]
            mask = self.BZ_path.contains_points(k_points, radius=dk)              # Test whether points are inside
            mask = mask.reshape(KX.shape)                                           # Reshape back to grid shape
        else:
            mask = np.ones_like(berry, dtype=bool)

        berry_masked = np.where(mask, berry, 0.0)

        self.plot_hexagonal_bz(ax)
        im = ax.imshow(berry_masked, extent=extent, origin='lower', 
                    interpolation='none', cmap=cmap)
        ax.set_title(r"$\Omega(\mathbf{k})$")
        plt.colorbar(im, ax=ax)
        
        return im
    
    def plot_hexagonal_bz(self, ax, lw=1.5):
        verts = self.BZ_path.vertices          
        corners = np.vstack([verts, verts[0]]) # close the loop for plotting
        ax.plot(corners[:, 0], corners[:, 1], color='k', lw=lw, zorder=3)
        ax.text((self.kappa[0] + self.kappa_pp[0]) / 2, self.kappa[1], "1BZ", fontsize=12, color='k')

    def plot_metric(self, ax, KX, KY, mask_bz=True):
        k_points = np.column_stack([KX.ravel(), KY.ravel()])
        g_xx, g_xy, g_yy = self.quantum_metric(k_points)
        g_xx = g_xx.reshape(KX.shape)
        g_xy = g_xy.reshape(KX.shape)
        g_yy = g_yy.reshape(KX.shape)

        if mask_bz:
            dk = KX[0, 1] - KX[0, 0]
            mask = self.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
            mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
        else:
            mask = np.ones_like(g_xx, dtype=bool)

        g_xx_masked = np.where(mask, g_xx, 0.0)
        g_xy_masked = np.where(mask, g_xy, 0.0)
        g_yx_masked = g_xy_masked
        g_yy_masked = np.where(mask, g_yy, 0.0)

        # rebuild matching extent
        extent = [KX.min(), KX.max(), KY.min(), KY.max()]

        lw = 0.6

        im00 = ax[0, 0].imshow(g_xx_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
        ax[0, 0].set_title(r"$g_{xx}$")
        ax[0, 0].set_xlabel(r"$k_x$")
        ax[0, 0].set_ylabel(r"$k_y$")
        self.plot_hexagonal_bz(ax[0, 0], lw=lw)
        plt.colorbar(im00, ax=ax[0, 0])

        im01 = ax[0, 1].imshow(g_xy_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
        ax[0, 1].set_title(r"$g_{xy}$")
        ax[0, 1].set_xlabel(r"$k_x$")
        ax[0, 1].set_ylabel(r"$k_y$")
        self.plot_hexagonal_bz(ax[0, 1], lw=lw)
        plt.colorbar(im01, ax=ax[0, 1])

        im10 = ax[1, 0].imshow(g_yx_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
        ax[1, 0].set_title(r"$g_{yx}$")
        ax[1, 0].set_xlabel(r"$k_x$")
        ax[1, 0].set_ylabel(r"$k_y$")
        self.plot_hexagonal_bz(ax[1, 0], lw=lw)
        plt.colorbar(im10, ax=ax[1, 0])

        im11 = ax[1, 1].imshow(g_yy_masked, origin='lower', extent=extent, cmap='RdBu_r', interpolation='none')
        ax[1, 1].set_title(r"$g_{yy}$")
        ax[1, 1].set_xlabel(r"$k_x$")
        ax[1, 1].set_ylabel(r"$k_y$")
        self.plot_hexagonal_bz(ax[1, 1], lw=lw)
        plt.colorbar(im11, ax=ax[1, 1])

    def plot_trace_metric(self, ax, KX, KY, mask_bz=True):
        k_points = np.column_stack([KX.ravel(), KY.ravel()])
        g_xx, _, g_yy = self.quantum_metric(k_points)
        g_xx = g_xx.reshape(KX.shape)
        g_yy = g_yy.reshape(KX.shape)

        if mask_bz:
            dk = KX[0, 1] - KX[0, 0]
            mask = self.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
            mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
        else:
            mask = np.ones_like(g_xx, dtype=bool)

        g_xx_masked = np.where(mask, g_xx, 0.0)
        g_yy_masked = np.where(mask, g_yy, 0.0)

        trace = g_xx_masked + g_yy_masked
        extent = [KX.min(), KX.max(), KY.min(), KY.max()]

        lw = 0.6

        im00 = ax.imshow(trace, origin='lower', extent=extent, cmap='Reds')
        ax.set_title(r"$\text{Tr}(g(\mathbf{k}))$")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        self.plot_hexagonal_bz(ax, lw=lw)
        plt.colorbar(im00, ax=ax)

    def plot_det_metric(self, ax, KX, KY, mask_bz=True):
        k_points = np.column_stack([KX.ravel(), KY.ravel()])
        g_xx, g_xy, g_yy = self.quantum_metric(k_points)
        g_xx = g_xx.reshape(KX.shape)
        g_xy = g_xy.reshape(KX.shape)
        g_yy = g_yy.reshape(KX.shape)

        if mask_bz:
            dk = KX[0, 1] - KX[0, 0]
            mask = self.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
            mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
        else:
            mask = np.ones_like(g_xx, dtype=bool)

        g_xx_masked = np.where(mask, g_xx, 0.0)
        g_xy_masked = np.where(mask, g_xy, 0.0)
        g_yy_masked = np.where(mask, g_yy, 0.0)

        det = g_xx_masked * g_yy_masked - g_xy_masked * g_xy_masked
        extent = [KX.min(), KX.max(), KY.min(), KY.max()]

        lw = 0.6

        im = ax.imshow(2 * np.sqrt(det), origin='lower', extent=extent, cmap='Reds')
        ax.set_title(r"$2\sqrt{\text{det}(g(\mathbf{k}))}$")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        self.plot_hexagonal_bz(ax, lw=lw)
        plt.colorbar(im, ax=ax)

    def plot_trace_det_metric(self, ax, KX, KY, mask_bz=True):
        k_points = np.column_stack([KX.ravel(), KY.ravel()])
        g_xx, g_xy, g_yy = self.quantum_metric(k_points)
        g_xx = g_xx.reshape(KX.shape)
        g_xy = g_xy.reshape(KX.shape)
        g_yy = g_yy.reshape(KX.shape)

        if mask_bz:
            dk = KX[0, 1] - KX[0, 0]
            mask = self.BZ_path.contains_points(k_points, radius=dk)             # Test whether points are inside
            mask = mask.reshape(KX.shape)                                        # Reshape back to grid shape
        else:
            mask = np.ones_like(g_xx, dtype=bool)

        g_xx_masked = np.where(mask, g_xx, 0.0)
        g_xy_masked = np.where(mask, g_xy, 0.0)
        g_yy_masked = np.where(mask, g_yy, 0.0)

        trace = g_xx_masked + g_yy_masked
        det = g_xx_masked * g_yy_masked - g_xy_masked * g_xy_masked
        extent = [KX.min(), KX.max(), KY.min(), KY.max()]

        lw = 0.6

        im0 = ax[0].imshow(trace, origin='lower', extent=extent, cmap='Reds')
        ax[0].set_title(r"$\text{Tr}(g(\mathbf{k}))$")
        ax[0].set_xlabel(r"$k_x$")
        ax[0].set_ylabel(r"$k_y$")
        self.plot_hexagonal_bz(ax[0], lw=lw)
        plt.colorbar(im0, ax=ax[0])

        im1 = ax[1].imshow(2 * np.sqrt(det), origin='lower', extent=extent, cmap='Reds')
        ax[1].set_title(r"$2\sqrt{\text{det}(g(\mathbf{k}))}$")
        ax[1].set_xlabel(r"$k_x$")
        ax[1].set_ylabel(r"$k_y$")
        self.plot_hexagonal_bz(ax[1], lw=lw)
        plt.colorbar(im1, ax=ax[1])

