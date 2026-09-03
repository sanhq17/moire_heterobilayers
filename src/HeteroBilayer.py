import numpy as np
from numba import njit, prange
from matplotlib.path import Path as mplPath

import utils


@njit(cache=True, parallel=True)
def _coulomb_integral_real(rho_left, rho_right, x, y, v0, soft, dxdy2):
    """Evaluate a real Coulomb bilinear form without materializing its kernel."""
    n = rho_left.size
    row_sums = np.empty(n, dtype=np.float64)

    for i in prange(n):
        total = 0.0
        for j in range(n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r_soft = np.sqrt(dx * dx + dy * dy + soft * soft)
            kernel = 1.0 / r_soft
            total += v0 * kernel * rho_right[j]
        row_sums[i] = total

    return dxdy2 * np.dot(rho_left, row_sums)


@njit(cache=True, parallel=True)
def _coulomb_integral_complex(rho_left, rho_right, x, y, v0, soft, dxdy2):
    """Evaluate a complex Coulomb bilinear form without materializing its kernel."""
    n = rho_left.size
    row_sums = np.empty(n, dtype=np.complex128)

    for i in prange(n):
        total = 0.0j
        for j in range(n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r_soft = np.sqrt(dx * dx + dy * dy + soft * soft)
            kernel = 1.0 / r_soft
            total += v0 * kernel * rho_right[j]
        row_sums[i] = total

    return dxdy2 * np.dot(rho_left, row_sums)

class HeteroBilayer:
    """
    Moiré superlattice class for twisted heterobilayers.

    Models a moiré pattern formed by two hexagonal monolayers
    rotated by twist angle θ. Computes the moiré lattice,
    Hamiltonian, band structure in a plane-wave basis, and related 
    quantum geometric quantities.

    Parameters
    ----------
    a0       : float  Monolayer lattice constant [Å]. Default 3.32 Å (WSe₂).
    theta_deg: float  Twist angle between layers [degrees]. Default 2.5°.
    V        : float  Moiré potential amplitude [meV]. Default 6.6 meV.
    psi_deg  : float  Moiré potential phase [degrees]. Default 26.0°.
    n_shells : int    Number of reciprocal-lattice shells to include
                      in the plane-wave basis. Default 8.
    epsilon  : float  Dielectric constant for Coulomb interaction. Default 10.0.
    """
    # ------------------------------------------------------------------
    # Default parameters for a twisted heterobilayer (e.g., WSe₂/MoSe₂)
    # ------------------------------------------------------------------
    DEFAULT_A0 = 3.32    
    DEFAULT_THETA_DEG = 2.5
    DEFAULT_V = 6.6
    DEFAULT_PSI_DEG = 26.0
    DEFAULT_SHELLS = 10
    DEFAULT_EPSILON = 15.0


    DEFAULT_K_MESH = 21
    DEFAULT_NX = 80
    DEFAULT_NY = 80
    DEFAULT_SCALE = 4.0

    # ------------------------------------------------------------------
    # Triangular lattice parameters
    # ------------------------------------------------------------------
    A1_TRIANGULAR = np.array([np.sqrt(3)/2, 0.5])
    A2_TRIANGULAR = np.array([0, 1.0])
    B1_TRIANGULAR = np.array([4 * np.pi / np.sqrt(3), 0.0])
    B2_TRIANGULAR = np.array([-2 * np.pi / np.sqrt(3), 2 * np.pi])
    DK = 1e-2                   # infinitesimal k-step for derivatives in units of 1/aM
    DX = 1e-2                   # infinitesimal x-step for derivatives in units of aM
    DY = DX                     # infinitesimal y-step for derivatives in units of aM

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    HBAR2_M = 7618              # hbar^2 / (m) in units of meV * Angstrom^2
    MSTAR = 0.35                # effective mass in units of electron mass
    E_SQUARED = 14440           # Coulomb's constant in units of meV * Angstroms

    def __init__(self, a0=DEFAULT_A0, theta_deg=DEFAULT_THETA_DEG, V=DEFAULT_V, psi_deg=DEFAULT_PSI_DEG,
                 n_shells=DEFAULT_SHELLS, epsilon=DEFAULT_EPSILON, kmesh=DEFAULT_K_MESH, nx=DEFAULT_NX, ny=DEFAULT_NY, 
                 scale=DEFAULT_SCALE):
        self.a0 = a0
        self.theta_deg = theta_deg
        self.V = V
        self.psi_deg = psi_deg
        self.n_shells = n_shells
        self.epsilon = epsilon

        self.kmesh = kmesh
        self.nx = nx
        self.ny = ny
        self.scale = scale

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
        First moiré real-space lattice vector in aM units.

        For a triangular lattice:
            a1 = (√3/2,  1/2)
        """
        return self.A1_TRIANGULAR
    
    @property
    def a2(self) -> np.ndarray:
        """
        Second moiré real-space lattice vector in aM units.

        For a triangular lattice:
            a2 = (0, 1)
        """
        return self.A2_TRIANGULAR

    @property
    def A_unit_cell(self) -> float:
        """
        Area of the moiré unit cell in aM^2 units.

        For a hexagonal lattice:
            A_unit_cell = a1 × a2 = |a1| * |a2| * sin(120°) = (√3/2)
        """
        return np.linalg.norm(np.cross(self.A1_TRIANGULAR, self.A2_TRIANGULAR))

    @property
    def b1(self) -> np.ndarray:
        """
        First moiré reciprocal-space lattice vector in 1/aM units.

        For a hexagonal lattice:
            b1 = (4π / √3, 0)
        """
        return self.B1_TRIANGULAR

    @property
    def b2(self) -> np.ndarray:
        """
        Second moiré reciprocal-space lattice vector in 1/aM units.

        For a hexagonal lattice:
            b1 = (-2π / √3, 2π)
        """
        return self.B2_TRIANGULAR

    @property
    def g1(self) -> np.ndarray:
        """
        First shell reciprocal-space lattice vector in 1/aM units.

        For a hexagonal lattice:
            g1 = b1 = (4π / √3, 0)
        """
        return self.B1_TRIANGULAR

    @property
    def g3(self) -> np.ndarray:
        """
        First shell reciprocal-space lattice vector in 1/aM units.

        For a hexagonal lattice:
            g3 = R120 @ g1
        """
        return utils.rotate(120) @ self.g1

    @property
    def g5(self) -> np.ndarray:
        """
        First shell reciprocal-space lattice vector in 1/aM units.

        For a hexagonal lattice:
            g5 = R120 @ g2
        """
        return utils.rotate(120) @ self.g3

    # ------------------------------------------------------------------
    # High-symmetry points in the Brillouin zone (BZ)
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> np.ndarray:
        """
        Gamma point in the Brillouin zone (BZ) in 1/aM units.
        
            γ = (0, 0)
        """
        return np.zeros(2)

    @property
    def kappa_pp(self) -> np.ndarray:
        """
        Kappa'' point in the Brillouin zone (BZ) in 1/aM units.

            κ'' = (2*b1 + b2) / 3
        """
        kappa_pp = ( 2 * self.b1 + self.b2 ) / 3
        return kappa_pp
    
    @property
    def kappa_p(self) -> np.ndarray:
        """
        Kappa' point in the Brillouin zone (BZ) in 1/aM units.

            κ' = R60 * κ''
        """
        kappa_p = utils.rotate(60) @ self.kappa_pp
        return kappa_p
    
    @property
    def kappa(self) -> np.ndarray:
        """
        Kappa point in the Brillouin zone (BZ) in 1/aM units.

            κ = R180 * κ'
        """
        kappa = utils.rotate(180) @ self.kappa_p
        return kappa

    @property
    def ktheta(self) -> np.ndarray:
        """
        Ktheta momentumm scale in 1/aM units.

            κθ = ||g_1||
        """
        ktheta = np.linalg.norm(self.g1)
        return ktheta

    @property
    def A_BZ(self) -> float:
        """
        Area of the BZ in 1/aM^2 units.

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
        v0 = self.kappa_pp
        verts = [v0]
        v = v0.copy()
        for _ in range(6):
            v = utils.rotate(60) @ v
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
            return self.V * np.exp(- 1j * self.psi)
        if (dm, dn) in ((-1, 0), (0, -1), (1, 1)):
            return self.V * np.exp(1j * self.psi)
        return 0.0

    def _build_offdiag(self):
        """
        Off diagonal part of the Hammiltonian (k-independent).
        """
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
        g_vecs = np.array([g for g, _ in self.basis])
        k_plus_g = (kpath[:, None, :] + g_vecs[None, :, :]) / self.aM

        # squared norms → shape (n_k, N)
        k2 = np.einsum('ijk,ijk->ij', k_plus_g, k_plus_g)

        return -self.HBAR2_M * k2 / (2 * self.MSTAR)

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
            k_plus_g = (k + g) / self.aM
            H[i, i] = - self.HBAR2_M * np.linalg.norm(k_plus_g)**2 / (2 * self.MSTAR)

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
        kpath : np.ndarray, shape (n_pts, 2)  k-points.

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
        k_point : array_like, shape (2,), optional Crystal momentum 
                  in 1/aM units. Defaults to κ.

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

    def moire_potential(self, X, Y):
        """
        Real-space moiré potential on a grid.

        Sum of three cosine terms:
            V(r) = 2V Σ_{j=1,3,5} cos(g_j · r + ψ)

        Parameters
        ----------
        X : array_like, shape (Ny, Nx)  X-coordinates in aM units.
        Y : array_like, shape (Ny, Nx)  Y-coordinates in aM units.

        Returns
        -------
        float  Potential value [meV].
        """
        potential = 0.0
        g_list = [self.g1, self.g3, self.g5]

        for g in g_list:
            potential += np.cos(g[0] * X + g[1] * Y + self.psi)

        return 2 * self.V * potential

    def mesh_k_unit_cell(self):
        """
        Sample one parallelogram unit cell of the reciprocal lattice.
        i/Nx and j/Ny run over [0, 1), so no double-counting at boundaries.
        """
        Nx = self.kmesh
        Ny = self.kmesh
        i = np.arange(0, Nx)
        j = np.arange(0, Ny)
        ii, jj = np.meshgrid(i, j, indexing='ij')
        k_points = (ii[..., None] * self.b1 / Nx) + (jj[..., None] * self.b2 / Ny)
        return k_points.reshape(-1, k_points.shape[-1])

    # ------------------------------------------------------------------
    # Wannier functions by projection
    # ------------------------------------------------------------------

    def wannier_function(self, X, Y, trial_centers, bands=(-1,), R0=None):
        """
        Two-band projective Wannier functions
        Returns array (n_w, *x.shape), one Wannier function per trial center.
        """
        if R0 is None:
            R0 = np.zeros(2)

        x, y = np.asarray(X), np.asarray(Y)
        orig_shape = x.shape
        xf, yf = x.ravel(), y.ravel()                         # (Npts,)
        k_points = self.mesh_k_unit_cell()
        n_k = k_points.shape[0]

        trial_centers = np.asarray(trial_centers)             # (n_w, 2)
        n_w = trial_centers.shape[0]

        # --- Bloch coefficients z^alpha_{k,G} for the chosen bands -------------
        H_all = self.H_k_all(k_points)                        # (n_k, N, N)
        _, eigenvectors = np.linalg.eigh(H_all)
        zk = eigenvectors[:, :, list(bands)]                  # (n_k, N, n_w)  [k, G, alpha]

        g_vecs = np.array([g for g, _ in self.basis])         # (N, 2)
        kG = k_points[:, None, :] + g_vecs[None, :, :]        # (n_k, N, 2)  (k+G)

        tphase = np.exp(-1j * np.einsum('kGd,id->kGi', kG, trial_centers))  # (n_k, N, n_w)
        O = np.einsum('kGa,kGi->kai', np.conj(zk), tphase)             # (n_k, n_w, n_w)  [k, alpha, i]

        C = np.empty_like(O)                                  # coeff of psi_alpha in psi_tilde_i
        smin = np.empty(n_k)
        for kk in range(n_k):
            S = O[kk].conj().T @ O[kk]                        # S_{ij} = sum_a O*_{a i} O_{a j}
            d, V = np.linalg.eigh(S)                          # S = V diag(d) V^dag
            smin[kk] = np.sqrt(max(d.min(), 0.0))
            d = np.maximum(d, 1e-12)                          # roundoff guard
            S_inv_sqrt = V @ np.diag(1 / np.sqrt(d)) @ V.conj().T
            C[kk] = O[kk] @ S_inv_sqrt                        # (alpha, i)
        self._last_smin = smin.min()                          # conditioning monitor

        # --- gauge-fixed plane-wave coeffs: ztilde^i_{kG} = sum_a C_{a i} (z^a_{kG})*
        ztilde = np.einsum('kai,kGa->kGi', C, zk)    # (n_k, N, n_w)

        # --- step 5/6: Wannier functions W_i(r) = (1/sqrt(N)) sum_k sum_G ztilde e^{i(k+G).r}
        g_phase = np.exp(1j * (g_vecs[:, 0, None] * xf[None, :]
                            + g_vecs[:, 1, None] * yf[None, :]))   # (N, Npts)
        k_phase = np.exp(1j * (k_points[:, 0, None] * xf[None, :]
                            + k_points[:, 1, None] * yf[None, :]))  # (n_k, Npts)

        shift = np.exp(-1j * (k_points @ R0))    # (n_k,)

        u = np.einsum('kGi,GP->kiP', ztilde, g_phase)         # (n_k, n_w, Npts)
        bloch = u * k_phase[:, None, :]                       # (n_k, n_w, Npts)
        bloch *= shift[:, None, None]
        wannier = bloch.sum(axis=0) / (n_k * np.sqrt(self.A_unit_cell))

        res = wannier.reshape((n_w,) + orig_shape)       # (n_w, *orig_shape)

        if n_w == 1:
            return res[0, :, :]                          # (*orig_shape)
        else:
            return res

    def integrate_wannier(self, fun, wannier, X, Y):
        """
        Computes the expectation value in real space of a function 
        f(X, Y) with respect to the wannier probability density.

        Parameters
        ----------
        wannier : array_like, shape (Ny, Nx) wannier wavefunction.
        X : array_like, shape (Ny, Nx)  X-coordinates in aM units.
        Y : array_like, shape (Ny, Nx)  Y-coordinates in aM units.

        Returns
        -------
        float  value of the integral.
        """
        density = np.abs(wannier)**2

        integral = np.trapz(
            np.trapz(fun(X, Y) * density, X[0,:], axis=1),
            Y[:,0], axis=0 )

        return integral

    # ------------------------------------------------------------------
    # Covariant derivatives and quantum geometric quantities
    # ------------------------------------------------------------------

    def dual_states(self, k_points, bands=(-1,)):
        """
        Calculates dual states

        Parameters
        ----------
        TODO

        Returns
        -------
        float  TODO
        """
        shifts = self.DK * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

        k_shifted = k_points[None, :, :] + shifts[:, None, :]
        k_all = np.vstack([k_points, k_shifted.reshape(-1, 2)])

        H_all = self.H_k_all(k_all)
        _, eigvecs = np.linalg.eigh(H_all)              # ascending
        U_all = eigvecs[:, :, list(bands)]              # (5*N, Nh, nb)

        N = len(k_points)
        U = U_all[0*N:1*N]                              # base states u_{nk}  (N, Nh, nb)
        U_shifted = (U_all[1*N:2*N], U_all[2*N:3*N],
                     U_all[3*N:4*N], U_all[4*N:5*N])

        def make_dual(U_shift):
            S = np.einsum('kan,kam->knm', np.conj(U), U_shift)
            Sinv = np.linalg.inv(S)
            
            return np.einsum('kam,kmn->kan', U_shift, Sinv)

        dual_xp, dual_xm, dual_yp, dual_ym = map(make_dual, U_shifted)

        return dual_xp, dual_xm, dual_yp, dual_ym

    def covariant_derivatives(self, k_points, bands=(-1,)):
        """
        Calculates covariant derivatives

        Parameters
        ----------
        TODO

        Returns
        -------
        float  TODO
        """
        dual_xp, dual_xm, dual_yp, dual_ym = self.dual_states(k_points, bands)
        dX = (dual_xp - dual_xm) / (2 * self.DK)
        dY = (dual_yp - dual_ym) / (2 * self.DK)
        return dX, dY

    def berry_curvature(self, k_points, bands=(-1,)):
        """
        Calculates Berry curvature

        Parameters
        ----------
        TODO

        Returns
        -------
        float  TODO
        """
        dX, dY = self.covariant_derivatives(k_points, bands)
        overlap = np.sum(np.conj(dX) * dY, axis=-2)         # (N_kpoints, n_bands)
        return -2 * np.imag(overlap)

    def quantum_metric(self, k_points, bands=(-1,)):
        """
        Calculates quantum metric

        Parameters
        ----------
        TODO

        Returns
        -------
        float  TODO
        """
        dX, dY = self.covariant_derivatives(k_points, bands)
        g_xx = np.real(np.sum(np.conj(dX) * dX, axis=-2))   # (N_kpoints, n_bands)
        g_yy = np.real(np.sum(np.conj(dY) * dY, axis=-2))
        g_xy = np.real(np.sum(np.conj(dX) * dY, axis=-2))
        return g_xx, g_xy, g_yy
    
    def integrate_trace(self, Nx, Ny, bands=(-1,)):
        """
        Integrates trace of metric

        Parameters
        ----------
        TODO

        Returns
        -------
        float  TODO
        """
        k_points, dkx, dky, jac = self.mesh_k_hex(Nx=Nx, Ny=Ny)
        k_points_bz = self.map_to_first_BZ(k_points)

        g_xx, _, g_yy = self.quantum_metric(k_points_bz, bands)

        if len(bands) == 1:
            trace = g_xx + g_yy

            dA = dkx * dky * jac
            integral = (1 / self.A_BZ) * np.sum(trace) * dA

            return integral
        
        elif len(bands) == 2:
            GX = g_xx[:, 0] + g_xx[:, 1]
            GY = g_yy[:, 0] + g_yy[:, 1]
            trace = GX + GY
            
            dA = dkx * dky * jac
            integral = (1 / self.A_BZ) * np.sum(trace) * dA

            integral0 = (1 / self.A_BZ) * np.sum(g_xx[:, 0] + g_yy[:, 0]) * dA
            integral1 = (1 / self.A_BZ) * np.sum(g_xx[:, 1] + g_yy[:, 1]) * dA

            return integral, integral0, integral1
        

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

        # k_reduced: (N, 2), G_vectors: (M, 2)
        diff = k_reduced[:, None, :] - G_vectors[None, :, :]  # (N, M, 2)
        norms = np.linalg.norm(diff, axis=-1)                 # (N, M)
        best = np.argmin(norms, axis=-1)                      # (N,)
        k_first_BZ = k_reduced - G_vectors[best]              # (N, 2)

        return k_first_BZ

    def mesh_k_hex(self, Nx=10, Ny=10):
        """
        mesh of...

        Parameters
        ----------
        TODO

        Returns
        -------
        float  TODO
        """
        i = np.arange(0, Nx)
        j = np.arange(0, Ny)
        dkx = np.linalg.norm(self.b1) / Nx
        dky = np.linalg.norm(self.b2) / Ny
        jac = np.sqrt(3) / 2                # area factor for triangular mesh

        ii, jj = np.meshgrid(i, j, indexing='ij')
        k_points = (ii[..., None] * self.b1 / (Nx)) + (jj[..., None] * self.b2 / (Ny))
        
        return k_points.reshape(-1, k_points.shape[-1]), dkx, dky, jac

    # ------------------------------------------------------------------
    # Tight-binding parameters
    # ------------------------------------------------------------------

    def coulomb_potential(self, r, soft=1e-2):
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
        # Dimensionless units converted through aM
        V0 = self.E_SQUARED / (self.epsilon * self.aM)
        radial = 1 / np.sqrt(r**2 + soft**2)

        return V0 * radial

    def hopping_1band(self, orders):
        shifts = []
        for order in orders:
            if order == 0:
                shifts.append(np.zeros(2))
            elif order == 1:
                shifts.append(self.a2)
            elif order == 2:
                shifts.append(self.a2 + self.a1)
            elif order == 3:
                shifts.append(2 * self.a2)

        delta_center = [np.zeros(2)]

        x = np.linspace(-self.scale, self.scale, self.nx)
        y = np.linspace(-self.scale, self.scale, self.ny)
        X, Y = np.meshgrid(x, y)

        wannier0 = self.wannier_function(X, Y, trial_centers=delta_center)
        wannier_xp = self.wannier_function(X + self.DX, Y, trial_centers=delta_center)
        wannier_xm = self.wannier_function(X - self.DX, Y, trial_centers=delta_center)
        wannier_yp = self.wannier_function(X, Y + self.DY, trial_centers=delta_center)
        wannier_ym = self.wannier_function(X, Y - self.DY, trial_centers=delta_center)

        laplacian = (wannier_xp + wannier_xm + wannier_yp + wannier_ym - 4 * wannier0) / (self.aM * self.DX)**2

        hopping_integrals = []
        for Rshift in shifts:
            wannier2 = self.wannier_function(X, Y, trial_centers=delta_center, R0=Rshift)

            integralT = 0.5 * (self.HBAR2_M / self.MSTAR) * np.trapz(
                np.trapz(np.conj(wannier2) * laplacian, X[0,:], axis=1),
                Y[:,0], axis=0
            )

            integralV = np.trapz(
                np.trapz(np.conj(wannier2) * self.moire_potential(X, Y) * wannier0, X[0,:], axis=1),
                Y[:,0], axis=0
            )

            integral = integralT + integralV
            hopping_integrals.append(integral)

        return hopping_integrals

    def hopping_1band_kspace(self, orders):
        shifts = []
        for order in orders:
            if order == 0:
                shifts.append(np.zeros(2))
            elif order == 1:
                shifts.append(self.a2)
            elif order == 2:
                shifts.append(self.a2 + self.a1)
            elif order == 3:
                shifts.append(2 * self.a2)

        k_points = self.mesh_k_unit_cell()
        bands, _ = self.calculate_bands(k_points)   # bands: (n_pts, N), meV
        Ek = bands[:, 0]                # (n_pts,)
        N = k_points.shape[0]

        hopping_integrals = []
        for Rshift in shifts:
            phase = np.exp(1j * (k_points @ Rshift))   # k·R0 for every k
            t_R = np.sum(Ek * phase) / N
            hopping_integrals.append(t_R)

        return hopping_integrals

    def interaction_1band(self, orders):
        shifts = []
        for order in orders:
            if order == 0:
                shifts.append(np.zeros(2))
            elif order == 1:
                shifts.append(self.a2)
            elif order == 2:
                shifts.append(self.a2 + self.a1)
            elif order == 3:
                shifts.append(2 * self.a2)

        delta_center = [np.zeros(2)]
        x = np.linspace(-self.scale, self.scale, self.nx)
        y = np.linspace(-self.scale, self.scale, self.ny)
        X, Y = np.meshgrid(x, y)

        dx_grid = np.abs(X[0, 1] - X[0, 0])
        dy_grid = np.abs(Y[1, 0] - Y[0, 0])

        wannier1 = self.wannier_function(X, Y, trial_centers=delta_center)
        density1 = np.abs(wannier1)**2
        rho1 = density1.ravel()

        interactions = []
        for Rshift in shifts:
            wannier2 = self.wannier_function(X, Y, trial_centers=delta_center, R0=Rshift)

            density2 = np.abs(wannier2)**2
            rho2 = density2.ravel()

            softness = np.min([dx_grid, dy_grid]) / 2
            integral = _coulomb_integral_real(
                rho2,
                rho1,
                X.ravel(),
                Y.ravel(),
                self.E_SQUARED / (self.epsilon * self.aM),
                softness,
                (dx_grid * dy_grid)**2,
            )
            interactions.append(integral)

        return interactions

    def exchange_1band(self, orders=[1]):
        shifts = []
        for order in orders:
            if order == 0:
                shifts.append(np.zeros(2))
            elif order == 1:
                shifts.append(self.a2)
            elif order == 2:
                shifts.append(self.a2 + self.a1)
            elif order == 3:
                shifts.append(2 * self.a2)

        delta_center = [np.zeros(2)]
        x = np.linspace(-self.scale, self.scale, self.nx)
        y = np.linspace(-self.scale, self.scale, self.ny)
        X, Y = np.meshgrid(x, y)

        dx_grid = np.abs(X[0, 1] - X[0, 0])
        dy_grid = np.abs(Y[1, 0] - Y[0, 0])

        wannier1 = self.wannier_function(X, Y, trial_centers=delta_center)

        interactions = []
        for Rshift in shifts:
            wannier2 = self.wannier_function(X, Y, trial_centers=delta_center, R0=Rshift)

            A = np.conj(wannier1) * wannier2
            a = A.ravel()
    
            softness = np.min([dx_grid, dy_grid])/2
            integral = _coulomb_integral_complex(
                np.conj(a),
                a,
                X.ravel(),
                Y.ravel(),
                self.E_SQUARED / (self.epsilon * self.aM),
                softness,
                (dx_grid * dy_grid)**2,
            )
            interactions.append(integral)

        return interactions

    def spin_coupling_1band(self):
        t1 = np.real( self.hopping_1band_kspace([1])[0] )
        U0 = np.real( self.interaction_1band([0])[0] )
        X = np.real( self.exchange_1band([1])[0] )

        ref_energy = (self.HBAR2_M / self.aM**2) * np.linalg.norm(self.kappa)**2

        J = 4 * t1**2 / U0 - 2 * X
        J /= ref_energy

        return J

    def hopping_integral(self, X, Y, k_points, sublattice1, sublattice2, order=1, Rshift=None):
        if order == 0:
            R0 = np.zeros(2)
        elif order == 1:
            R0 = self.a2 - self.a1

        if Rshift is not None:
            R0 = Rshift

        ## Trial functions centers
        RA = np.zeros(2)
        RB = (2 * self.a1 - self.a2) / 3
        ##

        if sublattice1 == "A":
            ind1 = 0
        elif sublattice1 == "B":
            ind1 = 1

        if sublattice2 == "A":
            ind2 = 0
        elif sublattice2 == "B":
            ind2 = 1

        wannier0 = self.wannier_wavefunction_2band(X, Y, k_points, trial_centers= np.array([RA, RB]))[ind1, :, :]
        wannier_xp = self.wannier_wavefunction_2band(X + utils.DX, Y, k_points, trial_centers= np.array([RA, RB]))[ind1, :, :]
        wannier_xm = self.wannier_wavefunction_2band(X - utils.DX, Y, k_points, trial_centers= np.array([RA, RB]))[ind1, :, :]
        wannier_yp = self.wannier_wavefunction_2band(X, Y + utils.DX, k_points, trial_centers= np.array([RA, RB]))[ind1, :, :]
        wannier_ym = self.wannier_wavefunction_2band(X, Y - utils.DX, k_points, trial_centers= np.array([RA, RB]))[ind1, :, :]

        wannier2 = self.wannier_wavefunction_2band(X, Y, k_points, trial_centers= np.array([RA, RB]), R0=R0)[ind2, :, :]
        
        laplacian = (wannier_xp + wannier_xm + wannier_yp + wannier_ym - 4 * wannier0) / (utils.DX**2)

        integralT = 0.5 * (self.HBAR2_M / self.MSTAR) * np.trapz(
            np.trapz(np.conj(wannier2) * laplacian, X[0,:], axis=1),
            Y[:,0],
            axis=0
        )

        integralV = np.trapz(
            np.trapz(np.conj(wannier2) * self.moire_potential_XY(X, Y) * wannier0, X[0,:], axis=1),
            Y[:,0],
            axis=0
        )

        integral = integralT + integralV

        return integral
    
    def nearest_neighbour_interaction(self, X, Y, k_points, sublattice1, sublattice2, order=1, Rshift=None):
        if order == 0:
            R0 = np.zeros(2)
        elif order == 1:
            R0 = self.a2 - self.a1

        if Rshift is not None:
            R0 = Rshift

        ## Trial functions centers
        RA = np.zeros(2)
        RB = (2 * self.a1 - self.a2) / 3
        ##

        if sublattice1 == "A":
            ind1 = 0
        elif sublattice1 == "B":
            ind1 = 1

        if sublattice2 == "A":
            ind2 = 0
        elif sublattice2 == "B":
            ind2 = 1

        dx_grid = np.abs(X[0, 1] - X[0, 0])
        dy_grid = np.abs(Y[1, 0] - Y[0, 0])

        wannier1 = self.wannier_wavefunction_2band(X, Y, k_points, trial_centers= np.array([RA, RB]))[ind1, :, :]
        wannier2 = self.wannier_wavefunction_2band(X, Y, k_points, trial_centers= np.array([RA, RB]), R0=R0)[ind2, :, :]

        density1 = np.abs(wannier1)**2
        density2 = np.abs(wannier2)**2
        rho1 = density1.ravel()
        rho2 = density2.ravel()

        R = np.column_stack((X.ravel(), Y.ravel()))

        dr = R[:, None, :] - R[None, :, :]
        dist = np.linalg.norm(dr, axis=-1)

        softness = np.min([dx_grid, dy_grid]) / 2
        K = self.coulomb_potential(dist, soft=softness)      # shape (N, N)

        integral = (dx_grid * dy_grid)**2 * np.dot(rho2, K @ rho1)
        
        return integral
    
    def exchange_interaction(self, X, Y, k_points, order=1):
        if order == 0:
            R0 = np.zeros(2)
        elif order == 1:
            R0 = self.a2
        elif order == 2:
            R0 = self.a2 + self.a1
            
        dx_grid = np.abs(X[0, 1] - X[0, 0])
        dy_grid = np.abs(Y[1, 0] - Y[0, 0])

        wannier1 = self.wannier_wavefunction(X, Y, k_points)
        wannier2 = self.wannier_wavefunction(X, Y, k_points, R0=R0)

        A = np.conj(wannier1) * wannier2
        a = A.ravel()

        R = np.column_stack((X.ravel(), Y.ravel()))

        dr = R[:, None, :] - R[None, :, :]
        dist = np.linalg.norm(dr, axis=-1)

        softness = np.min([dx_grid, dy_grid])/2
        K = self.coulomb_potential(dist, soft=softness)      # shape (N, N)

        integral = (dx_grid * dy_grid)**2 * np.vdot(a, K @ a)
        
        return integral

    def _prepare_hopping_2band(self, k_points, trial_centers, bands):
        """Prepare projection data shared by hopping_2band evaluations."""
        trial_centers = np.asarray(trial_centers)
        n_k = k_points.shape[0]
        H_all = self.H_k_all(k_points)
        _, eigenvectors = np.linalg.eigh(H_all)
        zk = eigenvectors[:, :, list(bands)]

        g_vecs = np.array([g for g, _ in self.basis])
        kG = k_points[:, None, :] + g_vecs[None, :, :]
        tphase = np.exp(-1j * np.einsum(
            "kGd,id->kGi", kG, trial_centers
        ))
        O = np.einsum("kGa,kGi->kai", np.conj(zk), tphase)

        C = np.empty_like(O)
        smin = np.empty(n_k)
        for kk in range(n_k):
            S = O[kk].conj().T @ O[kk]
            d, V = np.linalg.eigh(S)
            smin[kk] = np.sqrt(max(d.min(), 0.0))
            d = np.maximum(d, 1e-12)
            S_inv_sqrt = V @ np.diag(1 / np.sqrt(d)) @ V.conj().T
            C[kk] = O[kk] @ S_inv_sqrt
        self._last_smin = smin.min()
        return np.einsum("kai,kGa->kGi", C, zk), g_vecs

    def _evaluate_hopping_2band(self, X, Y, k_points, prepared, R0=None):
        """Evaluate a projected Wannier function from prepared data."""
        if R0 is None:
            R0 = np.zeros(2)

        x, y = np.asarray(X), np.asarray(Y)
        orig_shape = x.shape
        xf, yf = x.ravel(), y.ravel()
        n_k = k_points.shape[0]
        ztilde, g_vecs = prepared
        n_w = ztilde.shape[-1]

        g_phase = np.exp(1j * (
            g_vecs[:, 0, None] * xf[None, :]
            + g_vecs[:, 1, None] * yf[None, :]
        ))
        k_phase = np.exp(1j * (
            k_points[:, 0, None] * xf[None, :]
            + k_points[:, 1, None] * yf[None, :]
        ))
        shift = np.exp(-1j * (k_points @ R0))

        u = np.einsum("kGi,GP->kiP", ztilde, g_phase)
        bloch = u * k_phase[:, None, :]
        bloch *= shift[:, None, None]
        wannier = bloch.sum(axis=0) / (n_k * np.sqrt(self.A_unit_cell))
        res = wannier.reshape((n_w,) + orig_shape)
        return res[0, :, :] if n_w == 1 else res

    def hopping_2band(self, orders, sublattice1, sublattice2):
        shifts = []
        for order in orders:
            if order == 0:
                shifts.append(np.zeros(2))
            elif order == 1:
                shifts.append(self.a2 - self.a1)

        ## Trial functions centers
        RA = np.zeros(2)
        RB = (2 * self.a1 - self.a2) / 3
        delta_centers = np.array([RA, RB])

        ind1 = {"A": 0, "B": 1}[sublattice1]
        ind2 = {"A": 0, "B": 1}[sublattice2]

        x = np.linspace(-self.scale, self.scale, self.nx)
        y = np.linspace(-self.scale, self.scale, self.ny)
        X, Y = np.meshgrid(x, y)

        k_points = self.mesh_k_unit_cell()
        prepared = self._prepare_hopping_2band(k_points, delta_centers, bands=(-2, -1))
        wannier0 = self._evaluate_hopping_2band(X, Y, k_points, prepared)[ind1, :, :]
        wannier_xp = self._evaluate_hopping_2band(X + self.DX, Y, k_points, prepared)[ind1, :, :]
        wannier_xm = self._evaluate_hopping_2band(X - self.DX, Y, k_points, prepared)[ind1, :, :]
        wannier_yp = self._evaluate_hopping_2band(X, Y + self.DX, k_points, prepared)[ind1, :, :]
        wannier_ym = self._evaluate_hopping_2band(X, Y - self.DX, k_points, prepared)[ind1, :, :]

        laplacian = (wannier_xp + wannier_xm + wannier_yp + wannier_ym - 4 * wannier0) / (self.aM * self.DX)**2

        hopping_integrals = []
        for Rshift in shifts:
            wannier2 = self._evaluate_hopping_2band(X, Y, k_points, prepared, R0=Rshift)[ind2, :, :]

            integralT = 0.5 * (self.HBAR2_M / self.MSTAR) * np.trapz(
                np.trapz(np.conj(wannier2) * laplacian, X[0,:], axis=1),
                Y[:,0], axis=0
            )

            integralV = np.trapz(
                np.trapz(np.conj(wannier2) * self.moire_potential(X, Y) * wannier0, X[0,:], axis=1),
                Y[:,0], axis=0
            )

            integral = integralT + integralV
            hopping_integrals.append(integral)

        return hopping_integrals

    def hopping_2band_kspace(self, orders, sublattice1, sublattice2):
        """Calculate two-band hoppings by Fourier transforming the projected Hamiltonian.

        Parameters
        ----------
        orders : iterable of int
            ``0`` selects the onsite term and ``1`` selects the nearest-neighbor
            displacement ``a2 - a1``.
        sublattice1, sublattice2 : {"A", "B"}
            Row and column sublattice labels of the hopping matrix element.

        Returns
        -------
        list of complex
            Fourier coefficients for the requested displacements.
        """
        shifts = []
        for order in orders:
            if order == 0:
                shifts.append(np.zeros(2))
            elif order == 1:
                shifts.append(self.a2 - self.a1)

        ind1 = {"A": 0, "B": 1}[sublattice1]
        ind2 = {"A": 0, "B": 1}[sublattice2]
        trial_centers = np.array([
            np.zeros(2),
            (2 * self.a1 - self.a2) / 3,
        ])
        k_points = self.mesh_k_unit_cell()
        bands = (-2, -1)

        H_all = self.H_k_all(k_points)
        eigenvalues, eigenvectors = np.linalg.eigh(H_all)
        zk = eigenvectors[:, :, list(bands)]

        g_vecs = np.array([g for g, _ in self.basis])
        kG = k_points[:, None, :] + g_vecs[None, :, :]
        tphase = np.exp(-1j * np.einsum(
            "kGd,id->kGi", kG, trial_centers
        ))
        overlap = np.einsum("kGa,kGi->kai", np.conj(zk), tphase)

        n_k = k_points.shape[0]
        n_w = len(bands)
        gauge = np.empty_like(overlap)
        for kk in range(n_k):
            overlap_metric = overlap[kk].conj().T @ overlap[kk]
            metric_values, metric_vectors = np.linalg.eigh(overlap_metric)
            metric_values = np.maximum(metric_values, 1e-12)
            inverse_sqrt = metric_vectors @ np.diag(
                1 / np.sqrt(metric_values)
            ) @ metric_vectors.conj().T
            gauge[kk] = overlap[kk] @ inverse_sqrt

        selected_energies = eigenvalues[:, list(bands)]
        projected_hamiltonian = np.einsum(
            "kai,ka,kaj->kij",
            np.conj(gauge),
            selected_energies,
            gauge,
        )

        return [
            np.mean(
                projected_hamiltonian[:, ind2, ind1]
                * np.exp(1j * (k_points @ Rshift))
            )
            for Rshift in shifts
        ]

    def interaction_2band(self, orders, sublattice1, sublattice2):
        shifts = []
        for order in orders:
            if order == 0:
                shifts.append(np.zeros(2))
            elif order == 1:
                shifts.append(self.a2 - self.a1)

        ## Trial functions centers
        RA = np.zeros(2)
        RB = (2 * self.a1 - self.a2) / 3
        delta_centers = np.array([RA, RB])

        ind1 = {"A": 0, "B": 1}[sublattice1]
        ind2 = {"A": 0, "B": 1}[sublattice2]

        x = np.linspace(-self.scale, self.scale, self.nx)
        y = np.linspace(-self.scale, self.scale, self.ny)
        X, Y = np.meshgrid(x, y)

        dx_grid = np.abs(X[0, 1] - X[0, 0])
        dy_grid = np.abs(Y[1, 0] - Y[0, 0])

        wannier1 = self.wannier_function(X, Y, trial_centers=delta_centers)[ind1, :, :]
        density1 = np.abs(wannier1)**2
        rho1 = density1.ravel()

        interactions = []
        for Rshift in shifts:
            wannier2 = self.wannier_function(X, Y, trial_centers=delta_centers, R0=Rshift)[ind2, :, :]
            density2 = np.abs(wannier2)**2
            rho2 = density2.ravel()

            softness = np.min([dx_grid, dy_grid]) / 2
            integral = _coulomb_integral_real(
                rho2,
                rho1,
                X.ravel(),
                Y.ravel(),
                self.E_SQUARED / (self.epsilon * self.aM),
                softness,
                (dx_grid * dy_grid)**2,
            )
            interactions.append(integral)

        return interactions

    def exchange_2band(self, orders, sublattice1, sublattice2):
        shifts = []
        for order in orders:
            if order == 0:
                shifts.append(np.zeros(2))
            elif order == 1:
                shifts.append(self.a2 - self.a1)

        ## Trial functions centers
        RA = np.zeros(2)
        RB = (2 * self.a1 - self.a2) / 3
        delta_centers = np.array([RA, RB])

        ind1 = {"A": 0, "B": 1}[sublattice1]
        ind2 = {"A": 0, "B": 1}[sublattice2]

        x = np.linspace(-self.scale, self.scale, self.nx)
        y = np.linspace(-self.scale, self.scale, self.ny)
        X, Y = np.meshgrid(x, y)

        dx_grid = np.abs(X[0, 1] - X[0, 0])
        dy_grid = np.abs(Y[1, 0] - Y[0, 0])

        wannier1 = self.wannier_function(X, Y, trial_centers=delta_centers)[ind1, :, :]

        interactions = []
        for Rshift in shifts:
            wannier2 = self.wannier_function(X, Y, trial_centers=delta_centers, R0=Rshift)[ind2, :, :]
            A = np.conj(wannier1) * wannier2
            a = A.ravel()
    
            softness = np.min([dx_grid, dy_grid])/2
            integral = _coulomb_integral_complex(
                np.conj(a),
                a,
                X.ravel(),
                Y.ravel(),
                self.E_SQUARED / (self.epsilon * self.aM),
                softness,
                (dx_grid * dy_grid)**2,
            )
            interactions.append(integral)

        return interactions

    def spin_coupling_2band(self):
        hoppingsA = self.hopping_2band_kspace(orders=[0, 1], sublattice1="A", sublattice2="A")
        onsiteA, tAA = hoppingsA[0], hoppingsA[1]

        onsiteB = self.hopping_2band_kspace(orders=[0], sublattice1="B", sublattice2="B")[0]
        Delta = onsiteA - onsiteB

        shift = (onsiteB + onsiteA) / 2

        tAB = self.hopping_2band_kspace(orders=[0], sublattice1="A", sublattice2="B")[0]

        UA = self.interaction_2band(orders=[0], sublattice1="A", sublattice2="A")[0]
        UB = self.interaction_2band(orders=[0], sublattice1="B", sublattice2="B")[0]
        XA = self.exchange_2band(orders=[1], sublattice1="A", sublattice2="A")[0]

        ref_energy = (self.HBAR2_M / self.aM**2) * np.linalg.norm(self.kappa)**2

        J = (4 * (tAA**2) / UA) + (4 * (tAB**2) / (UA * Delta**2)) * (UA + 2 * Delta) * (-tAA) + (4 * (tAB**4) / (UA * Delta**2)) * (2 * UA + UB + 2 * Delta) / (UB + 2 * Delta)  - 2 * XA
        J /= ref_energy

        return J