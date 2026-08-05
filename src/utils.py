import numpy as np
import scipy as sp
from functools import lru_cache

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

HBAR2_M = 7618              # hbar^2 / (m) in units of meV*Angstrom^2
MSTAR = 0.35                # effective mass in units of electron mass
CHARGE_SQUARED = 14440      # Coulomb's constant in units of meV*Angstroms
DK = 1e-7                   # infinitesimal k-step for derivatives in units of Angstrom^-1
DX = 1e-3                   # infinitesimal x-step for derivatives in units of Angstrom
DY = DX                     # infinitesimal y-step for derivatives in units of Angstrom
ADIMB1 = np.array([4 * np.pi / np.sqrt(3), 0.0])
ADIMB2 = np.array([-2 * np.pi / np.sqrt(3), 2 * np.pi])


# ------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------

@lru_cache(maxsize=None)
def compute_basis_adim(n_shells):
    N = n_shells + 2
    candidates = []
    for n1 in range(-N, N + 1):
        for n2 in range(-N, N + 1):
            vec = n1 * np.array(ADIMB1) + n2 * np.array(ADIMB2)
            mag2 = round(float(vec @ vec), 6)  # consistent precision
            candidates.append((mag2, vec, (n1, n2)))

    # find the n_shells smallest NONZERO distinct magnitudes
    nonzero_mag2 = sorted(set(m for m, _, _ in candidates if m > 1e-10))
    thresholds = nonzero_mag2[:n_shells]

    basis = [(np.zeros(2), (0, 0))]
    for thresh in thresholds:
        for m, v, n in candidates:
            if abs(m - thresh) < 1e-5:
                basis.append((v, n))

    return basis

def integrate_wavefunction(psi, x, y):
    density = np.abs(psi) ** 2
    return np.trapz(np.trapz(density, x, axis=1), y, axis=0)

def ho_eigenstate(n, x, aM, V, psi):
    f = 16 * MSTAR * V * np.cos(psi) / (HBAR2_M * (aM**2))
    A = (1 / np.sqrt(2**n * sp.special.factorial(n)))
    B = (f)**0.125
    E = np.exp(-0.5 * np.sqrt(f * np.pi**2) * (x**2)) 
    H = sp.special.eval_hermite(n, ((f * np.pi**2)**0.25) * (-x))
    return A * B * E * H

def ho_2d_groundstate(x, y, aM, V, psi):
    """Calculate the 2D ground state of a harmonic oscillator."""
    return ho_eigenstate(0, x, aM, V, psi) * ho_eigenstate(0, y, aM, V, psi)

# def ho_2d_groundstate_perturbed(x, y, aM, V, psi):
#     """Calculate the 2D ground state of a harmonic oscillator with a small C3 perturbation."""
#     K00 = ho_2d_groundstate(x, y, aM, V, psi)

#     B = (np.tan(psi) / 12) * (HBAR2_M * np.pi**2 / (36 * MSTAR * V * np.cos(psi) * (aM**2))) ** 0.25
#     K12 = ho_eigenstate(1, x, aM, V, psi) * ho_eigenstate(2, y, aM, V, psi)
#     K30 = ho_eigenstate(3, x, aM, V, psi) * ho_eigenstate(0, y, aM, V, psi)
    
#     N = np.sqrt(1 + 2 * B**2 + (2 / 3) * B**2)
#     perturbed = ( 1 / N ) * ( K00 + B * (np.sqrt(2) * K12 - np.sqrt(2 / 3) * K30) )

#     return perturbed

def ho_2d_groundstate_perturbed(x, y, aM, V, psi):
    """Calculate the 2D ground state of a harmonic oscillator with a small C3 perturbation."""
    K00 = ho_2d_groundstate(x, y, aM, V, psi)
    K12 = ho_eigenstate(1, x, aM, V, psi) * ho_eigenstate(2, y, aM, V, psi)
    K30 = ho_eigenstate(3, x, aM, V, psi) * ho_eigenstate(0, y, aM, V, psi)
    K20 = ho_eigenstate(2, x, aM, V, psi) * ho_eigenstate(0, y, aM, V, psi)
    K02 = ho_eigenstate(0, x, aM, V, psi) * ho_eigenstate(2, y, aM, V, psi)
    K22 = ho_eigenstate(2, x, aM, V, psi) * ho_eigenstate(2, y, aM, V, psi)
    K40 = ho_eigenstate(4, x, aM, V, psi) * ho_eigenstate(0, y, aM, V, psi)
    K04 = ho_eigenstate(0, x, aM, V, psi) * ho_eigenstate(4, y, aM, V, psi)
    aW = spread_harmonic_approx(aM, V, psi)

    L3 = (16 * V * np.sin(psi) * np.pi**3) / (3 * np.sqrt(3) * aM**3)
    L4 = (8 * V * np.cos(psi) * np.pi**4) / (3 * aM**4)
    
    psi3 = - L3 * ( (0.5 * aW**2)**(3/2) ) * (MSTAR * aW**2 / (3 * HBAR2_M)) * (-3 * np.sqrt(2) * K12 + np.sqrt(6) * K30)
    psi4 = - L4 * ( (0.5 * aW**2)**2 ) * (MSTAR * aW**2 / (4 * HBAR2_M)) * (16 * np.sqrt(2) * (K20 + K02) + 4 * K22 + 2 * np.sqrt(6) * (K40 + K04))

    perturbed = K00 + psi3 + psi4

    return perturbed

def spread_harmonic_approx(aM, V, psi):
    """Calculate the spread of a harmonic oscillator."""
    A = np.sqrt(HBAR2_M * (aM**2) / (16 * MSTAR * V * np.cos(psi) * np.pi**2))
    return np.sqrt(A)

def spread_perturbation_old(aM, V, psi):
    """Calculate the spread of a harmonic oscillator with a small C3 perturbation."""
    B = (np.tan(psi) / 12) * (HBAR2_M * np.pi**2 / (36 * MSTAR * V * np.cos(psi) * (aM**2))) ** 0.25

    R2 = ( spread_harmonic_approx(aM, V, psi)**2 ) * (1 + (26 / 3) * B**2) / (1 + (8 / 3) * B**2)

    return np.sqrt(R2)

def spread_perturbation(aM, V, psi):
    """Calculate the spread of a harmonic oscillator with a small C3 perturbation."""
    # B = (np.tan(psi) / 12) * (HBAR2_M * np.pi**2 / (36 * MSTAR * V * np.cos(psi) * (aM**2))) ** 0.25

    R2 = ( spread_harmonic_approx(aM, V, psi)**2 ) * (1 +  (2 * np.pi**2 / 3) * (spread_harmonic_approx(aM, V, psi) / aM)**2 )

    return np.sqrt(R2)

