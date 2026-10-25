import numpy as np
import pytest

import utils
from HeteroBilayer import HeteroBilayer


@pytest.fixture
def hb():
    # small basis keeps the tests fast
    return HeteroBilayer(n_shells=3, kmesh=4)


# ------------------------------------------------------------------
# Lattice geometry
# ------------------------------------------------------------------

def test_moire_lattice_constant(hb):
    expected = hb.a0 / (2 * np.sin(np.radians(hb.theta_deg) / 2))
    assert np.isclose(hb.aM, expected)


def test_reciprocal_lattice_duality(hb):
    A = np.array([hb.a1, hb.a2])
    B = np.array([hb.b1, hb.b2])
    assert np.allclose(A @ B.T, 2 * np.pi * np.eye(2))


def test_areas(hb):
    assert np.isclose(hb.A_unit_cell, np.sqrt(3) / 2)
    assert np.isclose(hb.A_unit_cell * hb.A_BZ, (2 * np.pi) ** 2)


def test_first_shell_vectors_sum_to_zero(hb):
    assert np.allclose(hb.g1 + hb.g3 + hb.g5, 0.0)


def test_kappa_points_equidistant(hb):
    norms = [np.linalg.norm(k) for k in (hb.kappa, hb.kappa_p, hb.kappa_pp)]
    assert np.allclose(norms, norms[0])


@pytest.mark.parametrize("n_shells, size", [(1, 7), (2, 13), (3, 19)])
def test_basis_size(n_shells, size):
    assert len(HeteroBilayer(n_shells=n_shells).basis) == size


# ------------------------------------------------------------------
# Hamiltonian and bands
# ------------------------------------------------------------------

def test_H_k_is_hermitian(hb):
    H = hb.H_k(np.array([0.3, -0.7]))
    assert np.allclose(H, H.conj().T)


def test_H_k_all_matches_H_k(hb):
    kpts = np.array([hb.gamma, hb.kappa, hb.kappa_p, [0.3, -0.7]])
    H_all = hb.H_k_all(kpts)
    for k, H in zip(kpts, H_all):
        assert np.allclose(H, hb.H_k(k))


def test_free_particle_limit():
    hb = HeteroBilayer(V=0.0, n_shells=3)
    k = np.array([0.4, 0.1])
    bands, _ = hb.calculate_bands(k[None, :])
    expected = -hb.HBAR2_M * np.linalg.norm(k / hb.aM) ** 2 / (2 * hb.MSTAR)
    assert np.isclose(bands[0, 0], expected)


def test_bands_descending_and_eigenvectors(hb):
    kpath, _, _ = hb.make_k_path(*hb.default_path, n_pts=10)
    bands, states = hb.calculate_bands(kpath)
    H_all = hb.H_k_all(kpath)
    assert np.all(np.diff(bands, axis=1) <= 1e-10)
    # H v = E v for the top band
    v = states[:, :, 0]
    assert np.allclose(np.einsum('kij,kj->ki', H_all, v), bands[:, 0, None] * v)


def test_gap_and_bandwidth(hb):
    bands, _ = hb.calculate_bands(np.array([hb.gamma, hb.kappa]))
    assert np.isclose(abs(hb.calculate_gap()), bands[1, 0] - bands[1, 1])
    assert np.isclose(hb.calculate_bandwidth(), bands[0, 0] - bands[1, 0])


# ------------------------------------------------------------------
# k-space and real-space utilities
# ------------------------------------------------------------------

def test_make_k_path(hb):
    path, labels = hb.default_path
    kpath, kdist, ticks = hb.make_k_path(path, labels, n_pts=50)
    assert kpath.shape == (50, 2)
    assert kdist.shape == (50,)
    assert np.allclose(kpath[0], path[0])
    assert np.allclose(kpath[-1], path[-1])
    assert np.all(np.diff(kdist) >= 0)
    assert len(ticks) == len(path)


def test_mesh_k_unit_cell(hb):
    k_points = hb.mesh_k_unit_cell()
    assert k_points.shape == (hb.kmesh ** 2, 2)
    assert np.allclose(k_points[0], 0.0)


def test_moire_potential_symmetries(hb):
    rng = np.random.default_rng(0)
    r = rng.normal(size=(2, 20))
    V0 = hb.moire_potential(*r)
    # lattice periodicity
    assert np.allclose(hb.moire_potential(*(r + hb.a1[:, None])), V0)
    # C3 rotation symmetry
    assert np.allclose(hb.moire_potential(*(utils.rotate(120) @ r)), V0)
