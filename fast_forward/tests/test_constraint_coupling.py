"""
Tests for fast_forward.constraint_coupling
"""
import re

import MDAnalysis as mda
import numpy as np
import pytest
import vermouth
import vermouth.forcefield
from vermouth.molecule import Interaction

from fast_forward.constraint_coupling import (compute_coupling, estimate_lincs_order,
                                              report_constraint_coupling)
from fast_forward.universe_handler import res_as_mol

# equilateral triangle with side length 3 Angstrom
TRIANGLE = np.array([[0.0, 0.0, 0.0],
                     [3.0, 0.0, 0.0],
                     [1.5, 1.5 * np.sqrt(3), 0.0]])
TRI_CONSTRAINTS = [(0, 1), (0, 2), (1, 2)]
# bent 3-bead chain with a 120 degree angle; unlike the equilateral triangle,
# its lambda_max depends on the masses
CHAIN = np.array([[0.0, 0.0, 0.0],
                  [3.0, 0.0, 0.0],
                  [4.5, 1.5 * np.sqrt(3), 0.0]])
CHAIN_CONSTRAINTS = [(0, 1), (1, 2)]
# light central bead constrained to 4 heavy beads on a cone (half-angle 30 deg)
# around z. Off-diagonal dot products per row sum to 0.75 + 0.75 + 0.5 = 2, so
# lambda_max = 2 * M / (M + m): 1.6 for m = 18, M = 72, and 1.0 for equal masses
_HALF = np.radians(30)
STAR = np.vstack([[0.0, 0.0, 0.0]] +
                 [3.0 * np.array([np.sin(_HALF) * np.cos(phi),
                                  np.sin(_HALF) * np.sin(phi),
                                  np.cos(_HALF)])
                  for phi in np.radians([0, 90, 180, 270])])
STAR_CONSTRAINTS = [(0, i) for i in range(1, 5)]
STAR_MASSES = [18.0] + [72.0] * 4


def _lambda_max(A):
    return np.abs(np.linalg.eigvalsh(A)).max()


def test_compute_coupling_equal_mass_triangle():
    # equal masses: |A_ij| = (1/m) * (m/2) * cos(60) = 0.25, lambda_max = 0.5
    masses = {i: 72.0 for i in range(3)}
    positions = dict(enumerate(TRIANGLE))
    A = compute_coupling(TRI_CONSTRAINTS, masses, positions)
    assert np.allclose(A, A.T)
    assert np.allclose(np.abs(A[~np.eye(3, dtype=bool)]), 0.25)
    assert _lambda_max(A) == pytest.approx(0.5)


def test_compute_coupling_invariant_to_order_and_orientation():
    masses = {0: 36.0, 1: 72.0, 2: 54.0}
    positions = dict(enumerate(TRIANGLE))
    ref = _lambda_max(compute_coupling(TRI_CONSTRAINTS, masses, positions))
    shuffled = [(2, 1), (0, 1), (2, 0)]
    assert _lambda_max(compute_coupling(shuffled, masses, positions)) == pytest.approx(ref)


def test_compute_coupling_does_not_modify_input():
    constraints = [(1, 0), (2, 0)]
    compute_coupling(constraints, {i: 72.0 for i in range(3)}, dict(enumerate(TRIANGLE)))
    assert constraints == [(1, 0), (2, 0)]


def _make_system(block_masses, univ_mass, positions=CHAIN, constraints=CHAIN_CONSTRAINTS):
    """
    Universe with two single-bead SOL molecules followed by one MOL molecule
    (by default a bent 3-bead chain), so that block indices of MOL do not
    coincide with universe indices.
    """
    n_mol = len(positions)
    u = mda.Universe.empty(n_atoms=2 + n_mol, n_residues=3,
                           atom_resindex=[0, 1] + [2] * n_mol,
                           trajectory=True)
    u.add_TopologyAttr('resnames', ['SOL', 'SOL', 'MOL'])
    u.add_TopologyAttr('names', ['W', 'W'] + [f'B{i}' for i in range(n_mol)])
    u.add_TopologyAttr('masses', [72.0, 72.0] + [univ_mass] * n_mol)
    # SOL beads placed so that using them instead of MOL gives a different result
    u.atoms.positions = np.vstack([[[10.0, 10.0, 10.0], [10.0, 13.0, 10.0]], positions])
    res_as_mol(u)

    block = vermouth.molecule.Block(force_field=vermouth.forcefield.ForceField('dummy'),
                                    name='MOL')
    for i, mass in enumerate(block_masses):
        block.add_node(i, atomname=f'B{i}', resid=1, resname='MOL', mass=mass)
    for a, b in constraints:
        block.interactions['constraints'].append(
            Interaction(atoms=[a, b], parameters=['1', '0.3'], meta={}))
    return u, block


def _reported_eigenvalue(capsys):
    out = capsys.readouterr().out
    match = re.search(r"Largest eigenvalue: ([0-9.]+)", out)
    assert match, out
    return float(match.group(1))


def test_report_maps_block_to_universe_indices_and_uses_block_masses(capsys):
    block_masses = [36.0, 72.0, 54.0]
    u, block = _make_system(block_masses, univ_mass=72.0)

    report_constraint_coupling(u, block, estimate_lincs=True)
    reported = _reported_eigenvalue(capsys)

    expected = _lambda_max(compute_coupling(CHAIN_CONSTRAINTS,
                                            dict(enumerate(block_masses)),
                                            dict(enumerate(CHAIN))))
    # sanity check: universe masses (all 72) would give 0.25
    assert expected != pytest.approx(0.25, abs=0.01)
    assert reported == pytest.approx(expected, abs=0.005)


def test_report_falls_back_to_universe_masses(capsys):
    u, block = _make_system([36.0, 72.0, 54.0], univ_mass=72.0)
    for node in block.nodes:
        del block.nodes[node]['mass']

    report_constraint_coupling(u, block, estimate_lincs=True)
    # equal masses, 120 degrees: (1/m) * (m/2) * |cos(120)| = 0.25
    assert _reported_eigenvalue(capsys) == pytest.approx(0.25, abs=0.005)


@pytest.mark.parametrize('eig_max, expected', [
    (0.4, 4),
    (0.5, 6),     # 5.28 -> rounded up
    (0.8, 17),    # 16.4 -> rounded up
    (0.999, 3664),
    (1.0, None),
    (1 - 1e-12, None),  # floating point noise around 1
    (1.6, None),
])
def test_estimate_lincs_order(eig_max, expected):
    assert estimate_lincs_order(eig_max) == expected


def test_compute_coupling_star_exceeds_one():
    A = compute_coupling(STAR_CONSTRAINTS, dict(enumerate(STAR_MASSES)), dict(enumerate(STAR)))
    assert _lambda_max(A) == pytest.approx(1.6)


@pytest.mark.parametrize('estimate_lincs', [True, False])
def test_report_eigenvalue_above_one(capsys, estimate_lincs):
    u, block = _make_system(STAR_MASSES, univ_mass=72.0,
                            positions=STAR, constraints=STAR_CONSTRAINTS)
    # must not raise and must warn even without -lincs
    report_constraint_coupling(u, block, estimate_lincs=estimate_lincs)
    out = capsys.readouterr().out
    assert "Largest eigenvalue: 1.60" in out
    assert "will not converge" in out
    assert "Estimated LINCS order: none" in out
    assert "Estimated LINCS order: -" not in out


def test_report_eigenvalue_at_one(capsys):
    # equal masses in the star give lambda_max = 1.0 (up to rounding),
    # which used to raise OverflowError or report a huge negative order
    u, block = _make_system([72.0] * 5, univ_mass=72.0,
                            positions=STAR, constraints=STAR_CONSTRAINTS)
    report_constraint_coupling(u, block, estimate_lincs=False)
    out = capsys.readouterr().out
    assert "Largest eigenvalue: 1.00" in out
    assert "will not converge" in out
