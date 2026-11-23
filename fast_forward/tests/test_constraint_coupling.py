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

from fast_forward.constraint_coupling import compute_coupling, report_constraint_coupling
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


def _make_system(block_masses, univ_tri_mass):
    """
    Universe with two single-bead SOL molecules followed by one TRI molecule
    (a bent 3-bead chain), so that block indices of TRI do not coincide
    with universe indices.
    """
    u = mda.Universe.empty(n_atoms=5, n_residues=3,
                           atom_resindex=[0, 1, 2, 2, 2],
                           trajectory=True)
    u.add_TopologyAttr('resnames', ['SOL', 'SOL', 'TRI'])
    u.add_TopologyAttr('names', ['W', 'W', 'A', 'B', 'C'])
    u.add_TopologyAttr('masses', [72.0, 72.0] + [univ_tri_mass] * 3)
    # SOL beads placed so that using them instead of TRI gives a different result
    u.atoms.positions = np.vstack([[[10.0, 10.0, 10.0], [10.0, 13.0, 10.0]], CHAIN])
    res_as_mol(u)

    block = vermouth.molecule.Block(force_field=vermouth.forcefield.ForceField('dummy'),
                                    name='TRI')
    for i, (name, mass) in enumerate(zip('ABC', block_masses)):
        block.add_node(i, atomname=name, resid=1, resname='TRI', mass=mass)
    for a, b in CHAIN_CONSTRAINTS:
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
    u, block = _make_system(block_masses, univ_tri_mass=72.0)

    report_constraint_coupling(u, block, estimate_lincs=True)
    reported = _reported_eigenvalue(capsys)

    expected = _lambda_max(compute_coupling(CHAIN_CONSTRAINTS,
                                            dict(enumerate(block_masses)),
                                            dict(enumerate(CHAIN))))
    # sanity check: universe masses (all 72) would give 0.25
    assert expected != pytest.approx(0.25, abs=0.01)
    assert reported == pytest.approx(expected, abs=0.005)


def test_report_falls_back_to_universe_masses(capsys):
    u, block = _make_system([36.0, 72.0, 54.0], univ_tri_mass=72.0)
    for node in block.nodes:
        del block.nodes[node]['mass']

    report_constraint_coupling(u, block, estimate_lincs=True)
    # equal masses, 120 degrees: (1/m) * (m/2) * |cos(120)| = 0.25
    assert _reported_eigenvalue(capsys) == pytest.approx(0.25, abs=0.005)
