"""
Tests for fast_forward.constraint_coupling
"""
import re

import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
import numpy as np
import pytest
import vermouth
import vermouth.forcefield
from vermouth.gmx.itp_read import read_itp
from vermouth.molecule import Interaction

from fast_forward.constraint_coupling import (SOURCE_ANGLE, SOURCE_BOND, SOURCE_CONSTRAINTS,
                                              compute_coupling, cosines_from_positions,
                                              coupled_angles, estimate_lincs_order,
                                              report_constraint_coupling, topology_cosines,
                                              trajectory_cosines)
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


def _coupling_from_positions(constraints, masses, positions):
    cosines = cosines_from_positions(coupled_angles(constraints), dict(enumerate(positions)))
    return compute_coupling(constraints, dict(enumerate(masses)), cosines)


def _reference_coupling(constraints, masses, positions):
    """
    Original position-based implementation (explicit LINCS sign rule and
    constraint direction vectors), used to validate the cosine formulation.
    """
    constraints = sorted(tuple(sorted(c)) for c in constraints)
    S = [1/np.sqrt(1/masses[a] + 1/masses[b]) for a, b in constraints]
    B = [(positions[a] - positions[b]) / np.linalg.norm(positions[a] - positions[b])
         for a, b in constraints]
    A = np.zeros((len(constraints), len(constraints)))
    for i, c1 in enumerate(constraints):
        for j, c2 in enumerate(constraints):
            common = set(c1) & set(c2)
            if len(common) != 1:
                continue
            sign = -1 if (c1[0] == c2[0] or c1[1] == c2[1]) else 1
            A[i, j] = sign / masses[common.pop()] * S[i] * S[j] * np.dot(B[i], B[j])
    return A


# --- compute_coupling ---------------------------------------------------------

def test_compute_coupling_equal_mass_triangle():
    # equal masses: |A_ij| = (1/m) * (m/2) * cos(60) = 0.25, lambda_max = 0.5
    A = _coupling_from_positions(TRI_CONSTRAINTS, [72.0] * 3, TRIANGLE)
    assert np.allclose(A, A.T)
    assert np.allclose(np.abs(A[~np.eye(3, dtype=bool)]), 0.25)
    assert _lambda_max(A) == pytest.approx(0.5)


@pytest.mark.parametrize('seed', range(5))
def test_compute_coupling_matches_reference_implementation(seed):
    rng = np.random.default_rng(seed)
    n_atoms = 7
    positions = rng.normal(size=(n_atoms, 3))
    masses = rng.uniform(10, 100, size=n_atoms)
    # random connected network including triangles
    constraints = [(i, i + 1) for i in range(n_atoms - 1)] + [(0, 2), (2, 4), (3, 6), (1, 5)]
    A = _coupling_from_positions(constraints, masses, positions)
    A_ref = _reference_coupling(constraints, masses, positions)
    assert np.allclose(A, A_ref)


def test_compute_coupling_invariant_to_order_and_orientation():
    masses = [36.0, 72.0, 54.0]
    ref = _lambda_max(_coupling_from_positions(TRI_CONSTRAINTS, masses, TRIANGLE))
    shuffled = [(2, 1), (0, 1), (2, 0)]
    assert _lambda_max(_coupling_from_positions(shuffled, masses, TRIANGLE)) == pytest.approx(ref)


def test_compute_coupling_ignores_duplicate_constraints():
    masses = [36.0, 72.0, 54.0]
    ref = _coupling_from_positions(CHAIN_CONSTRAINTS, masses, CHAIN)
    dup = _coupling_from_positions(CHAIN_CONSTRAINTS + [(1, 0)], masses, CHAIN)
    assert np.allclose(ref, dup)


def test_compute_coupling_does_not_modify_input():
    constraints = [(1, 0), (2, 0)]
    _coupling_from_positions(constraints, [72.0] * 3, TRIANGLE)
    assert constraints == [(1, 0), (2, 0)]


def test_compute_coupling_star_exceeds_one():
    A = _coupling_from_positions(STAR_CONSTRAINTS, STAR_MASSES, STAR)
    assert _lambda_max(A) == pytest.approx(1.6)


def test_coupled_angles():
    assert coupled_angles([(1, 0), (1, 2), (3, 4)]) == [(0, 1, 2)]
    # one angle per vertex of the triangle, apex in the middle
    assert coupled_angles(TRI_CONSTRAINTS) == [(0, 1, 2), (0, 2, 1), (1, 0, 2)]


# --- estimate_lincs_order -----------------------------------------------------

@pytest.mark.parametrize('eig_max, expected', [
    (0.4, 4),
    (0.5, 6),     # 5.28 -> rounded up
    (0.8, 17),    # 16.4 -> rounded up
    (0.999, 3664),
    (1.0, None),
    (1 - 1e-12, None),  # floating point noise around 1
    (1 - 3e-8, None),   # float32 coordinate noise around 1
    (1.6, None),
])
def test_estimate_lincs_order(eig_max, expected):
    assert estimate_lincs_order(eig_max) == expected


# --- topology_cosines ---------------------------------------------------------

def _block_from_itp(itp):
    ff = vermouth.forcefield.ForceField('dummy')
    read_itp(itp.splitlines(True), ff)
    return next(iter(ff.blocks.values()))


ITP_HEADER = """
[ moleculetype ]
MOL 1
[ atoms ]
1 P1 1 MOL A 1 0 72
2 P1 1 MOL B 2 0 72
3 P1 1 MOL C 3 0 72
4 P1 1 MOL D 4 0 72
"""


def test_topology_cosines_constraint_triangle():
    block = _block_from_itp(ITP_HEADER + """
[ constraints ]
1 2 1 0.3
2 3 1 0.4
1 3 1 0.5
[ angles ]
; must be ignored, the triangle fixes the angle exactly
1 2 3 1 60 25
""")
    cos = topology_cosines(block, [(0, 1, 2)])
    # 3-4-5 triangle: right angle at atom 2
    assert cos[(0, 1, 2)][0] == pytest.approx(0.0, abs=1e-12)
    assert cos[(0, 1, 2)][1] == SOURCE_CONSTRAINTS


def test_topology_cosines_angle_term_preferred_over_bond():
    block = _block_from_itp(ITP_HEADER + """
[ constraints ]
1 2 1 0.3
2 3 1 0.3
[ bonds ]
1 3 1 0.3 5000
[ angles ]
3 2 1 2 120 25
""")
    cos, source = topology_cosines(block, [(0, 1, 2)])[(0, 1, 2)]
    assert cos == pytest.approx(-0.5)
    assert source == SOURCE_ANGLE


def test_topology_cosines_closing_bond():
    block = _block_from_itp(ITP_HEADER + """
[ constraints ]
1 2 1 0.3
2 3 1 0.3
[ bonds ]
1 3 1 0.3 5000
""")
    cos, source = topology_cosines(block, [(0, 1, 2)])[(0, 1, 2)]
    assert cos == pytest.approx(0.5)
    assert source == SOURCE_BOND


def test_topology_cosines_unresolved_and_non_numeric():
    block = _block_from_itp(ITP_HEADER + """
[ constraints ]
1 2 1 0.3
2 3 1 0.3
3 4 1 0.3
[ angles ]
1 2 3 1 theta_macro 25
2 3 4 8 1 25
""")
    # macro parameter and tabulated angle (func 8) cannot be used
    assert topology_cosines(block, [(0, 1, 2), (1, 2, 3)]) == {}


def test_topology_cosines_fitted_parameters():
    # fitted interactions store numeric (not string) parameters
    block = _block_from_itp(ITP_HEADER)
    block.interactions['constraints'] += [
        Interaction(atoms=[0, 1], parameters=[1, np.float64(0.3)], meta={}),
        Interaction(atoms=[1, 2], parameters=[1, np.float64(0.3)], meta={})]
    block.interactions['angles'].append(
        Interaction(atoms=[0, 1, 2], parameters=[2, np.float64(90.0), 25.0], meta={}))
    cos, source = topology_cosines(block, [(0, 1, 2)])[(0, 1, 2)]
    assert cos == pytest.approx(0.0, abs=1e-12)
    assert source == SOURCE_ANGLE


# --- trajectory_cosines -------------------------------------------------------

def _make_universe(positions, univ_mass=72.0, box=None, frames=None):
    """
    Universe with two single-bead SOL molecules followed by one MOL molecule,
    so that block indices of MOL do not coincide with universe indices.
    `frames` optionally gives several (n_mol, 3) coordinate sets for MOL.
    """
    n_mol = len(positions)
    u = mda.Universe.empty(n_atoms=2 + n_mol, n_residues=3,
                           atom_resindex=[0, 1] + [2] * n_mol,
                           trajectory=True)
    u.add_TopologyAttr('resnames', ['SOL', 'SOL', 'MOL'])
    u.add_TopologyAttr('names', ['W', 'W'] + [f'B{i}' for i in range(n_mol)])
    u.add_TopologyAttr('masses', [72.0, 72.0] + [univ_mass] * n_mol)
    frames = [positions] if frames is None else frames
    # SOL beads placed so that using them instead of MOL gives a different result
    coords = np.array([np.vstack([[[10.0, 10.0, 10.0], [10.0, 13.0, 10.0]], f])
                       for f in frames], dtype=np.float32)
    dims = None if box is None else np.array(list(box) + [90.0, 90.0, 90.0])
    u.load_new(coords, format=MemoryReader, dimensions=dims)
    res_as_mol(u)
    return u


def test_trajectory_cosines_minimum_image():
    # apex (atom 1) at x = 9.5 in a 10 A box; atom 0 belongs at x = 11.0,
    # i.e. 1.5 A further along +x, but is wrapped to x = 1.0
    pos = np.array([[1.0, 5.0, 5.0], [9.5, 5.0, 5.0], [8.0, 7.0, 5.0]])
    # true angle: v1 = (1.5, 0), v2 = (-1.5, 2) -> cos = -2.25 / (1.5 * 2.5) = -0.6
    u = _make_universe(pos, box=[10.0, 10.0, 10.0])
    assert trajectory_cosines(u, 'MOL', [(0, 1, 2)])[(0, 1, 2)] == pytest.approx(-0.6, abs=1e-5)
    # without a box the wrapped molecule gives v1 = (-8.5, 0) -> cos = +0.6
    u_nobox = _make_universe(pos)
    assert trajectory_cosines(u_nobox, 'MOL', [(0, 1, 2)])[(0, 1, 2)] == pytest.approx(0.6, abs=1e-5)


def test_trajectory_cosines_averages_frames_and_restores_frame():
    def bent(theta):
        t = np.radians(theta)
        return np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3 * np.cos(t), 3 * np.sin(t), 0.0]])
    u = _make_universe(bent(90), frames=[bent(90), bent(60), bent(120)])
    u.trajectory[1]
    cos = trajectory_cosines(u, 'MOL', [(0, 1, 2)])
    assert cos[(0, 1, 2)] == pytest.approx((0.0 + 0.5 - 0.5) / 3, abs=1e-5)
    assert u.trajectory.ts.frame == 1


# --- report_constraint_coupling -------------------------------------------------

def _make_block(block_masses, constraints, name='MOL'):
    block = vermouth.molecule.Block(force_field=vermouth.forcefield.ForceField('dummy'),
                                    name=name)
    for i, mass in enumerate(block_masses):
        block.add_node(i, atomname=f'B{i}', resid=1, resname='MOL', mass=mass)
    for a, b in constraints:
        block.interactions['constraints'].append(
            Interaction(atoms=[a, b], parameters=['1', '0.3'], meta={}))
    return block


def _make_system(block_masses, univ_mass, positions=CHAIN, constraints=CHAIN_CONSTRAINTS):
    """Universe plus block without angle terms, so angles come from the trajectory."""
    return _make_universe(positions, univ_mass), _make_block(block_masses, constraints)


def _reported_eigenvalue(out):
    match = re.search(r"Largest eigenvalue: ([0-9.]+)", out)
    assert match, out
    return float(match.group(1))


def test_report_maps_block_to_universe_indices_and_uses_block_masses(capsys):
    block_masses = [36.0, 72.0, 54.0]
    u, block = _make_system(block_masses, univ_mass=72.0)

    report_constraint_coupling(u, block, estimate_lincs=True)
    out = capsys.readouterr().out

    expected = _lambda_max(_coupling_from_positions(CHAIN_CONSTRAINTS, block_masses, CHAIN))
    # sanity check: universe masses (all 72) would give 0.25
    assert expected != pytest.approx(0.25, abs=0.01)
    assert _reported_eigenvalue(out) == pytest.approx(expected, abs=0.005)


def test_report_falls_back_to_universe_masses(capsys):
    u, block = _make_system([36.0, 72.0, 54.0], univ_mass=72.0)
    for node in block.nodes:
        del block.nodes[node]['mass']

    report_constraint_coupling(u, block, estimate_lincs=True)
    # equal masses, 120 degrees: (1/m) * (m/2) * |cos(120)| = 0.25
    assert _reported_eigenvalue(capsys.readouterr().out) == pytest.approx(0.25, abs=0.005)


def test_report_uses_topology_without_trajectory(capsys):
    # angle term in the topology, no universe at all
    block = _make_block([36.0, 72.0, 54.0], CHAIN_CONSTRAINTS)
    block.interactions['angles'].append(
        Interaction(atoms=[0, 1, 2], parameters=['1', '120', '25'], meta={}))
    report_constraint_coupling(None, block, estimate_lincs=True)
    out = capsys.readouterr().out
    expected = _lambda_max(_coupling_from_positions(CHAIN_CONSTRAINTS, [36.0, 72.0, 54.0], CHAIN))
    assert _reported_eigenvalue(out) == pytest.approx(expected, abs=0.005)


def test_report_topology_preferred_over_trajectory(capsys):
    # trajectory has a 120 degree angle, topology says 90 -> no coupling
    u, block = _make_system([72.0] * 3, univ_mass=72.0)
    block.interactions['angles'].append(
        Interaction(atoms=[0, 1, 2], parameters=['1', '90', '25'], meta={}))
    report_constraint_coupling(u, block, estimate_lincs=True)
    out = capsys.readouterr().out
    assert "No coupled constraints detected" in out


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
    # symmetric star: all four constraints contribute equally
    assert "Constraints contributing most to the largest eigenvalue:" in out
    for i in range(1, 5):
        assert f"B0-B{i} (1-{i+1}): 25%" in out


def test_report_dominant_constraints_names_culprits(capsys):
    # high-coupling star (atoms 0-4) plus an isolated, weakly coupled chain (5-7)
    block = _make_block(STAR_MASSES + [72.0] * 3,
                        STAR_CONSTRAINTS + [(5, 6), (6, 7)])
    for i in range(1, 5):
        for j in range(i + 1, 5):
            block.interactions['angles'].append(
                Interaction(atoms=[i, 0, j], parameters=['1', '150', '25'], meta={}))
    block.interactions['angles'].append(
        Interaction(atoms=[5, 6, 7], parameters=['1', '120', '25'], meta={}))
    report_constraint_coupling(None, block)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "B0-B1 (1-2)" in out
    # the chain is not part of the dominant eigenvector
    assert "B5-B6" not in out and "B6-B7" not in out


def test_report_dominant_constraints_below_threshold(capsys):
    u, block = _make_system([72.0] * 3, univ_mass=72.0)
    report_constraint_coupling(u, block, estimate_lincs=True)
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "Constraints contributing most to the largest eigenvalue:" in out
    assert "B0-B1 (1-2): 50%" in out
    assert "B1-B2 (2-3): 50%" in out


def test_report_eigenvalue_at_one(capsys):
    # equal masses in the star give lambda_max = 1.0 (up to rounding),
    # which used to raise OverflowError or report a huge negative order
    u, block = _make_system([72.0] * 5, univ_mass=72.0,
                            positions=STAR, constraints=STAR_CONSTRAINTS)
    report_constraint_coupling(u, block, estimate_lincs=False)
    out = capsys.readouterr().out
    assert "Largest eigenvalue: 1.00" in out
    assert "will not converge" in out


@pytest.mark.parametrize('block_mass, univ_mass, bad', [
    (0.0, 72.0, 'B1 (2)'),        # zero mass in the itp
    (-1.0, 72.0, 'B1 (2)'),       # negative mass in the itp
    (None, 0.0, 'B1 (2)'),        # no mass in the itp, zero in the universe
])
def test_report_invalid_mass_warns(capsys, block_mass, univ_mass, bad):
    u, block = _make_system([36.0, 72.0, 54.0], univ_mass=univ_mass)
    if block_mass is None:
        del block.nodes[1]['mass']
    else:
        block.nodes[1]['mass'] = block_mass
    report_constraint_coupling(u, block, estimate_lincs=True)
    out = capsys.readouterr().out
    assert f"atom(s) {bad} are part of constraints" in out
    assert "Largest eigenvalue" not in out


def test_report_missing_mass_without_trajectory_warns(capsys):
    block = _make_block([36.0, 72.0, 54.0], CHAIN_CONSTRAINTS)
    del block.nodes[2]['mass']
    report_constraint_coupling(None, block, estimate_lincs=True)
    assert "atom(s) B2 (3) are part of constraints" in capsys.readouterr().out


def test_report_molecule_not_in_trajectory_assumes_collinear(capsys):
    u, block = _make_system([72.0] * 3, univ_mass=72.0)
    block.name = 'ABSENT'
    report_constraint_coupling(u, block, estimate_lincs=True)
    out = capsys.readouterr().out
    # equal masses, |cos| = 1: (1/m) * (m/2) = 0.5
    assert _reported_eigenvalue(out) == pytest.approx(0.5, abs=0.005)


def test_report_coincident_atoms_does_not_crash(capsys):
    u, block = _make_system([72.0] * 3, univ_mass=72.0)
    # put MOL atoms 0 and 1 (universe indices 2 and 3) on top of each other;
    # MDAnalysis then reports an angle of 0, i.e. the collinear worst case
    u.atoms[3].position = u.atoms[2].position
    report_constraint_coupling(u, block, estimate_lincs=True)
    assert _reported_eigenvalue(capsys.readouterr().out) == pytest.approx(0.5, abs=0.005)


def test_report_no_coupling(capsys):
    # two separate constraints do not couple
    block = _make_block([72.0] * 4, [(0, 1), (2, 3)])
    report_constraint_coupling(None, block, estimate_lincs=True)
    assert "No coupled constraints detected" in capsys.readouterr().out
