"""
Constraint coupling analysis, following the LINCS coupling matrix
(Hess et al. 1997) as applied by Fabian et al. 2023.

The coupling between two constraints a-c and c-b that share atom c only
depends on the masses and on the angle theta at c:

    A_ij = -(1/m_c) * S_i * S_j * cos(theta_acb),  S_i = 1/sqrt(1/m_a + 1/m_c)

The angles are taken from the topology where possible (constraint triangles,
angle terms, closing bonds), and only otherwise from the trajectory.
"""
from collections import ChainMap

import numpy as np
from MDAnalysis.lib.distances import calc_angles
from vermouth.citation_parser import citation_formatter, read_bib
from vermouth.data import COMMON_CITATIONS

from . import DATA_PATH
from .itp_to_ag import find_mol_indices

# GROMACS function types whose first parameter is an equilibrium length (nm)
CONSTRAINT_FUNCS = {1, 2}
BOND_FUNCS = {1, 2, 6}
# GROMACS function types whose first parameter is an equilibrium angle (deg)
ANGLE_FUNCS = {1, 2, 5, 10}

# where the angle of a coupled constraint pair came from
SOURCE_CONSTRAINTS = 'constraint triangle'
SOURCE_ANGLE = 'angle term'
SOURCE_BOND = 'closing bond'
SOURCE_TRAJECTORY = 'trajectory average'
SOURCE_ASSUMED = 'assumed collinear'


def _normalise_constraints(constraints):
    """Sorted, de-duplicated list of (a, b) tuples with a < b."""
    return sorted({tuple(sorted(c)) for c in constraints})


def _triple_key(a, c, b):
    """Angle key with the apex in the middle and the outer atoms sorted."""
    return (min(a, b), c, max(a, b))


def coupled_angles(constraints):
    '''
    Find all angles a-c-b spanned by two constraints a-c and c-b that share
    exactly one atom c. These are the only pairs with nonzero coupling.

    Parameters
    ----------
    constraints: list
        List of (atom_a, atom_b) tuples.

    Returns
    -------
    list
        Sorted list of (a, c, b) tuples with a < b.
    '''
    constraints = _normalise_constraints(constraints)
    triples = set()
    for i, c1 in enumerate(constraints):
        for c2 in constraints[i+1:]:
            common = set(c1) & set(c2)
            if len(common) != 1:
                continue
            c = common.pop()
            a = c1[0] if c1[1] == c else c1[1]
            b = c2[0] if c2[1] == c else c2[1]
            triples.add(_triple_key(a, c, b))
    return sorted(triples)


def cosines_from_positions(triples, positions):
    '''
    Cosines of the angles a-c-b (apex c) from a set of positions.

    Parameters
    ----------
    triples: list
        List of (a, c, b) tuples.
    positions: dict
        Position (array of shape (3,)) of each atom, keyed by atom.

    Returns
    -------
    dict
        cos(theta) keyed by (a, c, b).
    '''
    cosines = {}
    for a, c, b in triples:
        v1 = np.asarray(positions[a]) - np.asarray(positions[c])
        v2 = np.asarray(positions[b]) - np.asarray(positions[c])
        cosines[_triple_key(a, c, b)] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cosines


def compute_coupling(constraints, masses, cosines):
    '''
    Function to calculate the constraint coupling matrix A used in the LINCS algorithm.
    Adopted from https://github.com/bio-phys/constraint-coupling-analysis

    Parameters
    ----------
    constraints: list
        List of (atom_a, atom_b) tuples, one per constraint. Duplicates are
        ignored. Atom keys are arbitrary hashable identifiers used to look
        up `masses` and `cosines`.
    masses: dict
        Mass of each atom, keyed by atom.
    cosines: dict
        cos(theta) of the angle a-c-b between every pair of constraints a-c
        and c-b, keyed by (a, c, b) with a < b (see :func:`coupled_angles`).

    Returns
    -------
    A: :class:`~numpy.ndarray`
        Constraint coupling matrix
    '''
    constraints = _normalise_constraints(constraints)

    n_con = len(constraints)
    S = [1/np.sqrt(1/masses[a] + 1/masses[b]) for a, b in constraints]

    A = np.zeros((n_con, n_con))
    for i, c1 in enumerate(constraints):
        for j, c2 in enumerate(constraints):
            common = set(c1) & set(c2)
            # coupling is zero if there is no shared atom or for the diagonal
            if len(common) != 1:
                continue
            c = common.pop()
            a = c1[0] if c1[1] == c else c1[1]
            b = c2[0] if c2[1] == c else c2[1]
            # the sign convention of LINCS and the orientation of the
            # constraint vectors cancel, leaving -cos(theta) at the shared atom
            A[i, j] = -(1/masses[c]) * S[i] * S[j] * cosines[_triple_key(a, c, b)]
    return A


def _float_param(interaction, index):
    """Parameter `index` of an interaction as float, or None if not numeric."""
    try:
        return float(interaction.parameters[index])
    except (IndexError, TypeError, ValueError):
        return None


def _func_type(interaction):
    try:
        return int(interaction.parameters[0])
    except (IndexError, TypeError, ValueError):
        return None


def _equilibrium_lengths(block, inter_type, funcs):
    """Equilibrium lengths keyed by frozenset of the two atoms."""
    lengths = {}
    for inter in block.interactions.get(inter_type, []):
        if len(inter.atoms) != 2 or _func_type(inter) not in funcs:
            continue
        length = _float_param(inter, 1)
        if length is not None and length > 0:
            lengths.setdefault(frozenset(inter.atoms), length)
    return lengths


def _law_of_cosines(d_ac, d_cb, d_ab):
    return np.clip((d_ac**2 + d_cb**2 - d_ab**2) / (2 * d_ac * d_cb), -1, 1)


def topology_cosines(block, triples):
    '''
    Determine the cosines of angles a-c-b from the topology alone.

    In order of preference:

    1. constraint triangle: a-b is constrained as well, so the angle is fixed
       exactly by the three constraint lengths;
    2. angle term a-c-b: cos of its equilibrium angle;
    3. closing bond a-b: law of cosines with its equilibrium length.

    Parameters
    ----------
    block: :class:`~vermouth.molecule.Block`
    triples: list
        List of (a, c, b) tuples, where a-c and c-b are constraints.

    Returns
    -------
    dict
        (cos(theta), source) keyed by (a, c, b), only for the triples that
        could be determined.
    '''
    constraint_lengths = _equilibrium_lengths(block, 'constraints', CONSTRAINT_FUNCS)
    bond_lengths = _equilibrium_lengths(block, 'bonds', BOND_FUNCS)
    angles = {}
    for inter in block.interactions.get('angles', []):
        if len(inter.atoms) != 3 or _func_type(inter) not in ANGLE_FUNCS:
            continue
        theta0 = _float_param(inter, 1)
        if theta0 is not None:
            angles.setdefault(_triple_key(*inter.atoms), theta0)

    result = {}
    for triple in triples:
        a, c, b = triple
        d_ac = constraint_lengths.get(frozenset((a, c)))
        d_cb = constraint_lengths.get(frozenset((c, b)))
        sides_known = d_ac is not None and d_cb is not None
        if sides_known and frozenset((a, b)) in constraint_lengths:
            cos = _law_of_cosines(d_ac, d_cb, constraint_lengths[frozenset((a, b))])
            result[triple] = (cos, SOURCE_CONSTRAINTS)
        elif triple in angles:
            result[triple] = (np.cos(np.radians(angles[triple])), SOURCE_ANGLE)
        elif sides_known and frozenset((a, b)) in bond_lengths:
            cos = _law_of_cosines(d_ac, d_cb, bond_lengths[frozenset((a, b))])
            result[triple] = (cos, SOURCE_BOND)
    return result


def trajectory_cosines(u, molname, triples):
    '''
    Average cosines of angles a-c-b over all frames and all molecules of type
    `molname` in the universe, using minimum image convention.

    Parameters
    ----------
    u: :class:`~MDAnalysis.core.universe.Universe`
    molname: str
    triples: list
        List of (a, c, b) tuples of block atom indices.

    Returns
    -------
    dict
        Mean cos(theta) keyed by (a, c, b); triples for which no valid angle
        could be computed are left out.
    '''
    if not triples:
        return {}
    flat = [atom for triple in triples for atom in triple]
    # (n_molecules, n_triples, 3) universe indices
    idx = np.array(find_mol_indices(u, flat, molname)).reshape(-1, len(triples), 3)
    apex_first = idx[:, :, 0].ravel(), idx[:, :, 1].ravel(), idx[:, :, 2].ravel()

    cos_sum = np.zeros(len(triples))
    count = np.zeros(len(triples))
    start_frame = u.trajectory.ts.frame
    for ts in u.trajectory:
        pos = u.atoms.positions
        box = ts.dimensions
        theta = calc_angles(pos[apex_first[0]], pos[apex_first[1]], pos[apex_first[2]],
                            box=box if box is not None and np.all(box[:3] > 0) else None)
        cos = np.cos(theta).reshape(-1, len(triples))
        valid = np.isfinite(cos)
        cos_sum += np.where(valid, cos, 0).sum(axis=0)
        count += valid.sum(axis=0)
    u.trajectory[start_frame]

    return {triple: cos_sum[i] / count[i]
            for i, triple in enumerate(triples) if count[i] > 0}


def estimate_lincs_order(eig_max):
    '''
    Estimate the LINCS expansion order needed for a coupling matrix with
    largest absolute eigenvalue `eig_max`, i.e. the smallest order at which the
    expansion error eig_max**order matches that of the GROMACS default
    (lincs-order 4) for a typical eigenvalue of 0.4.

    Parameters
    ----------
    eig_max: float
        Largest absolute eigenvalue of the constraint coupling matrix (> 0).

    Returns
    -------
    int or None
        Estimated LINCS order, or None if eig_max >= 1, in which case the
        LINCS expansion does not converge for any order.
    '''
    # treat values within noise of 1 as 1: trajectory angles come from float32
    # coordinates (~1e-7 precision), and an eigenvalue of 1 - 1e-6 would need
    # a lincs-order of ~5e6 anyway
    if eig_max >= 1 - 1e-6:
        return None
    # round up (with a small tolerance for floating point noise) so the
    # order is never underestimated
    return int(np.ceil(np.log(0.4**4)/np.log(eig_max) - 1e-9))


def _format_triples(triples):
    # 1-based atom numbers as in the itp
    return ', '.join('-'.join(str(atom + 1) for atom in triple) for triple in triples)


def report_constraint_coupling(u, block, estimate_lincs=False):
    '''
    Function to check the constraint coupling of a molecule.
    Will print a warning if the constraint coupling is high.

    Masses are taken from the block (i.e. the topology that is written out),
    falling back to the universe masses for atoms without a mass in the block.
    The angles between coupled constraints are taken from the topology (see
    :func:`topology_cosines`). Angles not determined by the topology are
    averaged over the trajectory (see :func:`trajectory_cosines`), or assumed
    collinear (|cos| = 1) if the molecule is not in the trajectory.

    Parameters
    ----------
    u: :class:`~MDAnalysis.core.universe.Universe` or None
        Universe containing the molecule(s); must have molecule types
        (e.g. via :func:`~fast_forward.itp_to_ag.res_as_mol`).
    block: :class:`~vermouth.molecule.Block`
        Block containing the molecule of interest.
    estimate_lincs: bool
        optional.
        Whether to report the estimated LINCS order required for the constraints, defaults to False
    '''
    molname = block.name
    no_coupling_msg = f"[ Constraint Coupling Report for {molname} ]\nNo coupled constraints detected.\n"

    pairs = _normalise_constraints(c.atoms for c in block.interactions['constraints'])
    if not pairs:
        if estimate_lincs:
            print(no_coupling_msg)
        return

    in_universe = u is not None and len(u.select_atoms(f'moltype {molname}')) > 0
    nodes = sorted({atom for pair in pairs for atom in pair})
    node_to_univ = {}
    if in_universe:
        # universe indices of the first molecule of this type
        node_to_univ = dict(zip(nodes, find_mol_indices(u, nodes, molname)[0]))

    masses = {}
    for node in nodes:
        mass = block.nodes[node].get('mass')
        if mass is None and node in node_to_univ:
            mass = u.atoms[node_to_univ[node]].mass
        masses[node] = float(mass) if mass is not None else np.nan

    # LINCS needs a finite, positive mass on every constrained atom
    bad_atoms = [node for node, mass in masses.items()
                 if not np.isfinite(mass) or mass <= 0]
    if bad_atoms:
        atom_list = ', '.join(str(node + 1) for node in bad_atoms)
        print(f"WARNING: cannot compute constraint coupling for {molname}: "
              f"atom(s) {atom_list} are part of constraints but have a zero, negative "
              "or missing mass.\n"
              "         Constraints on massless particles are not supported by LINCS; "
              "check the masses in the itp.\n")
        return

    triples = coupled_angles(pairs)
    if not triples:
        if estimate_lincs:
            print(no_coupling_msg)
        return

    # angles: topology first, then trajectory, then worst case
    resolved = topology_cosines(block, triples)
    missing = [t for t in triples if t not in resolved]
    if missing and in_universe:
        for triple, cos in trajectory_cosines(u, molname, missing).items():
            resolved[triple] = (cos, SOURCE_TRAJECTORY)
    for triple in triples:
        resolved.setdefault(triple, (1.0, SOURCE_ASSUMED))

    cosines = {triple: cos for triple, (cos, _) in resolved.items()}
    A = compute_coupling(pairs, masses, cosines)

    # A is symmetric, so eigvalsh gives real eigenvalues
    eig_max = np.abs(np.linalg.eigvalsh(A)).max()
    # e.g. cos(90 deg) is ~6e-17, not exactly 0
    if eig_max < 1e-8:
        if estimate_lincs:
            print(no_coupling_msg)
        return

    lincs_order = estimate_lincs_order(eig_max)
    threshold = 0.8
    if not (estimate_lincs or eig_max >= threshold):
        return

    lines = ['', f"[ Constraint Coupling Report for {molname} ]"]
    if lincs_order is None:
        lines += ["WARNING: Constraint coupling eigenvalue >= 1, the LINCS expansion "
                  "will not converge for any lincs-order!",
                  "         Constraints will not be satisfied accurately, "
                  "change your bonded network!", '']
    elif eig_max >= threshold:
        lines += ["WARNING: High constraint coupling detected, consider changing "
                  "your bonded network!", '']

    lines.append(f"Largest eigenvalue: {eig_max:.2f}")
    if lincs_order is None:
        lines.append("Estimated LINCS order: none (LINCS does not converge)")
    else:
        lines.append(f"Estimated LINCS order: {lincs_order}")

    # where the angles between coupled constraints came from
    by_source = {}
    for triple, (_, source) in resolved.items():
        by_source.setdefault(source, []).append(triple)
    order = [SOURCE_CONSTRAINTS, SOURCE_ANGLE, SOURCE_BOND, SOURCE_TRAJECTORY, SOURCE_ASSUMED]
    lines.append("Angles between coupled constraints: " +
                 ', '.join(f"{len(by_source[s])} from {s}" for s in order if s in by_source))
    if SOURCE_TRAJECTORY in by_source:
        lines.append(f"  not defined by the topology, averaged over the trajectory: "
                     f"{_format_triples(sorted(by_source[SOURCE_TRAJECTORY]))}")
    if SOURCE_ASSUMED in by_source:
        lines.append(f"  NOTE: not defined by the topology or trajectory, assumed collinear "
                     f"(|cos| = 1, likely an overestimate): "
                     f"{_format_triples(sorted(by_source[SOURCE_ASSUMED]))}")

    with open(DATA_PATH/'citations.bib') as citation_file:
        ff_citations = read_bib(citation_file)
    citation_map = ChainMap(ff_citations, COMMON_CITATIONS)
    lines += ['', "For more information see:"]
    lines += [citation_formatter(citation_map[c]) for c in ['cholesterol_constraints']]
    print('\n'.join(lines) + '\n')
