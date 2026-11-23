from . import DATA_PATH
from .itp_to_ag import find_mol_indices
import numpy as np
from vermouth.data import COMMON_CITATIONS
from vermouth.citation_parser import citation_formatter, read_bib
from collections import ChainMap


def compute_coupling(constraints, masses, positions):
    '''
    Function to calculate the constraint coupling matrix A used in the LINCS algorithm.
    Adopted from https://github.com/bio-phys/constraint-coupling-analysis

    Parameters
    ----------
    constraints: list
        List of (atom_a, atom_b) tuples, one per constraint. Atom keys
        are arbitrary hashable identifiers used to look up `masses`
        and `positions`.
    masses: dict
        Mass of each atom, keyed by atom.
    positions: dict
        Position (array of shape (3,)) of each atom, keyed by atom.

    Returns
    -------
    A: :class:`~numpy.ndarray`
        Constraint coupling matrix
    '''
    # sort the constraint list (without modifying the input)
    constraints = sorted(tuple(sorted(c)) for c in constraints)

    n_con = len(constraints)
    Sdiag = [np.sqrt(1/masses[a] + 1/masses[b]) for a, b in constraints]
    Sdiag_inv = [1/elem for elem in Sdiag]

    # normalized constraint direction vectors
    B = np.array([np.asarray(positions[a]) - np.asarray(positions[b]) for a, b in constraints])
    B = np.array([v/np.linalg.norm(v) for v in B])

    # CONSTRAINT COUPLING MATRIX
    A = np.zeros((n_con,n_con))
    mass_factor = np.zeros((n_con,n_con))
    # Iterate through pairs of constraints
    for i,c1 in enumerate(constraints):
        for j,c2 in enumerate(constraints):
            # find the common atom between the two constraints
            common = set(c1) & set(c2)
            if len(common) != 1: # coupling is zero if no identical atoms of if constraints are identical
                continue
            common = common.pop()

            # signs
            if (c1[0] == c2[0]) or (c1[1] == c2[1]):
                sign = -1
            else:
                sign = +1

            # mass factor
            invmass = 1/masses[common]
            mass_factor[i,j] = sign * (invmass*Sdiag_inv[i]*Sdiag_inv[j])

            # constraint couplings
            A[i,j] = mass_factor[i,j] * np.dot(B[i],B[j])
    return A

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
    # treat values within floating point noise of 1 as 1
    if eig_max >= 1 - 1e-8:
        return None
    # round up (with a small tolerance for floating point noise) so the
    # order is never underestimated
    return int(np.ceil(np.log(0.4**4)/np.log(eig_max) - 1e-9))

def report_constraint_coupling(u, block, estimate_lincs=False):
    '''
    Function to check the constraint coupling of a molecule.
    Will print a warning if the constraint coupling is high.

    Masses are taken from the block (i.e. the topology that is written out),
    falling back to the universe masses for atoms without a mass in the block.
    Positions are taken from the first molecule of type `block.name` in the
    universe at the current frame.

    Parameters
    ----------
    u: :class:`~MDAnalysis.core.universe.Universe`
        Universe containing the molecule(s); must have molecule types
        (e.g. via :func:`~fast_forward.itp_to_ag.res_as_mol`).
    block: :class:`~vermouth.molecule.Block`
        Block containing the molecule of interest.
    estimate_lincs: bool
        optional.
        Whether to report the estimated LINCS order required for the constraints, defaults to False
    '''

    constraints = block.interactions['constraints']

    if len(constraints) <= 0:
        if estimate_lincs:
            print(f"[ Constraint Coupling Report for {block.name} ]\nNo constraints coupling detected.\n")
        return

    if len(u.select_atoms(f'moltype {block.name}')) == 0:
        print(f"WARNING: cannot compute constraint coupling for {block.name}: "
              "molecule not found in the trajectory. Skipping.\n")
        return

    pairs = [tuple(c.atoms) for c in constraints]
    nodes = sorted({atom for pair in pairs for atom in pair})
    # map block atoms onto the universe indices of the first molecule of this type
    univ_indices = find_mol_indices(u, nodes, block.name)[0]
    node_to_univ = dict(zip(nodes, univ_indices))

    positions = {node: u.atoms[idx].position for node, idx in node_to_univ.items()}
    masses = {}
    for node, idx in node_to_univ.items():
        mass = block.nodes[node].get('mass')
        masses[node] = float(mass) if mass is not None else float(u.atoms[idx].mass)

    # LINCS needs a finite, positive mass on every constrained atom
    bad_atoms = [node for node, mass in masses.items()
                 if not np.isfinite(mass) or mass <= 0]
    if bad_atoms:
        # report 1-based atom numbers as in the itp
        atom_list = ', '.join(str(node + 1) for node in bad_atoms)
        print(f"WARNING: cannot compute constraint coupling for {block.name}: "
              f"atom(s) {atom_list} are part of constraints but have a zero, negative "
              "or missing mass.\n"
              "         Constraints on massless particles are not supported by LINCS; "
              "check the masses in the itp.\n")
        return

    with np.errstate(invalid='ignore', divide='ignore'):
        A = compute_coupling(pairs, masses, positions)

    # masses are checked above, so NaN/Inf can only come from the geometry
    if not np.isfinite(A).all():
        print(f"WARNING: cannot compute constraint coupling for {block.name}: "
              "coupling matrix contains NaN or Inf values.\n"
              "         Check for constrained atoms at identical positions in the trajectory.\n")
        return
    # eigenvalues
    # A is symmetric, so eigvalsh gives real eigenvalues
    w = np.linalg.eigvalsh(A)
    eig_max = np.abs(w).max() # lambda_max

    if eig_max == 0:
        if estimate_lincs:
            print(f"[ Constraint Coupling Report for {block.name} ]\nNo constraints coupling detected.\n")
        return

    lincs_order = estimate_lincs_order(eig_max)
    threshold = 0.8

    if estimate_lincs or eig_max >= threshold:
        with open(DATA_PATH/'citations.bib') as citation_file:
            ff_citations = read_bib(citation_file)
        buff = '\n'
        buff += f"[ Constraint Coupling Report for {{molname}} ]\n"
        if lincs_order is None:
            buff += ("WARNING: Constraint coupling eigenvalue >= 1, the LINCS expansion "
                     "will not converge for any lincs-order!\n"
                     "         Constraints will not be satisfied accurately, "
                     "change your bonded network!\n\n")
        elif eig_max >= threshold:
            buff += "WARNING: High constraint coupling detected, consider changing your bonded network!\n\n"

        buff += f"Largest eigenvalue: {{eig_max:.2f}}"
        if lincs_order is None:
            buff += "\nEstimated LINCS order: none (LINCS does not converge)\n\n"
        else:
            buff += f"\nEstimated LINCS order: {{lincs_order}}\n\n"
        buff += "For more information see:\n"
        citations = {'cholesterol_constraints'}
        citation_map = ChainMap(ff_citations, COMMON_CITATIONS)
        for citation in citations:
            cite_string = citation_formatter(
                citation_map[citation]
            )
            buff += (cite_string) + "\n"

        printable = buff.format(molname=block.name,
                        eig_max=eig_max,
                        lincs_order=lincs_order)
        print(printable)