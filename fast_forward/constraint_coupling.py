from . import DATA_PATH
import numpy as np
from vermouth.data import COMMON_CITATIONS
from vermouth.citation_parser import citation_formatter, read_bib
from collections import ChainMap


def compute_coupling(u, constraints):
    '''
    Function to calculate the constraint coupling matrix A used in the LINCS algorithm.
    Adopted from https://github.com/bio-phys/constraint-coupling-analysis

    Parameters
    ----------
    u: :class:`~MDAnalysis.core.universe.Universe`
    constraints: list
        List of constraints in the system

    Returns
    -------
    A: :class:`~numpy.ndarray`
        Constraint coupling matrix
    '''
    
    # sort the constraint list
    for c in constraints:
        if c.atoms[0] > c.atoms[1]:
            c.atoms[0], c.atoms[1] = c.atoms[1], c.atoms[0]
    constraints = sorted(constraints, key=lambda x: (x.atoms[0], x.atoms[1]))

    n_con = len(constraints)
    Sdiag = [np.sqrt(1/u.atoms[c.atoms[0]].mass +1/u.atoms[c.atoms[1]].mass) for c in constraints]
    Sdiag_inv = [1/elem for elem in Sdiag]

    # normalized constraint direction vectors
    B = np.array([u.atoms[c.atoms[0]].position - u.atoms[c.atoms[1]].position for c in constraints])
    B = np.array([v/np.linalg.norm(v) for v in B])

    # CONSTRAINT COUPLING MATRIX
    A = np.zeros((n_con,n_con))
    mass_factor = np.zeros((n_con,n_con))
    # Iterate through pairs of constraints
    for i,c1 in enumerate(constraints):
        for j,c2 in enumerate(constraints):
            # find the common atom between the two constraints
            common = set(tuple(c1.atoms)) & set(tuple(c2.atoms))
            if len(common) != 1: # coupling is zero if no identical atoms of if constraints are identical
                continue
            common = common.pop() # convert set to int
            
            # signs
            if (c1.atoms[0] == c2.atoms[0]) or (c1.atoms[1] == c2.atoms[1]):
                sign = -1
            else:
                sign = +1

            # mass factor
            invmass = 1/u.atoms[common].mass
            mass_factor[i,j] = sign * (invmass*Sdiag_inv[i]*Sdiag_inv[j])

            # constraint couplings
            A[i,j] = mass_factor[i,j] * np.dot(B[i],B[j])
    return A

def report_constraint_coupling(u, block, estimate_lincs=False):
    '''
    Function to check the constraint coupling of a molecule.
    Will print a warning if the constraint coupling is high.

    Parameters
    ----------
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
    A = compute_coupling(u, constraints)

    # check if A contains inf or nan
    if np.isnan(A).any() or np.isinf(A).any():
        print('WARNING: unable to calculate constraint coupling matrix, contains NaN or Inf values.')
        print('     Check the masses of the atoms in the itp.')
        return
    # eigenvalues
    w, _ = np.linalg.eig(A)
    eig_max = np.abs(w).max() # lambda_max

    if eig_max == 0:
        if estimate_lincs:
            print(f"[ Constraint Coupling Report for {block.name} ]\nNo constraints coupling detected.\n")
        return

    # ESTIMATE LINCS ORDER:
    lincs_order = int(np.log(0.4**4)/np.log(eig_max))
    threshold = 0.8

    if estimate_lincs or eig_max > threshold:
        with open(DATA_PATH/'citations.bib') as citation_file:
            ff_citations = read_bib(citation_file)
        buff = '\n'
        buff += f"[ Constraint Coupling Report for {{molname}} ]\n"
        if eig_max >= threshold:
            buff += "WARNING: High constraint coupling detected, consider changing your bonded network!\n\n"

        buff += f"Largest eigenvalue: {{eig_max:.2f}}"
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