"""
Extract interactions pairs from itp and convert to atom
groups of an MDAnalysis Universe.
"""
from collections import defaultdict
import numpy as np
import networkx as nx
from fast_forward.universe_handler import res_as_mol

def find_mol_indices(universe, atoms, moltype, atomnames=None, match_names=False):
    """
    Given a universe select all atoms that belong to molecules 
    of the given `moltype`. Subsequently, return indices of all
    multiples of the indices defined in atoms under the assumption
    that all considered molecules have the same number of atoms.

    Parameters
    ----------
    universe: mda.Universe
    atoms: list[int]
    moltype: str

    Returns
    -------
    list[numpy.array(dtype=int)]
    """
    mol_atoms = universe.select_atoms(f'moltype {moltype}')
    n_mols = len(np.unique(mol_atoms.molnums))

    try:
        mol_atom_indices = mol_atoms.indices.reshape(n_mols, -1)
    except ValueError:
        msg = ("The target molecules passed to find_mol_indices "
               "do not seem to all have the same number of atoms.")
        raise IndexError(msg) from None

    if match_names:
        target_atoms = universe.atoms.names[mol_atoms].reshape(n_mols, 1)
        order = [np.where(target_atoms == r)[0][0] for r in ref]
        mapped_indices = mol_atom_indices[:, atoms][order]
        return mapped_indices
    else:
        return list(mol_atom_indices[:, atoms])

class ITPInteractionMapper:
    """
    Class to extract interaction groups from itp files
    and map them to indices in an MDAnalysis Universe.
    """
    def __init__(self, universe, blocks, molnames):
        """
        Parameters
        ----------
        universe: mda.Universe
        blocks: list[vermouth.molecule.Block]
        molnames: list[str]
        """
        self.universe = universe
        self.blocks = dict(zip(molnames, blocks))
        # we ensure we either have molecule types, or we promote res info as such
        res_as_mol(self.universe)

    def get_interactions_group(self, molname, itp_mode=False):
        """
        Iterate over interactions in itp file and return dict of
        grouped indices corresponding to the atoms in universe.
        """
        block = self.blocks[molname]

        indices_dict = defaultdict(dict)
        initial_parameters = defaultdict(dict)
        block_indices = defaultdict(dict)
        for inter_type in block.interactions:
            for inter in block.interactions[inter_type]:
                atoms = inter.atoms
                atomnames=[block.nodes[atom]['atomname'] for atom in atoms]
                if itp_mode == "all":
                    group = "_".join(atomnames)
                    inter.meta["comment"] = group
                else:
                    group = inter.meta.get("comment", None)
                if group:
                    indices = find_mol_indices(self.universe, atoms, molname)
                    if inter_type == 'constraints': # treat constraints as bonds
                        inter_type = 'bonds'
                    old_indices = indices_dict[inter_type].get(group, [])
                    old_block_indices = block_indices[inter_type].get(group, [])
                    block_indices[inter_type][group] = [atoms] + old_block_indices

                    indices_dict[inter_type][group] = indices + old_indices
                    initial_parameters[inter_type][group] = inter.parameters

        return indices_dict, initial_parameters, block_indices

    def get_pairwise_interaction(self, molname):
        """
        Iterate over all atoms in itp file and return dict of
        paired indices corresponding to all pairwise distances in universe.
        group_names are given by the atomnames
        """
        block = self.blocks[molname]

        indices_dict = defaultdict(dict)

        for node1, name1 in block.nodes(data='atomname'):
            for node2, name2 in list(block.nodes(data='atomname'))[node1+1:]:
                atoms = np.array([node1, node2])
                group = f'{name1}_{name2}' # naming convention with node1 < node2
                indices = find_mol_indices(self.universe, atoms, molname)
                old_indices = indices_dict['distances'].get(group, [])
                indices_dict['distances'][group] = indices + old_indices

        return indices_dict
