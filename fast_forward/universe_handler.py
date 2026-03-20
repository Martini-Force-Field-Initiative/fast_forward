# Copyright 2021 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
import numpy as np
import networkx as nx
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.core.topologyattrs import Moltypes, Molnums
from pysmiles import PTE
from pysmiles.smiles_helper import correct_aromatic_rings, increment_bond_orders
from fast_forward.hydrogen import BUILD_HYDRO, find_helper_atoms
from tqdm import tqdm

def assign_order(g):
    increment_bond_orders(g)
    for n1, n2, order in g.edges(data='order'):
        if order > 1:
            g.nodes[n1]['aromatic'] = True
            g.nodes[n2]['aromatic'] = True
    correct_aromatic_rings(g)
    return g

def res_as_mol(universe):
    """
    For a universe without moltype/molnum info, promotes residues to molecules.

    Changes universe in place. Does nothing if moltype/molnum info is already
    available.
    """
    if hasattr(universe.atoms, "moltypes"):
        return

    moltypes = Moltypes(universe.residues.resnames)
    molnums = Molnums(range(len(universe.residues)))
    universe.add_TopologyAttr(moltypes)
    universe.add_TopologyAttr(molnums)

class Residue():
    """
    Helper class for a residue.
    """
    def __init__(self, resname, resid):
        self.resname = resname
        self.resid = resid

class ResidueIter():
    """
    Helper class mimicking to iterate over the residues
    of an mdanlaysis universe.
    """
    def __init__(self):
        self.residues = []

    def add_residue(self, resname, resid):
        self.residues.append(Residue(resname, resid))

    def res_iter(self):
        for residue in self.residues:
            yield residue

class UniverseHandler(mda.Universe):
    """
    Wrapper around mda.Universe which allows for
    some smart selections and exposes some useful
    properties.
    """
    def __init__(self, mol_names, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # guess bonds, if missing from the topology.
        try:
            self.bonds
        except mda.NoDataError:
            self.guess_TopologyAttrs(to_guess=['bonds'])

        # select which molecules to treat
        self.mol_names = mol_names

        # we copy residue info as moleculetype info, if it's missing.
        res_as_mol(self)

        self.molecules = self.select_atoms("moltype " + " ".join(mol_names))

        # set some useful attributes
        self.mol_idxs_by_name = defaultdict(list)
        for mol_num, mol_name in dict(zip(self.atoms.molnums, self.atoms.moltypes)).items():
            if mol_name in mol_names:
                self.mol_idxs_by_name[mol_name].append(mol_num)

        mol_idxs_names = [(idx, mol_name) for idx, mol_name in self.mol_idxs_by_name.items()]
        self.normal_order = mol_idxs_names.sort()

        # Hidden attribute labels
        self.__n_atoms = None
        self.__n_residues = None
        self.__pbc_completed = False
        self.__resids = None
        self.__molecule_graphs = None
        self.__mass_to_element = None

    def pbc_complete(self):
        if not self.__pbc_completed:
            self.trajectory.add_transformations(transformations.unwrap(self.atoms))
            self.__pbc_completed = True
        return self.__pbc_completed

    def shift_united_atom_carbons(self, association_dict):
        """
        Given an atomgroup shift it's coordinates
        to where the center of geometry would be if
        hydrogens were included.
        """
        # much faster to make a bonded graph even of the entire system
        bonded_graph = nx.Graph()
        bonded_graph.add_edges_from([ tuple(indices) for indices in self.atoms.get_connections("bonds").indices])
        # select the atoms to be treated
        select_string = "type " + " ".join(association_dict.keys())
        atoms_to_treat = self.select_atoms(select_string)
        # so they can be properly mapped to carbons later, since united-atoms
        # may end up with larger masses.
        atoms_to_treat.masses = PTE['C']['AtomicMass'] 
        for atom in tqdm(atoms_to_treat):
            carbon_type = association_dict[atom.type]
            for ts in self.trajectory:
                helper_atoms = find_helper_atoms(self, atom, carbon_type, bonded_graph)
                hydrogen_coords = BUILD_HYDRO[carbon_type](**helper_atoms)
                new_pos = atom.position
                for hydro_coord in hydrogen_coords:
                    new_pos += hydro_coord
                atom.position = new_pos / (len(hydrogen_coords) + 1)
        return atoms_to_treat.indices

    def generate_molecule_graphs(self):
        """
        Generate molecular graphs from the topology information.
        The molecular graphs contain the elements, bond order,
        and if the molecule is aromatic or charged.
        """
        mol_graphs = {}
        for mol_name, idxs in self.mol_idxs_by_name.items():
            molecule = self.select_atoms(f"molnum {idxs[0]}")
            bonds = [tuple(indices) for indices in molecule.bonds.indices]
            eles = [self.mass_to_element(mass) for mass in molecule.masses]
            lengths = {bond: length/10. for bond, length in zip(bonds, molecule.bonds.values())}
            g = nx.Graph()
            g.add_edges_from(bonds)
            nx.set_edge_attributes(g, lengths, 'length')
            nx.set_node_attributes(g, dict(zip(molecule.indices, eles)) ,'element')
            g=assign_order(g)
            mol_graphs[mol_name] = g
        return mol_graphs

    def mass_to_element(self, mass):
        if self.__mass_to_element is None:
            self.__mass_to_element = {round(PTE[ele]['AtomicMass']): ele for ele in PTE if type(ele)==str}
        try:
            ele = self.__mass_to_element[round(mass)]
        except KeyError:
            raise IOError(f"Did not find element with mass {mass}.")
        return ele

    @property
    def molecule_graphs(self):
        if self.__molecule_graphs is None:
            self.__molecule_graphs = self.generate_molecule_graphs()
        return self.__molecule_graphs

    @property
    def n_atoms(self):
        if self.__n_atoms is None:
            self.__n_atoms = len(self.molecules)
        return self.__n_atoms

    @property
    def resids(self):
        if self.__resids is None:
            self.__resids = self.molecules.resids
        return self.__resids

    @property
    def n_residues(self):
        if self.__n_residues is None:
            self.__n_residues = len(self.molecules.residues)
        return self.__n_residues

    def res_iter(self):
        for residue in self.molecules.residues:
            yield residue
